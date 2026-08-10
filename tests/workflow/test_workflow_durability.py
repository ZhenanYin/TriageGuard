from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from threading import Event
from typing import Any, Self

import pytest

from triageguard.config import Settings
from triageguard.domain import (
    DifferentialEvidence,
    RuntimeObservation,
    WorkflowStatus,
)
from triageguard.evidence import classify_differential
from triageguard.execution import ExecutionArtifacts, MissingObservationError
from triageguard.llm import ReplayGateway
from triageguard.provenance import canonical_sha256
from triageguard.research import ArtifactRecorder, RunHandle
from triageguard.runtime import RuntimeObservationEnvelope
from triageguard.workflow import (
    InterruptedExternalOperationError,
    MilestoneOneWorkflow,
    WorkflowTransitionError,
    resume_replay_workflow,
)
from triageguard.workflow.vertical_slice import ExecutionManifestError

FIXTURE_ROOT = (
    Path(__file__).parents[2] / "fixtures" / "patient_delete_authorization"
)


def _fixture(name: str) -> dict[str, Any]:
    value = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _gateway() -> ReplayGateway:
    return ReplayGateway(
        {
            "test_plan": _fixture("planner_response.json"),
            "pytest_generation": _fixture("generator_response.json"),
        }
    )


def _workflow(
    tmp_path: Path,
    *,
    run_id: str,
    gateway: Any | None = None,
    recorder: ArtifactRecorder | None = None,
    **dependencies: Any,
) -> MilestoneOneWorkflow:
    return MilestoneOneWorkflow(
        fixture_directory=FIXTURE_ROOT,
        settings=Settings(llm_mode="replay", artifacts_dir=tmp_path),
        gateway=gateway or _gateway(),
        recorder=recorder or ArtifactRecorder(tmp_path),
        run_id=run_id,
        **dependencies,
    )


def _approve(workflow: MilestoneOneWorkflow) -> None:
    prepared = workflow.prepare()
    workflow.approve_contract(prepared.contract, prepared.gherkin)


def _rewrite_artifact_and_journal(
    run_directory: Path,
    artifact_name: str,
    mutate: Any,
) -> None:
    """Model an out-of-band rewrite that also fabricates matching journal hashes."""
    artifact_path = run_directory / artifact_name
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(payload)
    content = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    artifact_path.write_bytes(content)

    event_path = run_directory / "events.jsonl"
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    for event in events:
        event_payload = event["payload"]
        if (
            event["event_type"]
            in {"artifact_write_started", "artifact_write_completed"}
            and event_payload.get("artifact_name") == artifact_name
        ):
            event_payload["artifact_sha256"] = digest
            event_payload["artifact_byte_count"] = len(content)
            event_payload["provenance"]["output_hashes"][artifact_name] = digest
        output_hashes = event_payload.get("output_hashes")
        if isinstance(output_hashes, dict) and artifact_name in output_hashes:
            output_hashes[artifact_name] = digest
    event_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _concurrent_generate_process(
    artifact_root: str,
    fixture_root: str,
    handle_payload: dict[str, Any],
    log_path: str,
    barrier: Any,
    outcomes: Any,
) -> None:
    gateway = ReplayGateway(
        {
            "test_plan": json.loads(
                (Path(fixture_root) / "planner_response.json").read_text(
                    encoding="utf-8"
                )
            ),
            "pytest_generation": json.loads(
                (Path(fixture_root) / "generator_response.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
    )
    real_generate = gateway.generate

    def logged_generate(request: Any) -> Any:
        with Path(log_path).open("a", encoding="utf-8") as stream:
            stream.write(f"{request.purpose}\n")
            stream.flush()
        time.sleep(0.05)
        return real_generate(request)

    gateway.generate = logged_generate  # type: ignore[method-assign]
    workflow = resume_replay_workflow(
        artifact_root=artifact_root,
        fixture_directory=fixture_root,
        handle=RunHandle.model_validate(handle_payload),
        gateway=gateway,
    )
    barrier.wait(timeout=10)
    try:
        generated = workflow.generate()
    except WorkflowTransitionError as error:
        outcomes.put(("transition", str(error)))
    else:
        outcomes.put(("generated", generated.validation.approved))


def _resume_prepared_process(
    artifact_root: str,
    fixture_root: str,
    handle_payload: dict[str, Any],
    outcomes: Any,
) -> None:
    """Resume one prepared snapshot in a spawned interpreter."""
    try:
        workflow = resume_replay_workflow(
            artifact_root=artifact_root,
            fixture_directory=fixture_root,
            handle=RunHandle.model_validate(handle_payload),
        )
        prepared = workflow._prepared
        if prepared is None:
            raise AssertionError("prepared snapshot was not reconstructed")
    except BaseException as error:  # noqa: BLE001 - child reports exact failure
        outcomes.put(("error", type(error).__name__, str(error)))
    else:
        outcomes.put(
            (
                "prepared",
                prepared.run_id,
                prepared.contract_sha256,
                prepared.cvss_profile_sha256,
            )
        )


class _FakeServer:
    def __init__(self, behavior: str) -> None:
        self.behavior = behavior
        self.base_url = "http://127.0.0.1:1"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _server_factory(**kwargs: str) -> _FakeServer:
    return _FakeServer(kwargs["behavior"])


def _observation(revision: str) -> RuntimeObservation:
    secure = revision == "base-revision"
    return RuntimeObservation(
        revision=revision,
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=True,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=403 if secure else 204,
        resource_exists_after=secure,
        pytest_exit_code=0 if secure else 1,
        reason_code=(
            "pytest_completed_with_observation"
            if secure
            else "pytest_failed_with_complete_observation"
        ),
    )


def _write_final_observation(
    path: Path, observation: RuntimeObservation
) -> None:
    envelope = RuntimeObservationEnvelope(
        **observation.model_dump(mode="json"),
        contract_sha256=canonical_sha256(_fixture("approved_contract.json")),
    )
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


class _BehaviorRunner:
    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root
        self.last_artifacts = None
        self.calls = 0

    def run(self, target: Any) -> RuntimeObservation:
        self.calls += 1
        run_directory = self._artifact_root / f"behavior-{self.calls}"
        run_directory.mkdir(parents=True)
        self.last_artifacts = ExecutionArtifacts(
            run_directory=run_directory,
            pytest_config_path=run_directory / "pytest.ini",
            feature_path=run_directory / "authorization.feature",
            test_path=run_directory / "test_authorization.py",
            observation_path=run_directory / "observation.json",
            pytest_outcome_path=run_directory / "pytest-outcome.json",
            stdout_path=run_directory / "pytest.stdout.txt",
            stderr_path=run_directory / "pytest.stderr.txt",
        )
        for path in (
            self.last_artifacts.pytest_config_path,
            self.last_artifacts.feature_path,
            self.last_artifacts.test_path,
            self.last_artifacts.pytest_outcome_path,
            self.last_artifacts.stdout_path,
            self.last_artifacts.stderr_path,
            Path(f"{self.last_artifacts.observation_path}.events.jsonl"),
        ):
            path.write_text(f"fixture bytes for {path.name}\n", encoding="utf-8")
        observation = _observation(target.revision)
        _write_final_observation(
            self.last_artifacts.observation_path, observation
        )
        return observation


def test_same_object_generation_mutex_prevents_duplicate_external_calls(
    tmp_path: Path,
) -> None:
    """Two simultaneous generate calls must not duplicate either LLM operation."""
    delegate = _gateway()
    first_call_entered = Event()
    release_first_call = Event()
    purposes: list[str] = []

    class BlockingGateway:
        def generate(self, request: Any) -> Any:
            purposes.append(request.purpose)
            if request.purpose == "test_plan":
                first_call_entered.set()
                assert release_first_call.wait(timeout=5)
            return delegate.generate(request)

    workflow = _workflow(
        tmp_path,
        run_id="run-concurrent-generate",
        gateway=BlockingGateway(),
    )
    _approve(workflow)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(workflow.generate)
        assert first_call_entered.wait(timeout=5)
        second = executor.submit(workflow.generate)
        release_first_call.set()
        outcomes = [
            future.result() if future.exception() is None else future.exception()
            for future in (first, second)
        ]

    assert purposes == ["test_plan", "pytest_generation"]
    assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
    assert sum(isinstance(item, WorkflowTransitionError) for item in outcomes) == 1


def test_cross_process_workflow_lease_prevents_duplicate_generation_calls(
    tmp_path: Path,
) -> None:
    """Two resumed processes must produce one durable logical generation stream."""
    workflow = _workflow(tmp_path, run_id="run-cross-process-generate")
    _approve(workflow)
    context = get_context("spawn")
    barrier = context.Barrier(2)
    outcomes = context.Queue()
    log_path = tmp_path / "external-model-calls.log"
    arguments = (
        str(tmp_path),
        str(FIXTURE_ROOT),
        workflow.run_handle.model_dump(mode="json"),
        str(log_path),
        barrier,
        outcomes,
    )
    processes = [
        context.Process(target=_concurrent_generate_process, args=arguments)
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    results = sorted(outcomes.get(timeout=2) for _ in processes)
    assert [kind for kind, _ in results] == ["generated", "transition"]
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "test_plan",
        "pytest_generation",
    ]


def test_transition_artifact_precedes_event_and_recovers_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An event-append crash recovers only from the exact completed artifact."""
    recorder = ArtifactRecorder(tmp_path)
    workflow = _workflow(
        tmp_path,
        run_id="run-artifact-before-transition-event",
        recorder=recorder,
    )
    real_record = recorder.record_transformation
    interrupted = False

    def interrupt_prepared_event(handle: RunHandle, event: Any) -> Any:
        nonlocal interrupted
        if event.event_type == "workflow_prepared" and not interrupted:
            interrupted = True
            raise OSError("process died before transition event append")
        return real_record(handle, event)

    monkeypatch.setattr(
        recorder, "record_transformation", interrupt_prepared_event
    )
    with pytest.raises(OSError, match="transition event append"):
        workflow.prepare()

    snapshot = workflow._preparation_snapshot
    assert snapshot is not None
    run_directory = tmp_path / workflow.run_handle.run_id
    assert (run_directory / snapshot.transition.artifact_name).read_bytes() == (
        snapshot.transition.content
    )
    initial_events = ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
    assert sum(
        event.event_type == "workflow_prepared" for event in initial_events
    ) == 0

    context = get_context("spawn")
    for _ in range(2):
        outcomes = context.Queue()
        process = context.Process(
            target=_resume_prepared_process,
            args=(
                str(tmp_path),
                str(FIXTURE_ROOT),
                workflow.run_handle.model_dump(mode="json"),
                outcomes,
            ),
        )
        process.start()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert outcomes.get(timeout=2) == (
            "prepared",
            workflow.run_handle.run_id,
            snapshot.prepared.contract_sha256,
            snapshot.prepared.cvss_profile_sha256,
        )

    events = ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
    event_types = [event.event_type for event in events]
    assert event_types.count("workflow_prepared") == 1
    assert sum(
        event.event_type == "artifact_write_started"
        and event.payload.get("artifact_name") == snapshot.transition.artifact_name
        for event in events
    ) == 1
    assert sum(
        event.event_type == "artifact_write_completed"
        and event.payload.get("artifact_name") == snapshot.transition.artifact_name
        for event in events
    ) == 1
    assert list((run_directory / "artifacts/prepared").iterdir()) == [
        run_directory / snapshot.transition.artifact_name
    ]


def test_legacy_transition_event_without_artifact_fails_closed_then_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the original frozen snapshot may repair a legacy event-first crash."""
    recorder = ArtifactRecorder(tmp_path)
    workflow = _workflow(
        tmp_path,
        run_id="run-legacy-event-before-artifact",
        recorder=recorder,
    )
    real_commit = workflow._commit_transition
    interrupted = False

    def commit_event_only(snapshot: Any) -> None:
        nonlocal interrupted
        if snapshot.event.event_type == "workflow_prepared" and not interrupted:
            interrupted = True
            recorder.record_transformation(workflow.run_handle, snapshot.event)
            monkeypatch.setattr(workflow, "_commit_transition", real_commit)
            raise OSError("legacy process died before transition artifact")
        real_commit(snapshot)

    monkeypatch.setattr(workflow, "_commit_transition", commit_event_only)
    with pytest.raises(OSError, match="transition artifact"):
        workflow.prepare()

    snapshot = workflow._preparation_snapshot
    assert snapshot is not None
    run_directory = tmp_path / workflow.run_handle.run_id
    assert not (run_directory / snapshot.transition.artifact_name).exists()
    with pytest.raises(
        WorkflowTransitionError, match="event exists without its artifact"
    ):
        resume_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            handle=workflow.run_handle,
        )

    before_retry = ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
    assert sum(event.event_type == "workflow_prepared" for event in before_retry) == 1
    assert workflow.prepare() == snapshot.prepared

    for _ in range(2):
        reloaded = resume_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            handle=workflow.run_handle,
        )
        assert reloaded._prepared == snapshot.prepared

    events = ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
    assert sum(event.event_type == "workflow_prepared" for event in events) == 1
    assert sum(
        event.event_type == "artifact_write_started"
        and event.payload.get("artifact_name") == snapshot.transition.artifact_name
        for event in events
    ) == 1
    assert sum(
        event.event_type == "artifact_write_completed"
        and event.payload.get("artifact_name") == snapshot.transition.artifact_name
        for event in events
    ) == 1
    assert list((run_directory / "artifacts/prepared").iterdir()) == [
        run_directory / snapshot.transition.artifact_name
    ]


def test_same_object_execution_mutex_prevents_duplicate_experiments(
    tmp_path: Path,
) -> None:
    """Concurrent execute calls must run one base/candidate pair only once."""
    delegate = _BehaviorRunner(tmp_path / "runner-artifacts")
    first_call_entered = Event()
    release_first_call = Event()

    class BlockingRunner:
        last_artifacts = None
        calls = 0

        def run(self, target: Any) -> RuntimeObservation:
            self.calls += 1
            if self.calls == 1:
                first_call_entered.set()
                assert release_first_call.wait(timeout=5)
            observation = delegate.run(target)
            self.last_artifacts = delegate.last_artifacts
            return observation

    runner = BlockingRunner()
    workflow = _workflow(
        tmp_path,
        run_id="run-concurrent-execute",
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(workflow.execute, repeat_count=1)
        assert first_call_entered.wait(timeout=5)
        second = executor.submit(workflow.execute, repeat_count=1)
        release_first_call.set()
        outcomes = [
            future.result() if future.exception() is None else future.exception()
            for future in (first, second)
        ]

    assert runner.calls == 2
    assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
    assert sum(isinstance(item, WorkflowTransitionError) for item in outcomes) == 1


def test_process_resume_abstains_after_pending_model_intent_without_repeating_call(
    tmp_path: Path,
) -> None:
    """A crash after durable intent leaves an unknown call that cannot be replayed."""
    class InterruptingGateway:
        def generate(self, request: Any) -> Any:
            raise SystemExit("simulated process death during model call")

    workflow = _workflow(
        tmp_path,
        run_id="run-pending-model-intent",
        gateway=InterruptingGateway(),
    )
    _approve(workflow)
    with pytest.raises(SystemExit, match="process death"):
        workflow.generate()

    resumed_gateway = _gateway()
    calls = 0
    real_generate = resumed_gateway.generate

    def counted_generate(request: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_generate(request)

    resumed_gateway.generate = counted_generate  # type: ignore[method-assign]
    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        gateway=resumed_gateway,
    )

    with pytest.raises(InterruptedExternalOperationError, match="unknown outcome"):
        resumed.generate()

    assert calls == 0
    result = resumed.result()
    assert result.status is WorkflowStatus.GENERATION_ABSTAINED
    assert result.reason_code == "model_operation_interrupted_unknown_outcome"


def test_process_resume_reconciles_completed_model_results_without_repeating_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable model results are replayed locally when stage commit was interrupted."""
    recorder = ArtifactRecorder(tmp_path)
    workflow = _workflow(
        tmp_path,
        run_id="run-completed-model-results",
        recorder=recorder,
    )
    _approve(workflow)
    real_write = recorder.write_artifact
    interrupted = False

    def interrupt_generated_transition(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        provenance = args[3]
        if provenance.event_type == "workflow_generated" and not interrupted:
            interrupted = True
            raise OSError("interrupted before generated transition commit")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(recorder, "write_artifact", interrupt_generated_transition)
    with pytest.raises(OSError, match="generated transition"):
        workflow.generate()

    forbidden_gateway = _gateway()

    def forbid_repeat(request: Any) -> Any:
        raise AssertionError(f"repeated external call: {request.purpose}")

    forbidden_gateway.generate = forbid_repeat  # type: ignore[method-assign]
    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        gateway=forbidden_gateway,
    )

    assert resumed.generate().validation.approved is True


def test_process_resume_completes_a_known_model_result_journal_without_repeating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known result bytes survive a crash before their completion event."""
    recorder = ArtifactRecorder(tmp_path)
    workflow = _workflow(
        tmp_path,
        run_id="run-model-result-journal-recovery",
        recorder=recorder,
    )
    _approve(workflow)
    real_append = recorder._append_event_locked
    interrupted = False

    def interrupt_result_completion(
        run_fd: int, event_type: str, payload: Any
    ) -> Any:
        nonlocal interrupted
        if (
            event_type == "artifact_write_completed"
            and payload.get("artifact_name")
            == "artifacts/operations/model/test_plan/result.json"
            and not interrupted
        ):
            interrupted = True
            raise OSError("interrupted before model result completion")
        return real_append(run_fd, event_type, payload)

    monkeypatch.setattr(
        recorder, "_append_event_locked", interrupt_result_completion
    )
    with pytest.raises(OSError, match="durable result commit interrupted"):
        workflow.generate()

    resumed_gateway = _gateway()
    purposes: list[str] = []
    real_generate = resumed_gateway.generate

    def counted_generate(request: Any) -> Any:
        purposes.append(request.purpose)
        return real_generate(request)

    resumed_gateway.generate = counted_generate  # type: ignore[method-assign]
    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        gateway=resumed_gateway,
    )

    assert resumed.generate().validation.approved is True
    assert purposes == ["pytest_generation"]


def test_process_resume_is_inconclusive_after_pending_experiment_intent(
    tmp_path: Path,
) -> None:
    """An experiment with an unknown outcome must never be executed a second time."""
    class InterruptingRunner:
        last_artifacts = None

        def run(self, target: Any) -> RuntimeObservation:
            raise SystemExit(f"process death during {target.revision}")

    workflow = _workflow(
        tmp_path,
        run_id="run-pending-experiment",
        runner_factory=lambda **kwargs: InterruptingRunner(),
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    with pytest.raises(SystemExit, match="base-revision"):
        workflow.execute(repeat_count=1)

    runner_calls = 0

    class ForbiddenRunner:
        last_artifacts = None

        def run(self, target: Any) -> RuntimeObservation:
            nonlocal runner_calls
            runner_calls += 1
            raise AssertionError("experiment was repeated")

    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        runner_factory=lambda **kwargs: ForbiddenRunner(),
        server_factory=_server_factory,
    )
    with pytest.raises(InterruptedExternalOperationError, match="unknown outcome"):
        resumed.execute(repeat_count=1)

    assert runner_calls == 0
    assert resumed.result().status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert resumed.result().reason_code == "experiment_interrupted_unknown_outcome"


def test_partial_experiment_recovery_finalizes_exact_completed_manifests(
    tmp_path: Path,
) -> None:
    """A later unknown outcome cannot discard an earlier durable observation."""
    delegate = _BehaviorRunner(tmp_path / "runner-partial-recovery-artifacts")

    class CandidateCrashRunner:
        last_artifacts = None

        def run(self, target: Any) -> RuntimeObservation:
            if target.revision == "candidate-revision":
                raise SystemExit("process died during candidate experiment")
            observation = delegate.run(target)
            self.last_artifacts = delegate.last_artifacts
            return observation

    workflow = _workflow(
        tmp_path,
        run_id="run-partial-experiment-recovery",
        runner_factory=lambda **kwargs: CandidateCrashRunner(),
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    with pytest.raises(SystemExit, match="candidate experiment"):
        workflow.execute(repeat_count=1)

    runner_constructions = 0

    def forbid_runner(**kwargs: Any) -> Any:
        nonlocal runner_constructions
        runner_constructions += 1
        raise AssertionError("a durable or unknown experiment was repeated")

    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        runner_factory=forbid_runner,
        server_factory=_server_factory,
    )
    with pytest.raises(InterruptedExternalOperationError, match="1:candidate"):
        resumed.execute(repeat_count=1)

    terminal = resumed.result()
    assert terminal.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert terminal.reason_code == "experiment_interrupted_unknown_outcome"
    assert terminal.differential_evidence is None
    assert len(terminal.execution_manifest_sha256s) == 1
    assert runner_constructions == 0

    for _ in range(2):
        reloaded = resume_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            handle=workflow.run_handle,
            runner_factory=forbid_runner,
            server_factory=_server_factory,
        )
        assert reloaded.result() == terminal

    events = ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
    event_types = [event.event_type for event in events]
    assert event_types.count("external_experiment_1_base_result") == 1
    assert event_types.count("external_experiment_1_candidate_intent") == 1
    assert event_types.count("external_experiment_1_candidate_result") == 0
    assert event_types.count("external_execution_1_base_manifest") == 1
    assert event_types.count("workflow_finalization_intent") == 1
    assert event_types.count("finalization_started") == 1
    assert event_types.count("finalization_completed") == 1
    assert runner_constructions == 0


def test_completed_experiment_results_are_reconciled_without_rerunning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before the executed transition reuses both durable observations."""
    recorder = ArtifactRecorder(tmp_path)
    runner = _BehaviorRunner(tmp_path / "runner-artifacts")
    workflow = _workflow(
        tmp_path,
        run_id="run-completed-experiment-results",
        recorder=recorder,
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    real_write = recorder.write_artifact
    interrupted = False

    def interrupt_executed_transition(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        provenance = args[3]
        if provenance.event_type == "workflow_executed" and not interrupted:
            interrupted = True
            raise OSError("interrupted before executed transition commit")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(recorder, "write_artifact", interrupt_executed_transition)
    with pytest.raises(OSError, match="executed transition"):
        workflow.execute(repeat_count=1)
    assert runner.calls == 2

    class ForbiddenRunner:
        last_artifacts = None

        def run(self, target: Any) -> RuntimeObservation:
            raise AssertionError(f"repeated experiment: {target.revision}")

    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        runner_factory=lambda **kwargs: ForbiddenRunner(),
        server_factory=_server_factory,
    )

    result = resumed.execute(repeat_count=1)
    assert result.status is WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED


def test_resume_completes_a_terminal_record_missing_its_completion_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process death between terminal bytes and completion cannot wedge resume."""
    recorder = ArtifactRecorder(tmp_path)
    runner = _BehaviorRunner(tmp_path / "runner-finalization-artifacts")
    workflow = _workflow(
        tmp_path,
        run_id="run-incomplete-finalization-resume",
        recorder=recorder,
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    real_append = recorder._append_event_locked

    def fail_completion_once(
        run_fd: int, event_type: str, payload: Any
    ) -> Any:
        if event_type == "finalization_completed":
            monkeypatch.setattr(recorder, "_append_event_locked", real_append)
            raise OSError("process died before finalization completion")
        return real_append(run_fd, event_type, payload)

    monkeypatch.setattr(recorder, "_append_event_locked", fail_completion_once)
    with pytest.raises(OSError, match="finalization completion"):
        workflow.execute(repeat_count=1)

    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        runner_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("completed experiments must not be reconstructed")
        ),
        server_factory=_server_factory,
    )

    assert resumed.result().status is WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED
    events = ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
    assert sum(event.event_type == "finalization_completed" for event in events) == 1


def test_recovered_operation_artifact_must_match_its_completed_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out-of-band result bytes cannot be trusted merely because JSON still parses."""
    recorder = ArtifactRecorder(tmp_path)
    runner = _BehaviorRunner(tmp_path / "runner-journal-artifacts")
    workflow = _workflow(
        tmp_path,
        run_id="run-tampered-experiment-result",
        recorder=recorder,
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    real_write = recorder.write_artifact
    interrupted = False

    def interrupt_executed_transition(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        provenance = args[3]
        if provenance.event_type == "workflow_executed" and not interrupted:
            interrupted = True
            raise OSError("interrupted before executed transition commit")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(recorder, "write_artifact", interrupt_executed_transition)
    with pytest.raises(OSError, match="executed transition"):
        workflow.execute(repeat_count=1)

    result_path = (
        tmp_path
        / workflow.run_handle.run_id
        / "artifacts/operations/experiment/0001-base/result.json"
    )
    result_path.write_bytes(result_path.read_bytes() + b" ")
    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        runner_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("durable experiments must not be repeated")
        ),
        server_factory=_server_factory,
    )

    with pytest.raises(WorkflowTransitionError, match="durable operation artifact"):
        resumed.execute(repeat_count=1)


def test_recovered_observation_must_match_manifest_bound_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a journal-consistent result cannot contradict its immutable manifest."""
    recorder = ArtifactRecorder(tmp_path)
    runner = _BehaviorRunner(tmp_path / "runner-observation-artifacts")
    workflow = _workflow(
        tmp_path,
        run_id="run-fabricated-recovered-observation",
        recorder=recorder,
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    real_write = recorder.write_artifact
    interrupted = False

    def interrupt_executed_transition(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        provenance = args[3]
        if provenance.event_type == "workflow_executed" and not interrupted:
            interrupted = True
            raise OSError("interrupted before executed transition commit")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(recorder, "write_artifact", interrupt_executed_transition)
    with pytest.raises(OSError, match="executed transition"):
        workflow.execute(repeat_count=1)

    artifact_name = "artifacts/operations/experiment/0001-base/result.json"

    def fabricate_observation(payload: dict[str, Any]) -> None:
        payload["observation"]["reason_code"] = "fabricated_execution_complete"

    _rewrite_artifact_and_journal(
        tmp_path / workflow.run_handle.run_id,
        artifact_name,
        fabricate_observation,
    )
    resumed = resume_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        handle=workflow.run_handle,
        runner_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("durable experiments must not be repeated")
        ),
        server_factory=_server_factory,
    )

    with pytest.raises(WorkflowTransitionError, match="execution manifest"):
        resumed.execute(repeat_count=1)


def test_known_execution_failure_remains_distinct_from_unknown_interruption(
    tmp_path: Path,
) -> None:
    """A returned runner error keeps its original reason rather than unknown outcome."""
    class FailingRunner:
        last_artifacts = None

        def run(self, target: Any) -> RuntimeObservation:
            raise MissingObservationError("known missing observation", None)  # type: ignore[arg-type]

    workflow = _workflow(
        tmp_path,
        run_id="run-known-execution-failure",
        runner_factory=lambda **kwargs: FailingRunner(),
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(MissingObservationError):
        workflow.execute(repeat_count=1)
    assert workflow.result().reason_code == "missing_runtime_observation"


def test_malformed_classifier_result_is_guarded_without_rerunning_experiments(
    tmp_path: Path,
) -> None:
    """An injected classifier cannot smuggle an incoherent terminal conclusion."""
    runner = _BehaviorRunner(tmp_path / "runner-artifacts")
    workflow = _workflow(
        tmp_path,
        run_id="run-malformed-classifier-result",
        runner_factory=lambda **kwargs: runner,
        classifier=lambda *args: {"status": "candidate_regression_observed"},
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(TypeError, match="DifferentialEvidence"):
        workflow.execute(repeat_count=1)

    assert runner.calls == 2
    assert workflow.result().status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert workflow.result().reason_code == "invalid_classifier_result"
    with pytest.raises(WorkflowTransitionError):
        workflow.execute(repeat_count=1)
    assert runner.calls == 2


def test_typed_classifier_result_must_equal_the_actual_observation_classification(
    tmp_path: Path,
) -> None:
    """Internal model coherence cannot legitimize fabricated representative facts."""
    runner = _BehaviorRunner(tmp_path / "runner-fabricated-classifier-artifacts")

    def fabricated_classifier(
        base: list[RuntimeObservation],
        candidate: list[RuntimeObservation],
        contract: Any,
    ) -> DifferentialEvidence:
        expected = classify_differential(base, candidate, contract)
        fabricated_base = RuntimeObservation.model_validate(
            {
                **expected.base.model_dump(mode="json"),
                "reason_code": "fabricated_execution_complete",
            }
        )
        return DifferentialEvidence.model_validate(
            {
                **expected.model_dump(mode="json"),
                "base": fabricated_base.model_dump(mode="json"),
            }
        )

    workflow = _workflow(
        tmp_path,
        run_id="run-coherent-fabricated-classifier",
        runner_factory=lambda **kwargs: runner,
        classifier=fabricated_classifier,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(TypeError, match="coherent DifferentialEvidence"):
        workflow.execute(repeat_count=1)

    assert runner.calls == 2
    assert workflow.result().reason_code == "invalid_classifier_result"


def test_preparation_and_terminal_failure_carry_both_exact_revisions(
    tmp_path: Path,
) -> None:
    """Revision identity must survive even when generation cannot produce evidence."""
    workflow = _workflow(
        tmp_path,
        run_id="run-revisions-through-failure",
        gateway=ReplayGateway({"test_plan": _fixture("planner_response.json")}),
        base_revision="base-abcdef1",
        candidate_revision="candidate-abcdef2",
    )

    prepared = workflow.prepare()
    assert prepared.base_revision == "base-abcdef1"
    assert prepared.candidate_revision == "candidate-abcdef2"
    workflow.approve_contract(prepared.contract, prepared.gherkin)
    with pytest.raises(Exception, match="pytest_generation"):
        workflow.generate()

    result = workflow.result()
    assert result.base_revision == prepared.base_revision
    assert result.candidate_revision == prepared.candidate_revision


@pytest.mark.parametrize(
    "mutation",
    [
        "score",
        "severity",
        "calculator",
        "evidence_hash",
        "profile_hash",
        "coordinated_profile",
        "base_reason",
    ],
)
def test_resume_rejects_rehashed_severity_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Rehashing a fabricated severity claim must not make recovery trust it."""
    runner = _BehaviorRunner(tmp_path / "runner-severity-tamper")
    workflow = _workflow(
        tmp_path,
        run_id="run-severity-tamper",
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    workflow.execute(repeat_count=1)
    handle = workflow.run_handle
    run_directory = tmp_path / handle.run_id
    executed_artifact = next((run_directory / "artifacts/executed").iterdir())

    def change_severity(payload: dict[str, Any]) -> None:
        severity = payload["run_record"]["severity_assessment"]
        candidate = severity["candidate"]
        if mutation == "score":
            candidate["score"] = 9.9
        elif mutation == "severity":
            candidate["severity"] = "Critical"
        elif mutation == "calculator":
            candidate["calculator"] = "cvss-python/0.0"
        elif mutation == "evidence_hash":
            candidate["evidence_sha256"] = "f" * 64
        elif mutation == "profile_hash":
            candidate["profile_sha256"] = "f" * 64
        elif mutation == "coordinated_profile":
            candidate["vector"] = candidate["vector"].replace("/VI:H/", "/VI:L/")
            candidate["metrics"][6]["value"] = "L"
            candidate["profile_sha256"] = canonical_sha256(
                {
                    "profile_id": candidate["profile_id"],
                    "cvss_version": "4.0",
                    "vector": candidate["vector"],
                    "metrics": candidate["metrics"],
                    "assessment_label": "expert_authored_provisional",
                }
            )
        else:
            severity["base"]["reason_code"] = (
                "insufficient_evidence_for_severity"
            )

    _rewrite_artifact_and_journal(
        run_directory,
        executed_artifact.relative_to(run_directory).as_posix(),
        change_severity,
    )

    with pytest.raises(WorkflowTransitionError, match="severity"):
        resume_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            handle=handle,
        )


def test_resume_rejects_executed_severity_bound_to_another_profile(
    tmp_path: Path,
) -> None:
    """A forged transition hash must not detach severity from preparation."""
    runner = _BehaviorRunner(tmp_path / "runner-profile-binding-tamper")
    workflow = _workflow(
        tmp_path,
        run_id="run-profile-binding-tamper",
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()
    workflow.execute(repeat_count=1)
    handle = workflow.run_handle
    event_path = tmp_path / handle.run_id / "events.jsonl"
    events = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    for event in events:
        if event["event_type"] == "workflow_executed":
            event["payload"]["input_hashes"]["cvss_profile"] = "f" * 64
        provenance = event["payload"].get("provenance")
        if (
            isinstance(provenance, dict)
            and provenance.get("event_type") == "workflow_executed"
        ):
            provenance["input_hashes"]["cvss_profile"] = "f" * 64
    event_path.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkflowTransitionError, match="another CVSS profile"):
        resume_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            handle=handle,
        )


def test_execution_manifests_bind_every_file_and_detect_tampering(
    tmp_path: Path,
) -> None:
    """Terminal evidence must transitively bind every persisted experiment byte."""
    workflow = _workflow(tmp_path, run_id="run-execution-manifest-binding")
    _approve(workflow)
    workflow.generate()

    result = workflow.execute(repeat_count=1)

    assert result.differential_evidence is not None
    assert result.base_revision == "base-revision"
    assert result.candidate_revision == "candidate-revision"
    assert result.execution_manifest_sha256s == (
        result.differential_evidence.execution_manifest_sha256s
    )
    assert len(result.execution_manifest_sha256s) == 2
    required_files = {
        "feature",
        "generated_test",
        "pytest_config",
        "raw_event_sidecar",
        "final_observation",
        "structured_pytest_outcome",
        "stdout",
        "stderr",
    }
    manifests: list[dict[str, Any]] = []
    run_directory = tmp_path / result.run_id
    for side, expected_digest in zip(
        ("base", "candidate"),
        result.execution_manifest_sha256s,
        strict=True,
    ):
        manifest_path = (
            run_directory
            / "artifacts"
            / "executions"
            / f"0001-{side}"
            / "manifest.json"
        )
        manifest_bytes = manifest_path.read_bytes()
        assert hashlib.sha256(manifest_bytes).hexdigest() == expected_digest
        manifest = json.loads(manifest_bytes)
        manifests.append(manifest)
        assert manifest["side"] == side
        assert manifest["revision"] == f"{side}-revision"
        assert manifest["repetition_index"] == 1
        assert set(manifest["files"]) == required_files
        for file_record in manifest["files"].values():
            path = run_directory / file_record["relative_path"]
            content = path.read_bytes()
            assert len(content) == file_record["byte_count"]
            assert hashlib.sha256(content).hexdigest() == file_record["sha256"]

    tampered_path = run_directory / manifests[0]["files"]["feature"]["relative_path"]
    tampered_path.write_bytes(tampered_path.read_bytes() + b"# tampered\n")

    with pytest.raises(WorkflowTransitionError, match="execution manifest"):
        workflow.result()


@pytest.mark.parametrize("link_kind", ["parent", "final"])
def test_execution_manifest_rejects_symlinked_source_components(
    tmp_path: Path, link_kind: str
) -> None:
    """No source path component may redirect the manifest reader."""
    runner = _BehaviorRunner(tmp_path / f"runner-symlink-{link_kind}")
    real_run = runner.run

    def run_with_symlink(target: Any) -> RuntimeObservation:
        observation = real_run(target)
        artifacts = runner.last_artifacts
        assert isinstance(artifacts, ExecutionArtifacts)
        source = artifacts.feature_path
        outside_directory = runner._artifact_root / f"outside-{runner.calls}"
        outside_directory.mkdir()
        outside = outside_directory / source.name
        outside.write_bytes(source.read_bytes())
        if link_kind == "parent":
            linked_parent = artifacts.run_directory / "linked-parent"
            linked_parent.symlink_to(outside_directory, target_is_directory=True)
            unsafe_feature = linked_parent / source.name
        else:
            source.unlink()
            source.symlink_to(outside)
            unsafe_feature = source
        runner.last_artifacts = replace(
            artifacts,
            feature_path=unsafe_feature,
        )
        return observation

    runner.run = run_with_symlink  # type: ignore[method-assign]
    workflow = _workflow(
        tmp_path,
        run_id=f"run-manifest-symlink-{link_kind}",
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(ExecutionManifestError, match="completed runner file"):
        workflow.execute(repeat_count=1)

    result = workflow.result()
    assert result.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert result.reason_code == "execution_manifest_invalid"
    assert result.execution_manifest_sha256s == []
    assert runner.calls == 1


def test_execution_manifest_rejects_final_component_swap_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspection and reading must share one no-follow file descriptor."""
    runner = _BehaviorRunner(tmp_path / "runner-source-swap")
    outside = tmp_path / "outside-observation.json"
    _write_final_observation(outside, _observation("base-revision"))
    real_read_bytes = Path.read_bytes
    real_os_open = os.open
    swapped = False

    def swap_source() -> None:
        nonlocal swapped
        if swapped:
            return
        artifacts = runner.last_artifacts
        assert isinstance(artifacts, ExecutionArtifacts)
        artifacts.observation_path.unlink()
        artifacts.observation_path.symlink_to(outside)
        swapped = True

    def racing_read_bytes(path: Path) -> bytes:
        artifacts = runner.last_artifacts
        if (
            isinstance(artifacts, ExecutionArtifacts)
            and path == artifacts.observation_path
        ):
            swap_source()
        return real_read_bytes(path)

    def racing_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        artifacts = runner.last_artifacts
        if (
            isinstance(artifacts, ExecutionArtifacts)
            and dir_fd is not None
            and os.fsdecode(path) == artifacts.observation_path.name
        ):
            opened_parent = os.fstat(dir_fd)
            expected_parent = artifacts.run_directory.stat()
            if (
                opened_parent.st_dev,
                opened_parent.st_ino,
            ) == (
                expected_parent.st_dev,
                expected_parent.st_ino,
            ):
                swap_source()
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    monkeypatch.setattr(os, "open", racing_os_open)
    workflow = _workflow(
        tmp_path,
        run_id="run-manifest-source-swap",
        runner_factory=lambda **kwargs: runner,
        server_factory=_server_factory,
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(ExecutionManifestError, match="completed runner file"):
        workflow.execute(repeat_count=1)

    assert swapped is True
    result = workflow.result()
    assert result.reason_code == "execution_manifest_invalid"
    assert result.execution_manifest_sha256s == []
    assert runner.calls == 1


def test_result_rechecks_the_exact_terminal_record_bytes(
    tmp_path: Path,
) -> None:
    """An in-memory result must not hide post-finalization record corruption."""
    workflow = _workflow(
        tmp_path,
        run_id="run-terminal-record-integrity",
        gateway=ReplayGateway({"test_plan": _fixture("planner_response.json")}),
    )
    _approve(workflow)
    with pytest.raises(Exception, match="pytest_generation"):
        workflow.generate()
    terminal_path = tmp_path / workflow.run_handle.run_id / "run_record.json"
    terminal_path.write_bytes(b"{}\n")

    with pytest.raises(WorkflowTransitionError, match="terminal record"):
        workflow.result()
