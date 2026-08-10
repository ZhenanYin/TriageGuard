from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Self

import pytest

from triageguard.config import Settings
from triageguard.domain import (
    CvssProfile,
    EnvironmentKind,
    RiskContract,
    RuntimeObservation,
    WorkflowStatus,
)
from triageguard.evidence import classify_differential
from triageguard.execution import (
    ExecutionArtifacts,
    ExecutionTimeoutError,
    MissingObservationError,
)
from triageguard.generation import validate_generated_code
from triageguard.llm import ModelOutputInvalid, ReplayGateway, ReplayResponseMissing
from triageguard.provenance import canonical_sha256
from triageguard.research import ArtifactRecorder
from triageguard.research.recorder import LifecycleEventType, RunOwnership
from triageguard.runtime import RuntimeObservationEnvelope
from triageguard.severity import CvssAssessmentError
from triageguard.workflow import (
    ContractApprovalError,
    MilestoneOneWorkflow,
    UnsafeGeneratedCodeError,
    WorkflowTransitionError,
    build_replay_workflow,
)

FIXTURE_ROOT = (
    Path(__file__).parents[2] / "fixtures" / "patient_delete_authorization"
)


def _fixture_payload(name: str) -> dict[str, Any]:
    payload = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _directory_fingerprint(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def _approve(workflow: Any) -> None:
    prepared = workflow.prepare()
    workflow.approve_contract(prepared.contract, prepared.gherkin)


def _final_payload(tmp_path: Path, run_id: str) -> dict[str, Any]:
    return json.loads(
        (tmp_path / run_id / "run_record.json").read_text(encoding="utf-8")
    )


class _FakeServer:
    """A no-socket context that exposes only the controlled target boundary."""

    def __init__(self, behavior: str, instances: list[_FakeServer]) -> None:
        self.behavior = behavior
        self.base_url = "http://127.0.0.1:1"
        self.entered = False
        self.stopped = False
        instances.append(self)

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self.stopped = True


def _server_factory(instances: list[_FakeServer]):
    def create(**kwargs: str) -> _FakeServer:
        return _FakeServer(kwargs["behavior"], instances)

    return create


def _observation(revision: str, behavior: str) -> RuntimeObservation:
    secure = behavior == "secure"
    return RuntimeObservation(
        revision=revision,
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=behavior != "inconclusive",
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
        contract_sha256=canonical_sha256(
            _fixture_payload("approved_contract.json")
        ),
    )
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


class _BehaviorRunner:
    def __init__(
        self,
        *,
        artifact_root: Path,
        base_behaviors: list[str],
        candidate_behaviors: list[str],
    ) -> None:
        self._behaviors = {
            "base-revision": iter(base_behaviors),
            "candidate-revision": iter(candidate_behaviors),
        }
        self._artifact_root = artifact_root
        self._run_count = 0
        self.last_artifacts = None

    def run(self, target: Any) -> RuntimeObservation:
        self._run_count += 1
        self.last_artifacts = _fake_execution_artifacts(
            self._artifact_root, f"behavior-{self._run_count}"
        )
        observation = _observation(
            target.revision, next(self._behaviors[target.revision])
        )
        _write_final_observation(
            self.last_artifacts.observation_path, observation
        )
        return observation


def _behavior_runner_factory(
    base_behaviors: list[str], candidate_behaviors: list[str]
):
    def create(**kwargs: Any) -> _BehaviorRunner:
        return _BehaviorRunner(
            artifact_root=Path(kwargs["artifact_root"]),
            base_behaviors=base_behaviors,
            candidate_behaviors=candidate_behaviors,
        )

    return create


class _FailingRunner:
    def __init__(self, artifact_root: Path, error_type: type[Exception]) -> None:
        self.last_artifacts = _fake_execution_artifacts(
            artifact_root, "failed-attempt"
        )
        self.last_artifacts.stdout_path.write_text(
            "retained stdout\n", encoding="utf-8"
        )
        self.last_artifacts.stderr_path.write_text(
            "retained stderr\n", encoding="utf-8"
        )
        self._error_type = error_type

    def run(self, target: Any) -> RuntimeObservation:
        raise self._error_type("controlled execution failure", self.last_artifacts)


def _failing_runner_factory(error_type: type[Exception]):
    def create(**kwargs: Any) -> _FailingRunner:
        return _FailingRunner(Path(kwargs["artifact_root"]), error_type)

    return create


class _CandidateFailingRunner:
    def __init__(self, artifact_root: Path) -> None:
        self._delegate = _FailingRunner(artifact_root, MissingObservationError)
        self.last_artifacts = self._delegate.last_artifacts

    def run(self, target: Any) -> RuntimeObservation:
        if target.revision == "base-revision":
            observation = _observation(target.revision, "secure")
            _write_final_observation(
                self.last_artifacts.observation_path, observation
            )
            return observation
        return self._delegate.run(target)


def _fake_execution_artifacts(
    artifact_root: Path, directory_name: str
) -> ExecutionArtifacts:
    run_directory = artifact_root / directory_name
    run_directory.mkdir(parents=True)
    artifacts = ExecutionArtifacts(
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
        artifacts.pytest_config_path,
        artifacts.feature_path,
        artifacts.test_path,
        artifacts.observation_path,
        artifacts.pytest_outcome_path,
        artifacts.stdout_path,
        artifacts.stderr_path,
        Path(f"{artifacts.observation_path}.events.jsonl"),
    ):
        path.write_text(f"fixture bytes for {path.name}\n", encoding="utf-8")
    return artifacts


class _CountingGateway:
    def __init__(self) -> None:
        self._delegate = ReplayGateway(
            {
                "test_plan": _fixture_payload("planner_response.json"),
                "pytest_generation": _fixture_payload("generator_response.json"),
            }
        )
        self.purposes: list[str] = []

    def generate(self, request: Any) -> Any:
        self.purposes.append(request.purpose)
        return self._delegate.generate(request)


def _direct_workflow(
    tmp_path: Path,
    *,
    run_id: str,
    gateway: Any,
    recorder: ArtifactRecorder | None = None,
    **dependencies: Any,
) -> MilestoneOneWorkflow:
    return MilestoneOneWorkflow(
        fixture_directory=FIXTURE_ROOT,
        settings=Settings(llm_mode="replay", artifacts_dir=tmp_path),
        gateway=gateway,
        recorder=recorder or ArtifactRecorder(tmp_path),
        run_id=run_id,
        **dependencies,
    )


def _install_transition_fault(
    monkeypatch: pytest.MonkeyPatch,
    recorder: ArtifactRecorder,
    *,
    target_event: str,
    timing: str,
) -> None:
    real_write = recorder.write_artifact
    triggered = False

    def fault_once(
        run_id: str,
        name: str,
        content: bytes,
        provenance: Any,
    ) -> Any:
        nonlocal triggered
        if provenance.event_type == target_event and not triggered:
            triggered = True
            if timing == "after_commit":
                real_write(run_id, name, content, provenance)
            raise OSError(f"simulated {timing} recorder interruption")
        return real_write(run_id, name, content, provenance)

    monkeypatch.setattr(recorder, "write_artifact", fault_once)


def test_prepare_binds_the_validated_cvss_profile_and_raw_fixture_hash(
    tmp_path: Path,
) -> None:
    """Removing profile provenance must make preparation evidence incomplete."""
    gateway = _CountingGateway()
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-prepared-cvss-profile",
        gateway=gateway,
    )

    prepared = workflow.prepare()

    assert prepared.cvss_profile.profile_id == "patient-delete-authz-cvss-001"
    assert prepared.cvss_profile_sha256 == canonical_sha256(
        _fixture_payload("cvss_profile.json")
    )
    assert gateway.purposes == []
    prepared_event = next(
        event
        for event in ArtifactRecorder(tmp_path).read_events(workflow.run_handle)
        if event.event_type == "workflow_prepared"
    )
    assert prepared_event.payload["input_hashes"]["cvss_profile_fixture"] == (
        hashlib.sha256((FIXTURE_ROOT / "cvss_profile.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("mutation", ["missing_metric", "unsupported_value"])
def test_invalid_cvss_profile_fails_preparation_before_any_model_call(
    tmp_path: Path,
    mutation: str,
) -> None:
    """An invalid scoring input must stop before a run can contact the LLM."""
    fixture_directory = tmp_path / "fixture"
    fixture_directory.mkdir()
    for name in ("approved_contract.json", "impact_report.json"):
        (fixture_directory / name).write_bytes((FIXTURE_ROOT / name).read_bytes())
    profile = _fixture_payload("cvss_profile.json")
    if mutation == "missing_metric":
        profile["metrics"] = [
            item for item in profile["metrics"] if item["metric"] != "VI"
        ]
    else:
        profile["vector"] = profile["vector"].replace("/AV:N/", "/AV:X/")
        profile["metrics"][0]["value"] = "X"
    (fixture_directory / "cvss_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    gateway = _CountingGateway()
    workflow = MilestoneOneWorkflow(
        fixture_directory=fixture_directory,
        settings=Settings(llm_mode="replay", artifacts_dir=tmp_path / "artifacts"),
        gateway=gateway,
        recorder=ArtifactRecorder(tmp_path / "artifacts"),
        run_id=f"run-invalid-profile-{mutation}",
    )

    expected_error = ValueError if mutation == "missing_metric" else CvssAssessmentError
    with pytest.raises(expected_error):
        workflow.prepare()

    assert gateway.purposes == []
    assert not (tmp_path / "artifacts" / f"run-invalid-profile-{mutation}").exists()


def test_vertical_slice_produces_attributable_candidate_regression(
    tmp_path: Path,
) -> None:
    """Removing the orchestrator must break the actual replay experiment."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-success",
    )

    prepared = workflow.prepare()
    assert prepared.environment_kind is EnvironmentKind.CONTROLLED_FIXTURE

    workflow.approve_contract(prepared.contract, prepared.gherkin)
    generated = workflow.generate()
    assert generated.validation.approved is True

    result = workflow.execute(repeat_count=3)
    assert result.status is WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED
    assert result.stable is True
    assert workflow.result() == result

    run_record_path = tmp_path / result.run_id / "run_record.json"
    assert run_record_path.exists()
    assert json.loads(run_record_path.read_text(encoding="utf-8"))["reason_code"] == (
        "candidate_regression_observed"
    )
    event_types = [
        json.loads(line)["event_type"]
        for line in (tmp_path / result.run_id / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for transition in (
        "workflow_prepared",
        "workflow_approved",
        "workflow_generated",
        "workflow_validated",
        "workflow_executed",
        "workflow_finalization_intent",
    ):
        assert transition in event_types

    with pytest.raises(WorkflowTransitionError, match="current state is finalized"):
        workflow.execute(repeat_count=3)


def test_generation_missing_replay_response_abstains_without_fallback(
    tmp_path: Path,
) -> None:
    """Adding a fallback response must make this explicit-abstention test fail."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-missing-replay",
        gateway=ReplayGateway({"test_plan": _fixture_payload("planner_response.json")}),
    )
    _approve(workflow)

    with pytest.raises(ReplayResponseMissing):
        workflow.generate()

    result = workflow.result()
    assert result.status is WorkflowStatus.GENERATION_ABSTAINED
    assert result.reason_code == "replay_response_missing"
    assert result.differential_evidence is None
    assert _final_payload(tmp_path, result.run_id)["reason_code"] == (
        "replay_response_missing"
    )
    finalized_artifact = next(
        (tmp_path / result.run_id / "artifacts" / "finalization_intent").iterdir()
    )
    payload = json.loads(finalized_artifact.read_text(encoding="utf-8"))
    provenance = payload["failure"]["details"]["llm_calls"][-1][
        "failure_provenance"
    ]
    assert provenance["provider"] == "replay"
    assert provenance["purpose"] == "pytest_generation"
    assert provenance["reason_code"] == "replay_response_missing"
    assert provenance["attempts"][0]["outcome"] == "failed"


def test_invalid_recorded_replay_output_retains_failure_provenance(
    tmp_path: Path,
) -> None:
    """A malformed recorded response must remain attributable in final artifacts."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-invalid-replay-output",
        gateway=ReplayGateway({"test_plan": {"not": "a plan"}}),
    )
    _approve(workflow)

    with pytest.raises(ModelOutputInvalid):
        workflow.generate()

    finalized_artifact = next(
        (
            tmp_path
            / "run-invalid-replay-output"
            / "artifacts"
            / "finalization_intent"
        ).iterdir()
    )
    payload = json.loads(finalized_artifact.read_text(encoding="utf-8"))
    provenance = payload["failure"]["details"]["llm_calls"][-1][
        "failure_provenance"
    ]
    assert provenance["provider"] == "replay"
    assert provenance["purpose"] == "test_plan"
    assert provenance["reason_code"] == "replay_invalid_output"
    assert provenance["response_sha256"] is not None
    assert provenance["attempts"][0]["outcome"] == "invalid_output"


def test_invalid_model_plan_abstains_and_preserves_failed_request(
    tmp_path: Path,
) -> None:
    """Accepting a plan with a changed actor must make this test fail."""
    invalid_plan = _fixture_payload("planner_response.json")
    invalid_plan["givens"][1]["inputs"]["actor"] = "administrator"
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-invalid-plan",
        gateway=ReplayGateway(
            {
                "test_plan": invalid_plan,
                "pytest_generation": _fixture_payload("generator_response.json"),
            }
        ),
    )
    _approve(workflow)

    with pytest.raises(ValueError, match="actor_changed"):
        workflow.generate()

    result = workflow.result()
    assert result.status is WorkflowStatus.GENERATION_ABSTAINED
    assert result.reason_code == "invalid_model_plan"
    finalized_artifact = next(
        (tmp_path / result.run_id / "artifacts" / "finalization_intent").iterdir()
    )
    payload = json.loads(finalized_artifact.read_text(encoding="utf-8"))
    calls = payload["failure"]["details"]["llm_calls"]
    assert [call["request"]["purpose"] for call in calls] == ["test_plan"]
    assert calls[0]["response"]["provider"] == "replay"


def test_unsafe_generated_code_finalizes_validation_failure(
    tmp_path: Path,
) -> None:
    """Allowing a forbidden subprocess import must make this test fail."""
    unsafe_generation = _fixture_payload("generator_response.json")
    unsafe_generation["code"] += "\nimport subprocess\n"
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-unsafe-code",
        gateway=ReplayGateway(
            {
                "test_plan": _fixture_payload("planner_response.json"),
                "pytest_generation": unsafe_generation,
            }
        ),
    )
    _approve(workflow)

    with pytest.raises(UnsafeGeneratedCodeError) as captured:
        workflow.generate()

    assert "forbidden_import" in captured.value.report.reason_codes
    result = workflow.result()
    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert result.reason_code == "unsafe_generated_code"
    assert (tmp_path / result.run_id / "artifacts" / "generated").is_dir()


def test_rejected_contract_finalizes_without_model_or_execution_evidence(
    tmp_path: Path,
) -> None:
    """Approving a semantically changed actor must make this test fail."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-unapproved",
    )
    prepared = workflow.prepare()
    changed = RiskContract.model_validate(
        {**prepared.contract.model_dump(mode="json"), "actor": "administrator"}
    )

    with pytest.raises(ContractApprovalError, match="contract_not_approved"):
        workflow.approve_contract(changed, prepared.gherkin)

    result = workflow.result()
    assert result.status is WorkflowStatus.AWAITING_HUMAN_APPROVAL
    assert result.reason_code == "contract_not_approved"
    assert result.differential_evidence is None
    assert not (tmp_path / result.run_id / "artifacts" / "generated").exists()


def test_out_of_order_transition_does_not_corrupt_prepared_state(
    tmp_path: Path,
) -> None:
    """Finalizing an out-of-order call would prevent the valid approval that follows."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-transition",
    )
    prepared = workflow.prepare()

    with pytest.raises(WorkflowTransitionError, match="requires approved"):
        workflow.generate()
    with pytest.raises(WorkflowTransitionError, match="before finalization"):
        workflow.result()

    workflow.approve_contract(prepared.contract, prepared.gherkin)
    assert workflow.generate().validation.approved is True
    with pytest.raises(WorkflowTransitionError, match="requires approved"):
        workflow.generate()


def test_every_public_operation_enforces_its_exact_predecessor(
    tmp_path: Path,
) -> None:
    """Skipping or repeating any state transition must make this test fail."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-exact-predecessors",
    )
    with pytest.raises(WorkflowTransitionError, match="requires prepared"):
        workflow.approve_contract(
            RiskContract.model_validate(_fixture_payload("approved_contract.json")),
            "not prepared",
        )
    with pytest.raises(WorkflowTransitionError, match="requires validated"):
        workflow.execute(repeat_count=1)
    prepared = workflow.prepare()
    with pytest.raises(WorkflowTransitionError, match="requires new"):
        workflow.prepare()
    workflow.approve_contract(prepared.contract, prepared.gherkin)
    with pytest.raises(WorkflowTransitionError, match="requires prepared"):
        workflow.approve_contract(prepared.contract, prepared.gherkin)
    with pytest.raises(WorkflowTransitionError, match="requires validated"):
        workflow.execute(repeat_count=1)


@pytest.mark.parametrize(
    ("error_type", "reason_code"),
    [
        (ExecutionTimeoutError, "generated_test_timeout"),
        (MissingObservationError, "missing_runtime_observation"),
    ],
)
def test_execution_boundary_failure_is_inconclusive_and_retains_logs(
    tmp_path: Path,
    error_type: type[Exception],
    reason_code: str,
) -> None:
    """Turning timeout or missing facts into security evidence must fail this test."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id=f"run-{reason_code}",
        runner_factory=_failing_runner_factory(error_type),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(error_type):
        workflow.execute(repeat_count=1)

    result = workflow.result()
    assert result.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert result.reason_code == reason_code
    assert result.differential_evidence is None
    assert len(servers) == 2
    assert all(server.entered and server.stopped for server in servers)
    finalized_artifact = next(
        (tmp_path / result.run_id / "artifacts" / "finalization_intent").iterdir()
    )
    payload = json.loads(finalized_artifact.read_text(encoding="utf-8"))
    paths = payload["failure"]["details"]["execution_artifacts"]
    assert Path(paths[0]["stdout_path"]).read_text(encoding="utf-8") == (
        "retained stdout\n"
    )


@pytest.mark.parametrize(
    (
        "base_behavior",
        "candidate_behavior",
        "expected_status",
        "base_severity_status",
        "candidate_severity_status",
    ),
    [
        (
            "secure",
            "vulnerable",
            WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
            "not_scored",
            "provisional",
        ),
        (
            "vulnerable",
            "secure",
            WorkflowStatus.CANDIDATE_FIX_OBSERVED,
            "provisional",
            "not_scored",
        ),
        (
            "secure",
            "secure",
            WorkflowStatus.NO_REGRESSION_OBSERVED,
            "not_scored",
            "not_scored",
        ),
        (
            "vulnerable",
            "vulnerable",
            WorkflowStatus.PRE_EXISTING_RISK_OBSERVED,
            "provisional",
            "provisional",
        ),
    ],
)
def test_execution_records_severity_only_for_vulnerable_revisions(
    tmp_path: Path,
    base_behavior: str,
    candidate_behavior: str,
    expected_status: WorkflowStatus,
    base_severity_status: str,
    candidate_severity_status: str,
) -> None:
    """Omitting severity or scoring the secure base must fail this workflow test."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id=f"run-{base_behavior}-{candidate_behavior}-severity",
        runner_factory=_behavior_runner_factory(
            [base_behavior], [candidate_behavior]
        ),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    result = workflow.execute(repeat_count=1)

    assert result.status is expected_status
    assert result.severity_assessment is not None
    assert result.severity_assessment.base.status == base_severity_status
    assert result.severity_assessment.candidate.status == candidate_severity_status
    for side in (
        result.severity_assessment.base,
        result.severity_assessment.candidate,
    ):
        if side.status == "provisional":
            assert side.score == 7.1
            assert side.severity == "High"
        else:
            assert side.reason_code == "tested_vulnerability_not_observed"
            assert side.score is None


def test_unstable_repetitions_use_all_fresh_pairs_before_classification(
    tmp_path: Path,
) -> None:
    """Reusing servers or classifying only run 1 must make this test fail."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-unstable",
        runner_factory=_behavior_runner_factory(
            ["secure", "secure", "secure"],
            ["vulnerable", "secure", "vulnerable"],
        ),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    result = workflow.execute(repeat_count=3)

    assert result.status is WorkflowStatus.UNSTABLE_RESULT
    assert result.reason_code == "security_relevant_tuple_unstable"
    assert result.stable is False
    assert result.differential_evidence is not None
    assert result.differential_evidence.candidate_differing_run_indexes == [2]
    assert result.severity_assessment is not None
    assert result.severity_assessment.base.status == "not_scored"
    assert result.severity_assessment.candidate.status == "not_scored"
    assert result.severity_assessment.base.reason_code == (
        "insufficient_evidence_for_severity"
    )
    assert result.severity_assessment.candidate.reason_code == (
        "insufficient_evidence_for_severity"
    )
    assert [server.behavior for server in servers] == [
        "secure",
        "vulnerable",
        "secure",
        "vulnerable",
        "secure",
        "vulnerable",
    ]
    assert len({id(server) for server in servers}) == 6
    assert all(server.stopped for server in servers)


def test_classifier_inconclusive_result_records_two_not_scored_assessments(
    tmp_path: Path,
) -> None:
    """Incomplete controls must not inherit a representative numeric score."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-classifier-inconclusive-severity",
        runner_factory=_behavior_runner_factory(
            ["inconclusive"], ["vulnerable"]
        ),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    result = workflow.execute(repeat_count=1)

    assert result.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert result.differential_evidence is not None
    assert result.severity_assessment is not None
    for side in (
        result.severity_assessment.base,
        result.severity_assessment.candidate,
    ):
        assert side.status == "not_scored"
        assert side.reason_code == "insufficient_evidence_for_severity"
        assert side.score is None


def test_cvss_calculator_failure_finalizes_without_a_vulnerability_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A calculator failure must preserve artifacts but abstain from scoring."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-cvss-calculator-failure",
        runner_factory=_behavior_runner_factory(["secure"], ["vulnerable"]),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    def fail_assessment(*args: Any, **kwargs: Any) -> None:
        raise CvssAssessmentError("simulated deterministic calculator failure")

    monkeypatch.setattr(
        "triageguard.workflow.vertical_slice.assess_differential_severity",
        fail_assessment,
    )

    with pytest.raises(CvssAssessmentError, match="simulated"):
        workflow.execute(repeat_count=1)

    result = workflow.result()
    assert result.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert result.reason_code == "cvss_assessment_failed"
    assert result.differential_evidence is None
    assert result.severity_assessment is None
    assert len(result.execution_manifest_sha256s) == 2


def test_partial_execution_is_retained_but_never_classified(
    tmp_path: Path,
) -> None:
    """Calling the classifier after only the base observation must fail this test."""
    servers: list[_FakeServer] = []
    classifier_calls = 0

    def forbidden_partial_classifier(*args: Any) -> Any:
        nonlocal classifier_calls
        classifier_calls += 1
        raise AssertionError("partial evidence reached classifier")

    def runner_factory(**kwargs: Any) -> _CandidateFailingRunner:
        return _CandidateFailingRunner(Path(kwargs["artifact_root"]))

    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-partial-execution",
        runner_factory=runner_factory,
        classifier=forbidden_partial_classifier,
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(MissingObservationError):
        workflow.execute(repeat_count=3)

    assert classifier_calls == 0
    assert workflow.result().reason_code == "missing_runtime_observation"
    assert all(server.stopped for server in servers)
    finalized_artifact = next(
        (
            tmp_path
            / "run-partial-execution"
            / "artifacts"
            / "finalization_intent"
        ).iterdir()
    )
    details = json.loads(finalized_artifact.read_text(encoding="utf-8"))[
        "failure"
    ]["details"]
    assert len(details["base_observations"]) == 1
    assert details["candidate_observations"] == []


@pytest.mark.parametrize(
    ("base_behavior", "candidate_behavior", "expected_status"),
    [
        ("secure", "secure", WorkflowStatus.NO_REGRESSION_OBSERVED),
        ("vulnerable", "vulnerable", WorkflowStatus.PRE_EXISTING_RISK_OBSERVED),
        ("vulnerable", "secure", WorkflowStatus.CANDIDATE_FIX_OBSERVED),
    ],
)
def test_classifier_outcomes_pass_through_without_collapsing_to_regression(
    tmp_path: Path,
    base_behavior: str,
    candidate_behavior: str,
    expected_status: WorkflowStatus,
) -> None:
    """Hard-coding candidate-regression on every complete run must fail this test."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id=f"run-{expected_status.value}",
        runner_factory=_behavior_runner_factory(
            [base_behavior], [candidate_behavior]
        ),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    result = workflow.execute(repeat_count=1)

    assert result.status is expected_status
    assert result.reason_code == expected_status.value
    assert result.stable is True


def test_repeat_count_is_explicit_positive_and_bounded_without_state_change(
    tmp_path: Path,
) -> None:
    """Accepting zero, bool, float, or unbounded repeats must make this test fail."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-repeat-bounds",
    )
    _approve(workflow)
    workflow.generate()

    for invalid in (0, 21, True, 1.0):
        with pytest.raises(ValueError, match="repeat_count"):
            workflow.execute(repeat_count=invalid)
    with pytest.raises(WorkflowTransitionError, match="before finalization"):
        workflow.result()


def test_successful_transitions_store_content_addressed_provenance(
    tmp_path: Path,
) -> None:
    """Dropping LLM responses or content hashes from transition records must fail."""
    servers: list[_FakeServer] = []
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-provenance",
        runner_factory=_behavior_runner_factory(["secure"], ["vulnerable"]),
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()
    result = workflow.execute(repeat_count=1)

    events = [
        json.loads(line)
        for line in (tmp_path / result.run_id / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    transformations = {
        event["event_type"]: event["payload"]
        for event in events
        if event["event_type"].startswith("workflow_")
    }
    assert set(transformations) == {
        "workflow_prepared",
        "workflow_approved",
        "workflow_generated",
        "workflow_validated",
        "workflow_executed",
        "workflow_finalization_intent",
    }
    for provenance in transformations.values():
        [(artifact_name, digest)] = provenance["output_hashes"].items()
        artifact = tmp_path / result.run_id / artifact_name
        assert artifact.exists()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest

    generated_name = next(
        iter(transformations["workflow_generated"]["outputs"])
    )
    generated_payload = json.loads(
        (tmp_path / result.run_id / generated_name).read_text(encoding="utf-8")
    )
    assert [
        (call["request"]["purpose"], call["response"]["provider"])
        for call in generated_payload["llm_calls"]
    ] == [("test_plan", "replay"), ("pytest_generation", "replay")]
    generated_provenance = transformations["workflow_generated"]
    attempt_times = [
        attempt[key]
        for call in generated_payload["llm_calls"]
        for attempt in call["response"]["attempts"]
        for key in ("started_at", "finished_at")
    ]
    assert generated_provenance["started_at"] <= min(attempt_times)
    assert generated_provenance["finished_at"] >= max(attempt_times)


def test_failure_finalization_does_not_mask_the_original_stage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing a replay failure with recorder failure must make this test fail."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-finalization-error",
        gateway=ReplayGateway({"test_plan": _fixture_payload("planner_response.json")}),
    )
    _approve(workflow)

    def fail_finalization(run_id: str, record: Any) -> None:
        raise OSError("simulated recorder finalization failure")

    monkeypatch.setattr(workflow._recorder, "finalize_run", fail_finalization)

    with pytest.raises(ReplayResponseMissing):
        workflow.generate()


def test_failure_finalization_can_recover_after_recorder_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing duplicate transition artifacts must not block Task 2 recovery."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-finalization-recovery",
        gateway=ReplayGateway({"test_plan": _fixture_payload("planner_response.json")}),
    )
    _approve(workflow)
    real_finalize = workflow._recorder.finalize_run
    attempts = 0

    def interrupt_once(run_id: str, record: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated recorder interruption")
        return real_finalize(run_id, record)

    monkeypatch.setattr(workflow._recorder, "finalize_run", interrupt_once)

    with pytest.raises(ReplayResponseMissing):
        workflow.generate()
    with pytest.raises(WorkflowTransitionError, match="before finalization"):
        workflow.result()
    with pytest.raises(ReplayResponseMissing):
        workflow.generate()

    assert attempts == 2
    assert workflow.result().reason_code == "replay_response_missing"
    assert (tmp_path / "run-finalization-recovery" / "run_record.json").exists()
    events = [
        json.loads(line)
        for line in (tmp_path / "run-finalization-recovery" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(
        event["event_type"] == "workflow_finalization_intent"
        for event in events
    ) == 1


def test_prepared_return_value_cannot_mutate_the_approval_snapshot(
    tmp_path: Path,
) -> None:
    """Shallow frozen models must not let caller list edits rewrite prepared truth."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-prepared-snapshot",
    )
    prepared = workflow.prepare()
    prepared.contract.cleanup.append("silently broaden cleanup authority")

    with pytest.raises(ContractApprovalError, match="contract_not_approved"):
        workflow.approve_contract(prepared.contract, prepared.gherkin)

    assert workflow.result().reason_code == "contract_not_approved"


def test_generated_return_value_cannot_mutate_execution_inputs(
    tmp_path: Path,
) -> None:
    """Caller mutation of a shallow plan list must not alter the executed snapshot."""
    servers: list[_FakeServer] = []
    received_cleanup_lengths: list[int] = []

    def runner_factory(**kwargs: Any) -> _BehaviorRunner:
        received_cleanup_lengths.append(len(kwargs["plan"].cleanup))
        return _BehaviorRunner(
            artifact_root=Path(kwargs["artifact_root"]),
            base_behaviors=["secure"], candidate_behaviors=["vulnerable"]
        )

    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-generated-snapshot",
        runner_factory=runner_factory,
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    returned = workflow.generate()
    returned.plan.cleanup.clear()

    result = workflow.execute(repeat_count=1)

    assert received_cleanup_lengths == [2]
    assert result.status is WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED


def test_revision_labels_must_be_canonical_and_distinct(tmp_path: Path) -> None:
    """Ambiguous display labels must not enter contract-bound observations."""
    with pytest.raises(ValueError):
        build_replay_workflow(
            artifact_root=tmp_path / "invalid",
            fixture_directory=FIXTURE_ROOT,
            base_revision="Main Branch",
        )
    with pytest.raises(ValueError, match="must differ"):
        build_replay_workflow(
            artifact_root=tmp_path / "same",
            fixture_directory=FIXTURE_ROOT,
            base_revision="same-revision",
            candidate_revision="same-revision",
        )


def test_malformed_contract_expectation_is_finalized_as_unapproved(
    tmp_path: Path,
) -> None:
    """A malformed secure oracle must not escape without a readable run record."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-malformed-approval",
    )
    prepared = workflow.prepare()
    malformed = RiskContract.model_validate(
        {
            **prepared.contract.model_dump(mode="json"),
            "secure_expectation": "the request is denied",
        }
    )

    with pytest.raises(ContractApprovalError):
        workflow.approve_contract(malformed, prepared.gherkin)

    assert workflow.result().status is WorkflowStatus.AWAITING_HUMAN_APPROVAL
    assert workflow.result().reason_code == "gherkin_alignment_failed"


def test_completed_recorder_finalization_is_resumed_after_post_commit_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public retry must reconcile terminal proof after post-commit interruption."""
    workflow = build_replay_workflow(
        artifact_root=tmp_path,
        fixture_directory=FIXTURE_ROOT,
        run_id="run-post-commit-interruption",
        gateway=ReplayGateway({"test_plan": _fixture_payload("planner_response.json")}),
    )
    _approve(workflow)
    real_finalize = workflow._recorder.finalize_run

    def commit_then_interrupt(run_id: str, record: Any) -> None:
        real_finalize(run_id, record)
        raise OSError("simulated interruption after durable commit")

    monkeypatch.setattr(workflow._recorder, "finalize_run", commit_then_interrupt)

    with pytest.raises(ReplayResponseMissing):
        workflow.generate()
    with pytest.raises(WorkflowTransitionError, match="before finalization"):
        workflow.result()
    with pytest.raises(ReplayResponseMissing):
        workflow.generate()

    assert workflow.result().reason_code == "replay_response_missing"


@pytest.mark.parametrize("timing", ["before_commit", "after_commit"])
@pytest.mark.parametrize(
    ("stage", "event_type", "artifact_directory"),
    [
        ("approval", "workflow_approved", "approved"),
        ("generation", "workflow_generated", "generated"),
        ("validation", "workflow_validated", "validated"),
        ("execution", "workflow_executed", "executed"),
    ],
)
def test_nonfinal_transition_recovery_reuses_exact_stage_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
    stage: str,
    event_type: str,
    artifact_directory: str,
) -> None:
    """Recorder interruption must not repeat model calls or experiments."""
    recorder = ArtifactRecorder(tmp_path)
    gateway = _CountingGateway()
    servers: list[_FakeServer] = []
    validator_calls = 0
    runner_calls = 0
    classifier_calls = 0

    def validator(*args: Any) -> Any:
        nonlocal validator_calls
        validator_calls += 1
        return validate_generated_code(*args)

    class CountingRunner(_BehaviorRunner):
        def run(self, target: Any) -> RuntimeObservation:
            nonlocal runner_calls
            runner_calls += 1
            return super().run(target)

    def runner_factory(**kwargs: Any) -> CountingRunner:
        return CountingRunner(
            artifact_root=Path(kwargs["artifact_root"]),
            base_behaviors=["secure"], candidate_behaviors=["vulnerable"]
        )

    def classifier(*args: Any) -> Any:
        nonlocal classifier_calls
        classifier_calls += 1
        return classify_differential(*args)

    workflow = _direct_workflow(
        tmp_path,
        run_id=f"run-recover-{stage}-{timing}",
        gateway=gateway,
        recorder=recorder,
        validator=validator,
        runner_factory=runner_factory,
        classifier=classifier,
        server_factory=_server_factory(servers),
    )
    prepared = workflow.prepare()
    if stage != "approval":
        workflow.approve_contract(prepared.contract, prepared.gherkin)
    if stage == "execution":
        workflow.generate()

    _install_transition_fault(
        monkeypatch,
        recorder,
        target_event=event_type,
        timing=timing,
    )

    if stage == "approval":
        operation = lambda: workflow.approve_contract(
            prepared.contract, prepared.gherkin
        )
    elif stage in {"generation", "validation"}:
        operation = workflow.generate
    else:
        operation = lambda: workflow.execute(repeat_count=1)

    with pytest.raises(OSError, match="recorder interruption"):
        operation()
    recovered = operation()

    run_directory = tmp_path / f"run-recover-{stage}-{timing}"
    events = [
        json.loads(line)
        for line in (run_directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(event["event_type"] == event_type for event in events) == 1
    assert len(list((run_directory / "artifacts" / artifact_directory).iterdir())) == 1
    for journal_event in ("artifact_write_started", "artifact_write_completed"):
        assert sum(
            event["event_type"] == journal_event
            and event["payload"].get("artifact_name", "").startswith(
                f"artifacts/{artifact_directory}/"
            )
            for event in events
        ) == 1
    assert sum(
        event["event_type"] == "contract_approved" for event in events
    ) == 1
    assert gateway.purposes == (
        [] if stage == "approval" else ["test_plan", "pytest_generation"]
    )
    assert validator_calls == (0 if stage == "approval" else 1)
    assert runner_calls == (2 if stage == "execution" else 0)
    assert classifier_calls == (1 if stage == "execution" else 0)
    assert len(servers) == (2 if stage == "execution" else 0)
    if stage == "execution":
        assert recovered.status is WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED


@pytest.mark.parametrize("timing", ["before_commit", "after_commit"])
def test_success_finalization_is_truthful_and_publicly_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
) -> None:
    """Finalization failure must never rerun an already completed experiment."""
    recorder = ArtifactRecorder(tmp_path)
    gateway = _CountingGateway()
    servers: list[_FakeServer] = []
    runner_calls = 0
    real_finalize = recorder.finalize_run
    triggered = False

    class CountingRunner(_BehaviorRunner):
        def run(self, target: Any) -> RuntimeObservation:
            nonlocal runner_calls
            runner_calls += 1
            return super().run(target)

    def runner_factory(**kwargs: Any) -> CountingRunner:
        return CountingRunner(
            artifact_root=Path(kwargs["artifact_root"]),
            base_behaviors=["secure"], candidate_behaviors=["vulnerable"]
        )

    def interrupt_finalization(run_id: str, record: Any) -> Any:
        nonlocal triggered
        if not triggered:
            triggered = True
            if timing == "after_commit":
                real_finalize(run_id, record)
            raise OSError(f"simulated {timing} finalization interruption")
        return real_finalize(run_id, record)

    monkeypatch.setattr(recorder, "finalize_run", interrupt_finalization)
    workflow = _direct_workflow(
        tmp_path,
        run_id=f"run-finalize-{timing}",
        gateway=gateway,
        recorder=recorder,
        runner_factory=runner_factory,
        server_factory=_server_factory(servers),
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(OSError, match="finalization interruption"):
        workflow.execute(repeat_count=1)
    with pytest.raises(WorkflowTransitionError, match="before finalization"):
        workflow.result()

    result = workflow.execute(repeat_count=1)

    assert result.status is WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED
    assert runner_calls == 2
    assert len(servers) == 2
    run_directory = tmp_path / f"run-finalize-{timing}"
    events = [
        json.loads(line)
        for line in (run_directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(
        event["event_type"] == "workflow_finalization_intent"
        for event in events
    ) == 1
    assert not any(event["event_type"] == "workflow_finalized" for event in events)
    assert sum(
        event["event_type"] == "finalization_completed" for event in events
    ) == 1
    assert sum(
        event["event_type"] == "finalization_started" for event in events
    ) == 1


def test_validator_exception_finalizes_validation_failure_and_is_reraised(
    tmp_path: Path,
) -> None:
    """A validator exception must not leave generated state without a result."""
    gateway = _CountingGateway()

    class ValidatorBoundaryError(RuntimeError):
        pass

    def failing_validator(*args: Any) -> Any:
        raise ValidatorBoundaryError("validator boundary failed")

    workflow = _direct_workflow(
        tmp_path,
        run_id="run-validator-boundary",
        gateway=gateway,
        validator=failing_validator,
    )
    _approve(workflow)

    with pytest.raises(ValidatorBoundaryError):
        workflow.generate()

    result = workflow.result()
    assert result.status is WorkflowStatus.VALIDATION_FAILED
    assert result.reason_code == "validator_invocation_failed"
    assert gateway.purposes == ["test_plan", "pytest_generation"]


def test_validator_rejects_untyped_result_with_honest_failure_record(
    tmp_path: Path,
) -> None:
    """A validator double returning untyped data must not create validated state."""
    gateway = _CountingGateway()

    def invalid_validator(*args: Any) -> Any:
        return {"approved": True}

    workflow = _direct_workflow(
        tmp_path,
        run_id="run-validator-untyped",
        gateway=gateway,
        validator=invalid_validator,
    )
    _approve(workflow)

    with pytest.raises(TypeError, match="CodeValidationReport"):
        workflow.generate()

    assert workflow.result().status is WorkflowStatus.VALIDATION_FAILED
    assert workflow.result().reason_code == "validator_invocation_failed"


def test_runner_constructor_exception_finalizes_without_classification(
    tmp_path: Path,
) -> None:
    """Runner construction failure must be inconclusive and preserve its type."""
    gateway = _CountingGateway()
    classifier_calls = 0
    constructor_calls = 0

    class RunnerConstructionError(RuntimeError):
        pass

    def failing_runner_factory(**kwargs: Any) -> Any:
        nonlocal constructor_calls
        constructor_calls += 1
        raise RunnerConstructionError("runner construction failed")

    def forbidden_classifier(*args: Any) -> Any:
        nonlocal classifier_calls
        classifier_calls += 1
        raise AssertionError("classifier must not run")

    workflow = _direct_workflow(
        tmp_path,
        run_id="run-runner-construction",
        gateway=gateway,
        runner_factory=failing_runner_factory,
        classifier=forbidden_classifier,
    )
    _approve(workflow)
    workflow.generate()

    with pytest.raises(RunnerConstructionError):
        workflow.execute(repeat_count=1)

    result = workflow.result()
    assert result.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert result.reason_code == "execution_runner_construction_failed"
    assert constructor_calls == 1
    assert classifier_calls == 0


def test_replay_builder_rejects_arbitrary_gateway_before_any_call(
    tmp_path: Path,
) -> None:
    """A replay mode label must not authorize an arbitrary network gateway."""
    gateway = _CountingGateway()

    with pytest.raises(ValueError, match="ReplayGateway"):
        build_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            gateway=gateway,
        )

    assert gateway.purposes == []
    assert list(tmp_path.iterdir()) == []


def test_replay_builder_rejects_live_settings_without_constructing_a_run(
    tmp_path: Path,
) -> None:
    """Live settings must fail closed at the replay builder boundary."""
    settings = Settings(llm_mode="live", groq_api_key="not-used")

    with pytest.raises(ValueError, match="replay settings"):
        build_replay_workflow(
            artifact_root=tmp_path,
            fixture_directory=FIXTURE_ROOT,
            settings=settings,
        )

    assert list(tmp_path.iterdir()) == []


def test_prepare_freezes_a_known_run_id_collision_without_adopting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known initial collision must never become locate-based recovery."""
    run_id = "run-preexisting-collision"
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(run_id, RunOwnership.issue(run_id))
    run_directory = recorder.verify_run_handle(handle)
    recorder.record_event(
        handle,
        LifecycleEventType.CONTRACT_APPROVED,
        {"id": "foreign-contract"},
    )
    (run_directory / "foreign-state.bin").write_bytes(b"do-not-adopt-or-mutate")
    before = _directory_fingerprint(run_directory)
    real_start = recorder.start_run
    real_locate = recorder.locate_run
    start_calls = 0
    locate_calls = 0

    def counting_start(value: str, ownership: RunOwnership) -> Path:
        nonlocal start_calls
        start_calls += 1
        return real_start(value, ownership)

    def counting_locate(value: str) -> Path:
        nonlocal locate_calls
        locate_calls += 1
        return real_locate(value)

    monkeypatch.setattr(recorder, "start_run", counting_start)
    monkeypatch.setattr(recorder, "locate_run", counting_locate)
    workflow = _direct_workflow(
        tmp_path,
        run_id=run_id,
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    outcomes: list[object] = []
    for _ in range(3):
        try:
            outcomes.append(workflow.prepare())
        except Exception as error:  # noqa: BLE001 - capture every retry outcome
            outcomes.append(error)

    assert all(isinstance(outcome, FileExistsError) for outcome in outcomes)
    assert outcomes[1] is outcomes[0]
    assert outcomes[2] is outcomes[0]
    assert start_calls == 1
    assert locate_calls == 0
    assert _directory_fingerprint(run_directory) == before


def test_prepare_ambiguous_precommit_start_safely_retries_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambiguous start error with no directory may retry exclusive creation."""
    recorder = ArtifactRecorder(tmp_path)
    real_start = recorder.start_run
    real_locate = recorder.locate_run
    start_calls = 0
    locate_calls = 0

    def interrupt_before_create(
        run_id: str, ownership: RunOwnership
    ) -> Path:
        nonlocal start_calls
        start_calls += 1
        if start_calls == 1:
            raise OSError("ambiguous precommit interruption")
        return real_start(run_id, ownership)

    def counting_locate(run_id: str) -> Path:
        nonlocal locate_calls
        locate_calls += 1
        return real_locate(run_id)

    monkeypatch.setattr(recorder, "start_run", interrupt_before_create)
    monkeypatch.setattr(recorder, "locate_run", counting_locate)
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-ambiguous-precommit",
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    with pytest.raises(OSError, match="ambiguous precommit"):
        workflow.prepare()
    prepared = workflow.prepare()

    assert prepared.run_id == "run-ambiguous-precommit"
    assert start_calls == 2
    assert locate_calls == 1
    assert (tmp_path / prepared.run_id / "events.jsonl").exists()


def test_prepare_ambiguous_start_rejects_an_empty_foreign_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-ID directory existence alone must never prove workflow ownership."""
    recorder = ArtifactRecorder(tmp_path)
    interrupted = False

    def create_empty_foreign_directory_then_interrupt(
        run_id: str, ownership: RunOwnership
    ) -> Path:
        nonlocal interrupted
        assert not interrupted
        interrupted = True
        (tmp_path / run_id).mkdir()
        raise OSError("ambiguous post-mkdir interruption")

    monkeypatch.setattr(
        recorder,
        "start_run",
        create_empty_foreign_directory_then_interrupt,
    )
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-ambiguous-empty-foreign",
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    with pytest.raises(OSError, match="post-mkdir"):
        workflow.prepare()
    run_directory = tmp_path / "run-ambiguous-empty-foreign"
    before = _directory_fingerprint(run_directory)

    with pytest.raises(WorkflowTransitionError, match="identity"):
        workflow.prepare()

    assert _directory_fingerprint(run_directory) == before == {}


def test_prepare_crash_after_mkdir_before_ownership_marker_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorder crash before its durable proof must leave an unadopted directory."""
    recorder = ArtifactRecorder(tmp_path)
    real_atomic_write = recorder._atomic_write_file
    marker_attempts = 0

    def interrupt_before_marker(
        directory_fd: int,
        name: str,
        content: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        nonlocal marker_attempts
        if name == ".run-ownership.json":
            marker_attempts += 1
            raise OSError("crashed before ownership marker")
        real_atomic_write(directory_fd, name, content, mode=mode)

    monkeypatch.setattr(recorder, "_atomic_write_file", interrupt_before_marker)
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-crash-before-ownership-marker",
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    with pytest.raises(OSError, match="before ownership marker"):
        workflow.prepare()
    run_directory = tmp_path / "run-crash-before-ownership-marker"
    before = _directory_fingerprint(run_directory)

    for _ in range(2):
        with pytest.raises(WorkflowTransitionError, match="identity"):
            workflow.prepare()
        assert _directory_fingerprint(run_directory) == before == {
            ".run.lock": b"",
            ".workflow.lock": b"",
        }
    assert marker_attempts == 1


@pytest.mark.parametrize(
    "marker_case",
    ["malformed", "extra", "coerced", "mismatched", "noncanonical"],
)
def test_prepare_rejects_invalid_ownership_markers_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_case: str,
) -> None:
    """Only exact typed canonical metadata may authorize ambiguous recovery."""
    recorder = ArtifactRecorder(tmp_path)
    interrupted = False

    def create_invalid_marker_then_interrupt(
        run_id: str, ownership: RunOwnership
    ) -> Path:
        nonlocal interrupted
        assert not interrupted
        interrupted = True
        run_directory = tmp_path / run_id
        run_directory.mkdir()
        payload = ownership.model_dump(mode="json")
        if marker_case == "malformed":
            marker_bytes = b"{not-json\n"
        else:
            if marker_case == "extra":
                payload["extra"] = "not-allowed"
            elif marker_case == "coerced":
                payload["schema_version"] = "1"
            elif marker_case == "mismatched":
                payload["run_id"] = "foreign-run"
            if marker_case == "noncanonical":
                marker_bytes = (json.dumps(payload) + "\n").encode("utf-8")
            else:
                marker_bytes = (
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
        (run_directory / ".run-ownership.json").write_bytes(marker_bytes)
        raise OSError("ambiguous invalid-marker interruption")

    monkeypatch.setattr(
        recorder,
        "start_run",
        create_invalid_marker_then_interrupt,
    )
    run_id = f"run-invalid-marker-{marker_case}"
    workflow = _direct_workflow(
        tmp_path,
        run_id=run_id,
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    with pytest.raises(OSError, match="invalid-marker"):
        workflow.prepare()
    run_directory = tmp_path / run_id
    before = _directory_fingerprint(run_directory)

    for _ in range(2):
        with pytest.raises(WorkflowTransitionError, match="identity"):
            workflow.prepare()
        assert _directory_fingerprint(run_directory) == before


def test_prepare_ambiguous_start_rejects_foreign_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid marker carrying another token must never prove ownership."""
    recorder = ArtifactRecorder(tmp_path)
    real_start = recorder.start_run
    interrupted = False

    def create_foreign_identity_then_interrupt(
        run_id: str, ownership: RunOwnership
    ) -> Path:
        nonlocal interrupted
        if interrupted:
            return real_start(run_id, ownership)
        interrupted = True
        real_start(run_id, RunOwnership.issue(run_id))
        raise OSError("ambiguous post-create interruption")

    monkeypatch.setattr(recorder, "start_run", create_foreign_identity_then_interrupt)
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-ambiguous-foreign",
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    with pytest.raises(OSError, match="ambiguous post-create"):
        workflow.prepare()
    run_directory = tmp_path / "run-ambiguous-foreign"
    before = _directory_fingerprint(run_directory)
    for _ in range(2):
        with pytest.raises(WorkflowTransitionError, match="identity"):
            workflow.prepare()
        assert _directory_fingerprint(run_directory) == before


def test_prepare_retry_reverifies_ownership_before_any_further_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained directory path must not bypass proof checks during recovery."""
    recorder = ArtifactRecorder(tmp_path)
    real_lifecycle = recorder.record_lifecycle_event
    interrupted = False

    def commit_run_started_then_interrupt(run_id: str, event: Any) -> Any:
        nonlocal interrupted
        recorded = real_lifecycle(run_id, event)
        if event.event_type is LifecycleEventType.RUN_STARTED and not interrupted:
            interrupted = True
            raise OSError("interrupted after run_started commit")
        return recorded

    monkeypatch.setattr(
        recorder,
        "record_lifecycle_event",
        commit_run_started_then_interrupt,
    )
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-reverify-retained-ownership",
        gateway=_CountingGateway(),
        recorder=recorder,
    )

    with pytest.raises(OSError, match="after run_started"):
        workflow.prepare()
    run_directory = tmp_path / "run-reverify-retained-ownership"
    (run_directory / ".run-ownership.json").write_bytes(b"{}\n")
    before = _directory_fingerprint(run_directory)

    with pytest.raises(WorkflowTransitionError, match="identity"):
        workflow.prepare()

    assert _directory_fingerprint(run_directory) == before


@pytest.mark.parametrize(
    "boundary", ["start_run", "run_started", "prepared_artifact"]
)
@pytest.mark.parametrize("timing", ["before_commit", "after_commit"])
def test_prepare_recovery_freezes_fixtures_and_exact_recorder_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    timing: str,
) -> None:
    """Prepare retry must not reread changed fixtures or mint new provenance."""
    fixture_directory = tmp_path / "fixture-source"
    fixture_directory.mkdir()
    original_bytes = {
        name: (FIXTURE_ROOT / name).read_bytes()
        for name in (
            "approved_contract.json",
            "impact_report.json",
            "cvss_profile.json",
        )
    }
    for name, content in original_bytes.items():
        (fixture_directory / name).write_bytes(content)
    artifact_root = tmp_path / "artifacts"
    recorder = ArtifactRecorder(artifact_root)
    reads: list[str] = []

    def fixture_reader(path: Path) -> bytes:
        reads.append(path.name)
        return path.read_bytes()

    workflow = MilestoneOneWorkflow(
        fixture_directory=fixture_directory,
        settings=Settings(llm_mode="replay", artifacts_dir=artifact_root),
        gateway=_CountingGateway(),
        recorder=recorder,
        run_id=f"run-prepare-{boundary}-{timing}",
        fixture_reader=fixture_reader,
    )
    triggered = False
    if boundary == "start_run":
        real_start = recorder.start_run

        def interrupt_start(run_id: str, ownership: RunOwnership) -> Any:
            nonlocal triggered
            if not triggered:
                triggered = True
                if timing == "after_commit":
                    real_start(run_id, ownership)
                raise OSError(f"simulated {timing} start_run interruption")
            return real_start(run_id, ownership)

        monkeypatch.setattr(recorder, "start_run", interrupt_start)
    elif boundary == "run_started":
        real_lifecycle = recorder.record_lifecycle_event

        def interrupt_lifecycle(run_id: str, event: Any) -> Any:
            nonlocal triggered
            if (
                event.event_type is LifecycleEventType.RUN_STARTED
                and not triggered
            ):
                triggered = True
                if timing == "after_commit":
                    real_lifecycle(run_id, event)
                raise OSError(f"simulated {timing} run_started interruption")
            return real_lifecycle(run_id, event)

        monkeypatch.setattr(recorder, "record_lifecycle_event", interrupt_lifecycle)
    else:
        real_write = recorder.write_artifact

        def interrupt_prepared_artifact(
            run_id: str,
            name: str,
            content: bytes,
            provenance: Any,
        ) -> Any:
            nonlocal triggered
            if provenance.event_type == "workflow_prepared" and not triggered:
                triggered = True
                if timing == "after_commit":
                    real_write(run_id, name, content, provenance)
                raise OSError(f"simulated {timing} prepared artifact interruption")
            return real_write(run_id, name, content, provenance)

        monkeypatch.setattr(recorder, "write_artifact", interrupt_prepared_artifact)

    with pytest.raises(OSError, match="interruption"):
        workflow.prepare()

    snapshot = workflow._preparation_snapshot
    assert snapshot is not None
    frozen_prepared = snapshot.prepared
    frozen_transition = snapshot.transition
    fingerprint_before_alternates = _directory_fingerprint(artifact_root)
    for alternate in (
        lambda: workflow.approve_contract(
            frozen_prepared.contract, frozen_prepared.gherkin
        ),
        workflow.generate,
        lambda: workflow.execute(repeat_count=1),
        workflow.result,
    ):
        with pytest.raises(WorkflowTransitionError):
            alternate()
    assert _directory_fingerprint(artifact_root) == fingerprint_before_alternates
    assert reads == [
        "approved_contract.json",
        "impact_report.json",
        "cvss_profile.json",
    ]

    (fixture_directory / "approved_contract.json").write_text(
        '{"mutated":true}', encoding="utf-8"
    )
    (fixture_directory / "impact_report.json").unlink()
    (fixture_directory / "cvss_profile.json").write_text(
        '{"mutated":true}', encoding="utf-8"
    )

    prepared = workflow.prepare()

    assert reads == [
        "approved_contract.json",
        "impact_report.json",
        "cvss_profile.json",
    ]
    assert prepared == frozen_prepared
    assert prepared.contract == RiskContract.model_validate_json(
        original_bytes["approved_contract.json"]
    )
    assert prepared.cvss_profile == CvssProfile.model_validate_json(
        original_bytes["cvss_profile.json"]
    )
    run_directory = artifact_root / prepared.run_id
    events = [
        json.loads(line)
        for line in (run_directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(event["event_type"] == "run_started" for event in events) == 1
    assert sum(event["event_type"] == "workflow_prepared" for event in events) == 1
    assert sum(
        event["event_type"] == "artifact_write_started"
        and event["payload"].get("artifact_name")
        == frozen_transition.artifact_name
        for event in events
    ) == 1
    assert sum(
        event["event_type"] == "artifact_write_completed"
        and event["payload"].get("artifact_name")
        == frozen_transition.artifact_name
        for event in events
    ) == 1
    transition_event = next(
        event for event in events if event["event_type"] == "workflow_prepared"
    )
    assert transition_event["payload"] == frozen_transition.event.model_dump(
        mode="json"
    )
    assert (run_directory / frozen_transition.artifact_name).read_bytes() == (
        frozen_transition.content
    )
    assert (run_directory / ".run-ownership.json").read_bytes() == (
        snapshot.ownership.canonical_bytes()
    )
    run_started = next(
        event for event in events if event["event_type"] == "run_started"
    )
    assert run_started["payload"] == {
        "id": prepared.run_id,
        "ownership_token": snapshot.ownership.ownership_token,
    }


def test_interrupted_generation_failure_can_only_resume_through_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternate public methods must not finalize a generation-owned failure."""
    recorder = ArtifactRecorder(tmp_path)

    class MissingGateway:
        def __init__(self) -> None:
            self.delegate = ReplayGateway(
                {"test_plan": _fixture_payload("planner_response.json")}
            )
            self.calls: list[str] = []

        def generate(self, request: Any) -> Any:
            self.calls.append(request.purpose)
            return self.delegate.generate(request)

    gateway = MissingGateway()
    workflow = _direct_workflow(
        tmp_path,
        run_id="run-owned-generation-failure",
        gateway=gateway,
        recorder=recorder,
    )
    prepared = workflow.prepare()
    workflow.approve_contract(prepared.contract, prepared.gherkin)
    real_finalize = recorder.finalize_run
    attempts = 0

    def interrupt_once(run_id: str, record: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated finalization interruption")
        return real_finalize(run_id, record)

    monkeypatch.setattr(recorder, "finalize_run", interrupt_once)
    with pytest.raises(ReplayResponseMissing):
        workflow.generate()

    fingerprint = _directory_fingerprint(tmp_path / prepared.run_id)
    calls = list(gateway.calls)
    for alternate in (
        workflow.prepare,
        lambda: workflow.approve_contract(prepared.contract, prepared.gherkin),
        lambda: workflow.execute(repeat_count=1),
        workflow.result,
    ):
        with pytest.raises(WorkflowTransitionError):
            alternate()
        assert _directory_fingerprint(tmp_path / prepared.run_id) == fingerprint
        assert gateway.calls == calls
        assert attempts == 1

    with pytest.raises(ReplayResponseMissing):
        workflow.generate()

    assert attempts == 2
    assert gateway.calls == calls
    assert workflow.result().reason_code == "replay_response_missing"


def test_interrupted_execution_failure_can_only_resume_through_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternate methods must not finalize or repeat an execution-owned failure."""
    recorder = ArtifactRecorder(tmp_path)
    gateway = _CountingGateway()
    servers: list[_FakeServer] = []
    runner_calls = 0

    class FailingExecutionRunner(_FailingRunner):
        def run(self, target: Any) -> RuntimeObservation:
            nonlocal runner_calls
            runner_calls += 1
            return super().run(target)

    def runner_factory(**kwargs: Any) -> FailingExecutionRunner:
        return FailingExecutionRunner(
            Path(kwargs["artifact_root"]), MissingObservationError
        )

    workflow = _direct_workflow(
        tmp_path,
        run_id="run-owned-execution-failure",
        gateway=gateway,
        recorder=recorder,
        runner_factory=runner_factory,
        server_factory=_server_factory(servers),
    )
    prepared = workflow.prepare()
    workflow.approve_contract(prepared.contract, prepared.gherkin)
    workflow.generate()
    real_finalize = recorder.finalize_run
    attempts = 0

    def interrupt_once(run_id: str, record: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated finalization interruption")
        return real_finalize(run_id, record)

    monkeypatch.setattr(recorder, "finalize_run", interrupt_once)
    with pytest.raises(MissingObservationError):
        workflow.execute(repeat_count=1)

    fingerprint = _directory_fingerprint(tmp_path / prepared.run_id)
    calls = list(gateway.purposes)
    for alternate in (
        workflow.prepare,
        lambda: workflow.approve_contract(prepared.contract, prepared.gherkin),
        workflow.generate,
        workflow.result,
    ):
        with pytest.raises(WorkflowTransitionError):
            alternate()
        assert _directory_fingerprint(tmp_path / prepared.run_id) == fingerprint
        assert gateway.purposes == calls
        assert runner_calls == 1
        assert len(servers) == 2
        assert attempts == 1

    with pytest.raises(MissingObservationError):
        workflow.execute(repeat_count=1)

    assert attempts == 2
    assert runner_calls == 1
    assert len(servers) == 2
    assert workflow.result().reason_code == "missing_runtime_observation"
