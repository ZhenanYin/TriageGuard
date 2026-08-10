import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest
from pydantic import ValidationError

from triageguard.domain import EnvironmentKind, RunRecord, WorkflowStatus
from triageguard.research import (
    ArtifactRecorder,
    RecorderCorruptionError,
    RunHandle,
    RunOwnership,
)
from triageguard.research.recorder import (
    LifecycleEvent,
    LifecycleEventType,
    TransformationEvent,
)


def test_start_run_persists_an_exact_unpredictable_ownership_proof(tmp_path):
    """A same-ID directory is recoverable only with the caller's frozen proof."""
    ownership = RunOwnership.issue("run-1")
    other = RunOwnership.issue("run-1")
    recorder = ArtifactRecorder(tmp_path)

    handle = recorder.start_run("run-1", ownership)
    run_directory = recorder.verify_run_handle(handle)

    marker = run_directory / ".run-ownership.json"
    expected = (
        "{\"marker_type\":\"triageguard_run_ownership\","
        f"\"ownership_token\":\"{ownership.ownership_token}\","
        "\"run_id\":\"run-1\",\"schema_version\":1}\n"
    ).encode()
    assert marker.read_bytes() == expected
    assert len(ownership.ownership_token) == 64
    assert set(ownership.ownership_token) <= set("0123456789abcdef")
    assert other.ownership_token != ownership.ownership_token
    assert recorder.verify_run_ownership("run-1", ownership) == run_directory


def test_run_started_event_is_bound_to_the_stored_ownership(tmp_path):
    """The lifecycle start cannot claim a different or unproven ownership token."""
    ownership = RunOwnership.issue("run-1")
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run("run-1", ownership)
    event = LifecycleEvent(
        event_type=LifecycleEventType.RUN_STARTED,
        payload={
            "id": "run-1",
            "ownership_token": ownership.ownership_token,
        },
    )

    recorded = recorder.record_lifecycle_event(handle, event)

    assert recorded.payload == event.payload


def test_run_started_rejects_a_foreign_ownership_without_appending(tmp_path):
    """A same-run event with another valid token must not claim lifecycle ownership."""
    ownership = RunOwnership.issue("run-1")
    foreign = RunOwnership.issue("run-1")
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run("run-1", ownership)
    event = LifecycleEvent(
        event_type=LifecycleEventType.RUN_STARTED,
        payload={"id": "run-1", "ownership_token": foreign.ownership_token},
    )

    with pytest.raises(ValueError, match="does not match"):
        recorder.record_lifecycle_event(handle, event)

    assert not (tmp_path / "run-1" / "events.jsonl").exists()
    assert recorder.verify_run_ownership("run-1", ownership) == tmp_path / "run-1"


def test_recorder_appends_events_and_hashes_artifacts(tmp_path):
    """A missing append or SHA-256 digest would make the provenance incomplete."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    first = recorder.record_event(handle, "contract_approved", {"id": "c1"})
    artifact = recorder.write_artifact(
        handle, "contract.json", b'{"id":"c1"}', _artifact_transformation_event()
    )
    events = _events(tmp_path)

    assert first.sequence == 1
    assert len(artifact.sha256) == 64
    assert (tmp_path / "run-1" / "events.jsonl").read_text().count("\n") == 3
    assert events[-2]["event_type"] == "artifact_write_started"
    assert events[-2]["payload"]["provenance"]["outputs"]["contract.json"] == "contract.json"
    assert (
        events[-2]["payload"]["provenance"]["output_hashes"]["contract.json"]
        == artifact.sha256
    )
    assert events[-1]["event_type"] == "artifact_write_completed"


def test_run_started_identity_must_match_the_recorder_run(tmp_path):
    """A typed event for another run must not contaminate this run's log."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    event = LifecycleEvent(
        event_type=LifecycleEventType.RUN_STARTED,
        payload={
            "id": "run-2",
            "ownership_token": handle.ownership.ownership_token,
        },
    )

    with pytest.raises(ValueError, match="RunHandle.run_id"):
        recorder.record_lifecycle_event(handle, event)

    assert not (tmp_path / "run-1" / "events.jsonl").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "id": "run-1",
            "ownership_token": "a" * 64,
            "extra": "not-allowed",
        },
        {"id": b"run-1", "ownership_token": "a" * 64},
        {"id": "run-1"},
    ],
)
def test_run_started_identity_payload_is_exact_and_strict(tmp_path, payload):
    """Extra fields or coercible non-string identities must fail before append."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    with pytest.raises(ValidationError):
        recorder.record_event(handle, LifecycleEventType.RUN_STARTED, payload)

    assert not (tmp_path / "run-1" / "events.jsonl").exists()


def test_recorder_refuses_to_overwrite_artifact(tmp_path):
    """Replacing recorded bytes would destroy the append-only evidence trail."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    recorder.write_artifact(handle, "contract.json", b"first", _artifact_transformation_event())

    with pytest.raises(RecorderCorruptionError, match="conflicting artifact intent"):
        recorder.write_artifact(
            handle, "contract.json", b"second", _artifact_transformation_event()
        )


def test_recorder_finalizes_a_run_once_with_a_hashed_record(tmp_path):
    """A second finalization or a missing record hash would make conclusions mutable."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    record = RunRecord(
        run_id="run-1",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        base_revision="base-revision",
        candidate_revision="candidate-revision",
        status=WorkflowStatus.VALIDATED_EVIDENCE,
        reason_code="evidence_complete",
        explanation="The controlled fixture produced attributable evidence.",
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        finished_at=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
    )

    artifact = recorder.finalize_run(handle, record)
    events = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "events.jsonl").read_text().splitlines()
    ]

    assert artifact.sha256 == hashlib.sha256(
        (tmp_path / "run-1" / "run_record.json").read_bytes()
    ).hexdigest()
    assert events[-1]["event_type"] == "finalization_completed"
    assert events[-1]["payload"] == {"record_sha256": artifact.sha256}
    assert recorder.finalize_run(handle, record) == artifact


def test_recorder_rejects_artifact_paths_outside_the_run_directory(tmp_path):
    """An escaping name could overwrite evidence belonging to a different run."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    with pytest.raises(ValueError):
        recorder.write_artifact(
            handle, "../other-run.json", b"unsafe", _artifact_transformation_event()
        )


@pytest.mark.parametrize("name", [".", "child/.."])
def test_recorder_rejects_artifact_names_without_a_file_path(tmp_path, name):
    """A name resolving to the run directory is not a writable artifact path."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    with pytest.raises(ValueError, match="normalized relative file path"):
        recorder.write_artifact(
            handle, name, b"unsafe", _artifact_transformation_event()
        )


def test_recorder_refuses_artifacts_without_typed_provenance(tmp_path):
    """Artifact bytes must never be persisted outside a typed transformation event."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    with pytest.raises(TypeError):
        recorder.write_artifact(handle, "contract.json", b"unsafe")

    assert not (tmp_path / "run-1" / "contract.json").exists()


def test_transformation_events_require_complete_typed_provenance(tmp_path):
    """A transformation cannot be recorded without its inputs, outputs, and evidence."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    with pytest.raises(TypeError):
        recorder.record_transformation(handle, {"event_type": "gherkin_rendered"})
    with pytest.raises(ValidationError):
        TransformationEvent(
            event_type="gherkin_rendered",
            inputs={"contract_id": "c1"},
            outputs={"scenario": "s1"},
            input_hashes={"contract.json": "a" * 64},
            versions={"renderer": "1.0"},
            started_at=datetime(2026, 8, 7, tzinfo=UTC),
            finished_at=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
            reason_code="rendered",
        )
    with pytest.raises(ValidationError):
        TransformationEvent(
            event_type="gherkin_rendered",
            inputs={"contract_id": "c1"},
            outputs={"scenario": "s1"},
            input_hashes={"contract.json": "a" * 64},
            output_hashes={"scenario.feature": "b" * 64},
            versions={"renderer": "1.0"},
            started_at=datetime(2026, 8, 7, tzinfo=UTC),
            finished_at=datetime(2026, 8, 7, tzinfo=UTC),
            reason_code="rendered",
        )


def test_transformation_events_store_typed_provenance_automatically(tmp_path):
    """A transformation event must persist every typed provenance field."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    event = recorder.record_transformation(handle, _transformation_event())

    assert event.event_type == "gherkin_rendered"
    assert event.payload == _transformation_event().model_dump(mode="json")


@pytest.mark.parametrize(
    "name",
    [
        "events.jsonl",
        "events.jsonl/child",
        "run_record.json",
        "run_record.json/child",
        ".run-ownership.json",
        ".run-ownership.json/child",
        ".finalization.lock",
        ".finalization.lock/child",
        ".artifact-locks",
        ".artifact-locks/child",
    ],
)
def test_recorder_refuses_public_writes_to_owned_artifacts(tmp_path, name):
    """Public artifact writes must not replace the recorder's journal or final record."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)

    with pytest.raises(ValueError):
        recorder.write_artifact(handle, name, b"unsafe", _artifact_transformation_event())


def test_finalization_recovers_after_completion_event_failure(tmp_path, monkeypatch):
    """A retry completes a recorded finalization when the durable record hash matches."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    record = _run_record()
    original_append_event = recorder._append_event_locked

    def fail_first_completion(run_fd, event_type, payload):
        if event_type == LifecycleEventType.FINALIZATION_COMPLETED.value:
            monkeypatch.setattr(recorder, "_append_event_locked", original_append_event)
            raise OSError("injected interruption after run record write")
        return original_append_event(run_fd, event_type, payload)

    monkeypatch.setattr(recorder, "_append_event_locked", fail_first_completion)
    with pytest.raises(OSError, match="injected interruption"):
        recorder.finalize_run(handle, record)

    artifact = recorder.finalize_run(handle, record)
    events = _events(tmp_path)

    assert artifact.sha256 == hashlib.sha256(
        (tmp_path / "run-1" / "run_record.json").read_bytes()
    ).hexdigest()
    assert [event["event_type"] for event in events].count("finalization_started") == 1
    assert [event["event_type"] for event in events].count("finalization_completed") == 1


def test_finalization_recovery_rejects_a_mismatched_existing_record(tmp_path):
    """A retry must not certify a final record whose bytes do not match the journal hash."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    recorder.finalize_run(handle, _run_record())
    (tmp_path / "run-1" / "run_record.json").write_bytes(b"different record")

    with pytest.raises(RecorderCorruptionError, match="does not match"):
        recorder.finalize_run(handle, _run_record())


def test_artifact_started_event_failure_leaves_no_artifact_bytes(tmp_path, monkeypatch):
    """A failed durable start record must prevent creation of unprovenanced bytes."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    original_append_event = recorder._append_event_locked

    def fail_started_event(run_fd, event_type, payload):
        if event_type == "artifact_write_started":
            raise OSError("injected start-record failure")
        return original_append_event(run_fd, event_type, payload)

    monkeypatch.setattr(recorder, "_append_event_locked", fail_started_event)
    with pytest.raises(OSError, match="start-record failure"):
        recorder.write_artifact(
            handle, "contract.json", b'{"id":"c1"}', _artifact_transformation_event()
        )

    assert not (tmp_path / "run-1" / "contract.json").exists()
    assert not (tmp_path / "run-1" / "events.jsonl").exists()


def test_artifact_retry_completes_without_duplicate_journal_records(tmp_path, monkeypatch):
    """A post-write interruption recovers the matching journal without duplicate completion."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start_run(recorder)
    original_append_event = recorder._append_event_locked

    def fail_first_completion(run_fd, event_type, payload):
        if event_type == "artifact_write_completed":
            monkeypatch.setattr(recorder, "_append_event_locked", original_append_event)
            raise OSError("injected completion-record failure")
        return original_append_event(run_fd, event_type, payload)

    monkeypatch.setattr(recorder, "_append_event_locked", fail_first_completion)
    with pytest.raises(OSError, match="completion-record failure"):
        recorder.write_artifact(
            handle, "contract.json", b'{"id":"c1"}', _artifact_transformation_event()
        )

    artifact = recorder.write_artifact(
        handle, "contract.json", b'{"id":"c1"}', _artifact_transformation_event()
    )
    events = _events(tmp_path)

    assert artifact.sha256 == hashlib.sha256(b'{"id":"c1"}').hexdigest()
    assert [event["event_type"] for event in events].count("artifact_write_started") == 1
    assert [event["event_type"] for event in events].count("artifact_write_completed") == 1


def test_concurrent_finalization_has_one_completion_and_two_idempotent_results(tmp_path):
    """The finalization journal must serialize simultaneous completion attempts."""
    first = ArtifactRecorder(tmp_path)
    second = ArtifactRecorder(tmp_path)
    handle = _start_run(first)
    start_barrier = Barrier(2)

    def finalize(recorder):
        start_barrier.wait()
        return recorder.finalize_run(handle, _run_record())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(finalize, recorder) for recorder in (first, second)]
        results = [future.result() if not future.exception() else future.exception() for future in futures]

    assert sum(not isinstance(result, BaseException) for result in results) == 2
    assert [event["event_type"] for event in _events(tmp_path)].count(
        "finalization_completed"
    ) == 1


def test_concurrent_artifact_write_has_one_journal_and_two_idempotent_results(tmp_path):
    """The artifact journal must serialize simultaneous writes across recorder instances."""
    first = ArtifactRecorder(tmp_path)
    second = ArtifactRecorder(tmp_path)
    handle = _start_run(first)
    start_barrier = Barrier(2)

    def write(recorder):
        start_barrier.wait()
        return recorder.write_artifact(
            handle, "contract.json", b'{"id":"c1"}', _artifact_transformation_event()
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write, recorder) for recorder in (first, second)]
        results = [
            future.result() if not future.exception() else future.exception()
            for future in futures
        ]

    assert sum(not isinstance(result, BaseException) for result in results) == 2
    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert event_types.count("artifact_write_started") == 1
    assert event_types.count("artifact_write_completed") == 1


def _transformation_event() -> TransformationEvent:
    return TransformationEvent(
        event_type="gherkin_rendered",
        inputs={"contract_id": "c1"},
        outputs={"scenario": "s1"},
        input_hashes={"contract.json": "a" * 64},
        output_hashes={"scenario.feature": "b" * 64},
        versions={"renderer": "1.0"},
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        finished_at=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
        reason_code="rendered",
    )


def _artifact_transformation_event() -> TransformationEvent:
    return TransformationEvent(
        event_type="contract_serialized",
        inputs={"contract_id": "c1"},
        outputs={"metadata": "metadata.json"},
        input_hashes={"contract-source": "a" * 64},
        output_hashes={"metadata.json": "b" * 64},
        versions={"serializer": "1.0"},
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        finished_at=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
        reason_code="serialized",
    )


def _run_record() -> RunRecord:
    return RunRecord(
        run_id="run-1",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        base_revision="base-revision",
        candidate_revision="candidate-revision",
        status=WorkflowStatus.VALIDATED_EVIDENCE,
        reason_code="evidence_complete",
        explanation="The controlled fixture produced attributable evidence.",
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        finished_at=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
    )


def _start_run(recorder: ArtifactRecorder, run_id: str = "run-1") -> RunHandle:
    ownership = RunOwnership.issue(run_id)
    return recorder.start_run(run_id, ownership)


def _events(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "run-1" / "events.jsonl").read_text().splitlines()
    ]
