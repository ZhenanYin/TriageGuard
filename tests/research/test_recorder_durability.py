from __future__ import annotations

import multiprocessing
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from triageguard.domain import EnvironmentKind, RunRecord, WorkflowStatus
from triageguard.research import (
    ArtifactRecorder,
    RecorderCorruptionError,
    RunHandle,
    RunOwnership,
    RunSealedError,
    UnsafeRecorderPathError,
)
from triageguard.research.recorder import (
    LifecycleEventType,
    TransformationEvent,
)


def _record_from_process(
    root: str,
    handle_payload: dict[str, object],
    index: int,
    ready: Any,
) -> None:
    recorder = ArtifactRecorder(root)
    handle = RunHandle.model_validate(handle_payload)
    ready.wait()
    recorder.record_event(
        handle,
        LifecycleEventType.CONTRACT_APPROVED,
        {"id": f"contract-{index}"},
    )


def _transformation(name: str = "contract.json") -> TransformationEvent:
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


def _record(*, explanation: str = "Evidence complete.") -> RunRecord:
    return RunRecord(
        run_id="run-1",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        base_revision="base-revision",
        candidate_revision="candidate-revision",
        status=WorkflowStatus.VALIDATION_FAILED,
        reason_code="unsafe_generated_code",
        explanation=explanation,
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        finished_at=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
    )


def _start(recorder: ArtifactRecorder) -> RunHandle:
    ownership = RunOwnership.issue("run-1")
    return recorder.start_run("run-1", ownership)


def test_every_recorder_mutation_requires_the_verified_run_handle(tmp_path: Path) -> None:
    """Knowing a run ID without its capability must never authorize a mutation."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start(recorder)
    foreign = RunHandle(run_id="run-1", ownership=RunOwnership.issue("run-1"))

    mutations = (
        lambda: recorder.record_event(
            foreign, LifecycleEventType.CONTRACT_APPROVED, {"id": "c1"}
        ),
        lambda: recorder.record_transformation(foreign, _transformation()),
        lambda: recorder.write_artifact(
            foreign, "contract.json", b"foreign", _transformation()
        ),
        lambda: recorder.finalize_run(foreign, _record()),
    )
    for mutate in mutations:
        with pytest.raises(ValueError, match="ownership"):
            mutate()

    with pytest.raises(TypeError, match="RunHandle"):
        recorder.record_event(  # type: ignore[arg-type]
            "run-1", LifecycleEventType.CONTRACT_APPROVED, {"id": "c1"}
        )
    assert recorder.verify_run_handle(handle) == tmp_path / "run-1"
    assert not (tmp_path / "run-1" / "events.jsonl").exists()


def test_terminal_finalization_seals_every_later_mutation(tmp_path: Path) -> None:
    """A terminal proof must make the entire run immutable, not only its record."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start(recorder)
    record = _record()

    first = recorder.finalize_run(handle, record)
    matching_retry = recorder.finalize_run(handle, record)

    assert matching_retry == first
    for mutate in (
        lambda: recorder.record_event(
            handle, LifecycleEventType.CONTRACT_APPROVED, {"id": "late"}
        ),
        lambda: recorder.record_transformation(handle, _transformation()),
        lambda: recorder.write_artifact(
            handle, "late.json", b"late", _transformation("late.json")
        ),
    ):
        with pytest.raises(RunSealedError, match="sealed"):
            mutate()
    with pytest.raises(RecorderCorruptionError, match="conflicting finalization"):
        recorder.finalize_run(handle, _record(explanation="Changed conclusion."))


def test_first_artifact_intent_binds_bytes_size_hash_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different retry must not replace an interrupted artifact intention."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start(recorder)
    real_atomic_write = recorder._atomic_write_file
    interrupted = False

    def interrupt_after_intent(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise OSError("interrupted after durable artifact intent")
        real_atomic_write(*args, **kwargs)

    monkeypatch.setattr(recorder, "_atomic_write_file", interrupt_after_intent)
    with pytest.raises(OSError, match="artifact intent"):
        recorder.write_artifact(
            handle, "contract.json", b"first", _transformation()
        )

    with pytest.raises(RecorderCorruptionError, match="conflicting artifact intent"):
        recorder.write_artifact(
            handle, "contract.json", b"second", _transformation()
        )

    artifact = recorder.write_artifact(
        handle, "contract.json", b"first", _transformation()
    )
    assert artifact.byte_count == 5
    assert recorder.read_artifact(handle, "contract.json") == b"first"


def test_first_finalization_intent_binds_exact_canonical_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted terminal intent cannot later certify another conclusion."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start(recorder)
    real_atomic_write = recorder._atomic_write_file
    interrupted = False

    def interrupt_final_record(
        directory_fd: int, name: str, content: bytes, *, mode: int = 0o600
    ) -> None:
        nonlocal interrupted
        if name == "run_record.json" and not interrupted:
            interrupted = True
            raise OSError("interrupted after durable finalization intent")
        real_atomic_write(directory_fd, name, content, mode=mode)

    monkeypatch.setattr(recorder, "_atomic_write_file", interrupt_final_record)
    with pytest.raises(OSError, match="finalization intent"):
        recorder.finalize_run(handle, _record())

    with pytest.raises(RecorderCorruptionError, match="conflicting finalization"):
        recorder.finalize_run(handle, _record(explanation="Changed conclusion."))

    assert recorder.finalize_run(handle, _record()).name == "run_record.json"


def test_symlinked_event_log_and_artifact_component_cannot_escape_root(
    tmp_path: Path,
) -> None:
    """Recorder opens must reject links rather than follow them outside the run."""
    recorder = ArtifactRecorder(tmp_path / "records")
    handle = _start(recorder)
    run_directory = tmp_path / "records" / "run-1"
    outside_event = tmp_path / "outside-events.jsonl"
    outside_event.write_bytes(b"outside\n")
    (run_directory / "events.jsonl").symlink_to(outside_event)

    with pytest.raises(UnsafeRecorderPathError):
        recorder.record_event(
            handle, LifecycleEventType.CONTRACT_APPROVED, {"id": "c1"}
        )
    assert outside_event.read_bytes() == b"outside\n"

    (run_directory / "events.jsonl").unlink()
    outside_directory = tmp_path / "outside-artifacts"
    outside_directory.mkdir()
    (run_directory / "artifacts").symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(UnsafeRecorderPathError):
        recorder.write_artifact(
            handle,
            "artifacts/contract.json",
            b"inside-only",
            _transformation("artifacts/contract.json"),
        )
    assert list(outside_directory.iterdir()) == []


def test_symlinked_run_directory_is_rejected_without_outside_mutation(
    tmp_path: Path,
) -> None:
    """A run-directory link cannot turn a locally scoped handle into outside access."""
    root = tmp_path / "records"
    root.mkdir()
    outside = tmp_path / "outside-run"
    outside.mkdir()
    (root / "run-1").symlink_to(outside, target_is_directory=True)
    recorder = ArtifactRecorder(root)
    handle = RunHandle(run_id="run-1", ownership=RunOwnership.issue("run-1"))

    with pytest.raises(UnsafeRecorderPathError):
        recorder.verify_run_handle(handle)
    assert list(outside.iterdir()) == []


def test_atomic_create_never_replaces_a_destination_created_at_commit_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A racing destination must win; recorder bytes must never replace it."""
    recorder = ArtifactRecorder(tmp_path)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_rename = os.rename
    real_link = os.link
    injected = False

    def create_racer() -> None:
        nonlocal injected
        if injected:
            return
        injected = True
        descriptor = os.open(
            "artifact.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, b"racing writer")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def racing_rename(*args: Any, **kwargs: Any) -> None:
        create_racer()
        real_rename(*args, **kwargs)

    def racing_link(*args: Any, **kwargs: Any) -> None:
        create_racer()
        real_link(*args, **kwargs)

    monkeypatch.setattr(os, "rename", racing_rename)
    monkeypatch.setattr(os, "link", racing_link)
    try:
        with pytest.raises(FileExistsError):
            recorder._atomic_write_file(
                directory_fd, "artifact.json", b"recorder bytes"
            )
    finally:
        os.close(directory_fd)

    assert (tmp_path / "artifact.json").read_bytes() == b"racing writer"
    assert not any(path.name.startswith(".tmp-") for path in tmp_path.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="fcntl durability is POSIX-only")
def test_cross_process_mutations_share_one_run_lock_and_sequence(tmp_path: Path) -> None:
    """Independent recorder processes must serialize the whole mutation boundary."""
    recorder = ArtifactRecorder(tmp_path)
    handle = _start(recorder)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    processes = [
        context.Process(
            target=_record_from_process,
            args=(str(tmp_path), handle.model_dump(mode="json"), index, barrier),
        )
        for index in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    events = recorder.read_events(handle)
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert {event.payload["id"] for event in events} == {
        "contract-0",
        "contract-1",
        "contract-2",
        "contract-3",
    }
