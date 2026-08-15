"""Authenticated, append-only local storage for attributable research evidence."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock, RLock
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from triageguard.domain import MilestoneTwoRunRecord, RunRecord
from triageguard.provenance import canonical_json

TerminalRecord = RunRecord | MilestoneTwoRunRecord


class RecorderCorruptionError(RuntimeError):
    """Durable recorder state contradicts an already-bound intent."""


class RunSealedError(RuntimeError):
    """A mutation was attempted after terminal or pending finalization."""


class UnsafeRecorderPathError(ValueError):
    """A recorder-owned path is a symlink or a non-regular object."""


class RunOwnership(BaseModel):
    """Frozen caller identity proving ownership of one recorder run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    marker_type: StrictStr = "triageguard_run_ownership"
    schema_version: StrictInt = 1
    run_id: StrictStr = Field(min_length=1)
    ownership_token: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def metadata_is_current(self) -> RunOwnership:
        if self.marker_type != "triageguard_run_ownership":
            raise ValueError("marker_type must identify TriageGuard run ownership")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _validate_leaf_name(self.run_id, label="run_id")
        return self

    @classmethod
    def issue(cls, run_id: str) -> RunOwnership:
        """Create a fresh unpredictable identity before recorder I/O begins."""
        return cls(run_id=run_id, ownership_token=secrets.token_hex(32))

    def canonical_bytes(self) -> bytes:
        """Return the one accepted on-disk representation of this identity."""
        return (
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")


class RunHandle(BaseModel):
    """Unforgeable-in-practice capability required for every run mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: StrictStr = Field(min_length=1)
    ownership: RunOwnership

    @model_validator(mode="after")
    def ownership_matches_run(self) -> RunHandle:
        if self.ownership.run_id != self.run_id:
            raise ValueError("ownership.run_id must match RunHandle.run_id")
        return self


class LifecycleEventType(str, Enum):
    """Non-transformation events accepted by the lifecycle-only API."""

    RUN_STARTED = "run_started"
    CONTRACT_APPROVED = "contract_approved"
    RISK_APPROVED = "risk_approved"
    GHERKIN_APPROVED = "gherkin_approved"
    FINALIZATION_STARTED = "finalization_started"
    FINALIZATION_COMPLETED = "finalization_completed"


class LifecycleEvent(BaseModel):
    """Validated payload for an event that does not transform research data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: LifecycleEventType
    payload: dict[StrictStr, StrictStr]

    @model_validator(mode="after")
    def payload_matches_event_type(self) -> LifecycleEvent:
        expected_keys = {
            LifecycleEventType.RUN_STARTED: {"id", "ownership_token"},
            LifecycleEventType.CONTRACT_APPROVED: {"id"},
            LifecycleEventType.RISK_APPROVED: {"id", "risk_sha256"},
            LifecycleEventType.GHERKIN_APPROVED: {"id", "gherkin_sha256"},
            LifecycleEventType.FINALIZATION_STARTED: {"record_sha256"},
            LifecycleEventType.FINALIZATION_COMPLETED: {"record_sha256"},
        }[self.event_type]
        if set(self.payload) != expected_keys:
            raise ValueError(
                f"{self.event_type.value} requires payload keys: {sorted(expected_keys)}"
            )

        digest_key = {
            LifecycleEventType.RISK_APPROVED: "risk_sha256",
            LifecycleEventType.GHERKIN_APPROVED: "gherkin_sha256",
            LifecycleEventType.FINALIZATION_STARTED: "record_sha256",
            LifecycleEventType.FINALIZATION_COMPLETED: "record_sha256",
        }.get(self.event_type)
        if digest_key is not None and not _is_sha256(self.payload[digest_key]):
            raise ValueError(f"{digest_key} must be a lowercase SHA-256 digest")
        return self


class TransformationEvent(BaseModel):
    """Complete, attributable provenance for a deterministic transformation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: str = Field(min_length=1)
    inputs: dict[str, str] = Field(min_length=1)
    outputs: dict[str, str] = Field(min_length=1)
    input_hashes: dict[str, str] = Field(min_length=1)
    output_hashes: dict[str, str] = Field(min_length=1)
    versions: dict[str, str] = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    reason_code: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> TransformationEvent:
        if self.started_at.utcoffset() != UTC.utcoffset(self.started_at) or (
            self.finished_at.utcoffset() != UTC.utcoffset(self.finished_at)
        ):
            raise ValueError("transformation timestamps must be UTC")
        if self.finished_at <= self.started_at:
            raise ValueError("finished_at must be later than started_at")
        for digest in [*self.input_hashes.values(), *self.output_hashes.values()]:
            if not _is_sha256(digest):
                raise ValueError(
                    "transformation hashes must be lowercase SHA-256 digests"
                )
        return self


class ArtifactWriteJournal(BaseModel):
    """Durable intent and completion details for one immutable artifact write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_name: str = Field(min_length=1)
    artifact_sha256: str
    artifact_byte_count: int = Field(ge=0)
    provenance: TransformationEvent

    @model_validator(mode="after")
    def provenance_matches_artifact(self) -> ArtifactWriteJournal:
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        if self.provenance.outputs.get(self.artifact_name) != self.artifact_name:
            raise ValueError(
                "artifact provenance must identify its normalized artifact name"
            )
        if (
            self.provenance.output_hashes.get(self.artifact_name)
            != self.artifact_sha256
        ):
            raise ValueError("artifact provenance must contain its computed SHA-256")
        return self


@dataclass(frozen=True)
class RecordedEvent:
    """Metadata for an event durably appended to a run's event log."""

    sequence: int
    timestamp: str
    event_type: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactMetadata:
    """Identifies immutable bytes written into a run directory."""

    name: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class _OpenRun:
    root_fd: int
    run_fd: int
    run_directory: Path


class ArtifactRecorder:
    """Persist authenticated events and artifacts without permitting replacement."""

    _OWNERSHIP_MARKER_NAME = ".run-ownership.json"
    _RUN_LOCK_NAME = ".run.lock"
    _WORKFLOW_LOCK_NAME = ".workflow.lock"
    _EVENTS_NAME = "events.jsonl"
    _FINAL_RECORD_NAME = "run_record.json"
    _RESERVED_TOP_LEVEL = frozenset(
        {
            _OWNERSHIP_MARKER_NAME,
            _RUN_LOCK_NAME,
            _WORKFLOW_LOCK_NAME,
            _EVENTS_NAME,
            _FINAL_RECORD_NAME,
            ".artifact-locks",
            ".finalization.lock",
        }
    )
    _process_run_locks: ClassVar[dict[Path, RLock]] = {}
    _process_run_locks_guard = Lock()
    _process_workflow_locks: ClassVar[dict[Path, RLock]] = {}
    _process_workflow_locks_guard = Lock()

    def __init__(self, root_directory: str | Path) -> None:
        self._root_directory = Path(os.path.abspath(os.fspath(root_directory)))

    def start_run(self, run_id: str, ownership: RunOwnership) -> RunHandle:
        """Create a run and return the only capability accepted for mutations."""
        self._validate_run_ownership(run_id, ownership)
        root_fd = self._open_root(create=True)
        run_fd: int | None = None
        try:
            os.mkdir(run_id, mode=0o700, dir_fd=root_fd)
            _fsync_fd(root_fd)
            run_fd = _open_directory_at(root_fd, run_id)
            self._atomic_write_file(run_fd, self._RUN_LOCK_NAME, b"")
            self._atomic_write_file(run_fd, self._WORKFLOW_LOCK_NAME, b"")
            self._atomic_write_file(
                run_fd,
                self._OWNERSHIP_MARKER_NAME,
                ownership.canonical_bytes(),
            )
            _fsync_fd(run_fd)
        finally:
            if run_fd is not None:
                os.close(run_fd)
            os.close(root_fd)
        return RunHandle(run_id=run_id, ownership=ownership)

    def resume_run(self, run_id: str, ownership: RunOwnership) -> RunHandle:
        """Authenticate and return a capability for an existing locally owned run."""
        self._validate_run_ownership(run_id, ownership)
        handle = RunHandle(run_id=run_id, ownership=ownership)
        self.verify_run_handle(handle)
        return handle

    def locate_run(self, run_id: str) -> Path:
        """Locate a run directory without granting mutation authority."""
        _validate_leaf_name(run_id, label="run_id")
        root_fd = self._open_root(create=False)
        try:
            run_fd = _open_directory_at(root_fd, run_id)
            os.close(run_fd)
        finally:
            os.close(root_fd)
        return self._root_directory / run_id

    def verify_run_handle(self, handle: RunHandle) -> Path:
        """Verify the exact durable ownership proof without changing the run."""
        with self._locked_run(handle, require_mutable=False) as opened:
            return opened.run_directory

    def verify_run_ownership(self, run_id: str, expected: RunOwnership) -> Path:
        """Compatibility read boundary; mutations still require a RunHandle."""
        return self.verify_run_handle(RunHandle(run_id=run_id, ownership=expected))

    @contextmanager
    def workflow_lease(self, handle: RunHandle) -> Iterator[None]:
        """Hold one authenticated cross-process lease across a workflow operation."""
        self._require_handle(handle)
        lock_path = self._root_directory / handle.run_id / self._WORKFLOW_LOCK_NAME
        with self._process_lock(
            lock_path,
            self._process_workflow_locks,
            self._process_workflow_locks_guard,
        ):
            opened = self._open_authenticated_run(handle)
            lock_fd: int | None = None
            try:
                lock_fd = _open_regular_at(
                    opened.run_fd,
                    self._WORKFLOW_LOCK_NAME,
                    os.O_RDWR,
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                self._verify_marker(opened.run_fd, handle)
                yield
            finally:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                os.close(opened.run_fd)
                os.close(opened.root_fd)

    def record_event(
        self,
        handle: RunHandle,
        event_type: LifecycleEventType | str,
        payload: Mapping[str, str],
    ) -> RecordedEvent:
        """Record a validated lifecycle event through an authenticated handle."""
        return self.record_lifecycle_event(
            handle, LifecycleEvent(event_type=event_type, payload=dict(payload))
        )

    def record_lifecycle_event(
        self, handle: RunHandle, event: LifecycleEvent
    ) -> RecordedEvent:
        """Append a typed event about run state, not a data transformation."""
        if not isinstance(event, LifecycleEvent):
            raise TypeError("event must be a LifecycleEvent")
        if event.event_type in {
            LifecycleEventType.FINALIZATION_STARTED,
            LifecycleEventType.FINALIZATION_COMPLETED,
        }:
            raise ValueError("finalization lifecycle events are recorder-internal")
        self._require_handle(handle)
        if event.event_type is LifecycleEventType.RUN_STARTED:
            if event.payload["id"] != handle.run_id:
                raise ValueError("run_started payload id must match RunHandle.run_id")
            if event.payload["ownership_token"] != handle.ownership.ownership_token:
                raise ValueError(
                    "run_started payload ownership does not match RunHandle"
                )
        with self._locked_run(handle, require_mutable=True) as opened:
            return self._append_event_locked(
                opened.run_fd,
                event.event_type.value,
                event.payload,
            )

    def record_transformation(
        self, handle: RunHandle, event: TransformationEvent
    ) -> RecordedEvent:
        """Append complete typed provenance through an authenticated handle."""
        if not isinstance(event, TransformationEvent):
            raise TypeError("event must be a TransformationEvent")
        with self._locked_run(handle, require_mutable=True) as opened:
            return self._append_event_locked(
                opened.run_fd,
                event.event_type,
                event.model_dump(mode="json"),
            )

    def write_artifact(
        self,
        handle: RunHandle,
        name: str,
        content: bytes,
        provenance: TransformationEvent,
    ) -> ArtifactMetadata:
        """Write immutable bytes, binding the first durable intent exactly."""
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        if not isinstance(provenance, TransformationEvent):
            raise TypeError("provenance must be a TransformationEvent")
        parts = self._artifact_parts(name)
        artifact_name = "/".join(parts)
        artifact = ArtifactMetadata(
            name=artifact_name,
            sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
        )
        journal = ArtifactWriteJournal(
            artifact_name=artifact.name,
            artifact_sha256=artifact.sha256,
            artifact_byte_count=artifact.byte_count,
            provenance=self._provenance_for_artifact(provenance, artifact),
        )
        expected_payload = journal.model_dump(mode="json")

        with self._locked_run(handle, require_mutable=True) as opened:
            events = self._read_event_dicts_locked(opened.run_fd)
            started_payloads = [
                item["payload"]
                for item in events
                if item["event_type"] == "artifact_write_started"
                and item["payload"].get("artifact_name") == artifact_name
            ]
            completed_payloads = [
                item["payload"]
                for item in events
                if item["event_type"] == "artifact_write_completed"
                and item["payload"].get("artifact_name") == artifact_name
            ]
            if any(payload != expected_payload for payload in started_payloads):
                raise RecorderCorruptionError(
                    f"conflicting artifact intent for {artifact_name}"
                )
            if any(payload != expected_payload for payload in completed_payloads):
                raise RecorderCorruptionError(
                    f"conflicting artifact completion for {artifact_name}"
                )
            if len(started_payloads) > 1 or len(completed_payloads) > 1:
                raise RecorderCorruptionError(
                    f"duplicate artifact journal entries for {artifact_name}"
                )
            if completed_payloads and not started_payloads:
                raise RecorderCorruptionError(
                    f"artifact completion lacks intent for {artifact_name}"
                )
            if not started_payloads:
                self._append_event_locked(
                    opened.run_fd,
                    "artifact_write_started",
                    expected_payload,
                )

            parent_fd = self._open_artifact_parent(
                opened.run_fd, parts[:-1], create=True
            )
            try:
                existing = _read_regular_optional(parent_fd, parts[-1])
                if existing is None:
                    self._atomic_write_file(parent_fd, parts[-1], content)
                elif existing != content:
                    raise RecorderCorruptionError(
                        f"existing artifact contradicts intent for {artifact_name}"
                    )
            finally:
                os.close(parent_fd)

            if not completed_payloads:
                self._append_event_locked(
                    opened.run_fd,
                    "artifact_write_completed",
                    expected_payload,
                )
            return artifact

    def read_artifact(self, handle: RunHandle, name: str) -> bytes:
        """Read one authenticated immutable artifact without following symlinks."""
        parts = self._artifact_parts(name, allow_final_record=True)
        with self._locked_run(handle, require_mutable=False) as opened:
            parent_fd = self._open_artifact_parent(
                opened.run_fd, parts[:-1], create=False
            )
            try:
                return _read_regular_at(parent_fd, parts[-1])
            finally:
                os.close(parent_fd)

    def read_events(self, handle: RunHandle) -> list[RecordedEvent]:
        """Read and validate the complete authenticated event sequence."""
        with self._locked_run(handle, require_mutable=False) as opened:
            return [
                RecordedEvent(
                    sequence=item["sequence"],
                    timestamp=item["timestamp"],
                    event_type=item["event_type"],
                    payload=item["payload"],
                )
                for item in self._read_event_dicts_locked(opened.run_fd)
            ]

    def finalize_run(
        self, handle: RunHandle, record: TerminalRecord
    ) -> ArtifactMetadata:
        """Bind and commit one exact canonical terminal record idempotently."""
        self._require_handle(handle)
        if not isinstance(record, (RunRecord, MilestoneTwoRunRecord)):
            raise TypeError("record must be a supported terminal record")
        if record.run_id != handle.run_id:
            raise ValueError("record.run_id must match RunHandle.run_id")
        serialized_record = (
            canonical_json(record.model_dump(mode="json")) + "\n"
        ).encode("utf-8")
        record_sha256 = hashlib.sha256(serialized_record).hexdigest()
        payload = {"record_sha256": record_sha256}

        with self._locked_run(handle, require_mutable=False) as opened:
            events = self._read_event_dicts_locked(opened.run_fd)
            completed = self._finalization_payloads(
                events, LifecycleEventType.FINALIZATION_COMPLETED
            )
            if completed:
                terminal_digest, terminal_bytes = self._verified_terminal(
                    opened.run_fd, events
                )
                if (
                    terminal_digest != record_sha256
                    or terminal_bytes != serialized_record
                ):
                    raise RecorderCorruptionError(
                        "conflicting finalization attempted after run was sealed"
                    )
                return ArtifactMetadata(
                    name=self._FINAL_RECORD_NAME,
                    sha256=record_sha256,
                    byte_count=len(serialized_record),
                )

            started = self._finalization_payloads(
                events, LifecycleEventType.FINALIZATION_STARTED
            )
            if any(item != payload for item in started) or len(started) > 1:
                raise RecorderCorruptionError("conflicting finalization intent")
            if not started:
                self._append_event_locked(
                    opened.run_fd,
                    LifecycleEventType.FINALIZATION_STARTED.value,
                    payload,
                )

            existing = _read_regular_optional(opened.run_fd, self._FINAL_RECORD_NAME)
            if existing is None:
                self._atomic_write_file(
                    opened.run_fd,
                    self._FINAL_RECORD_NAME,
                    serialized_record,
                )
            elif existing != serialized_record:
                raise RecorderCorruptionError("conflicting finalization record bytes")

            self._append_event_locked(
                opened.run_fd,
                LifecycleEventType.FINALIZATION_COMPLETED.value,
                payload,
            )
            return ArtifactMetadata(
                name=self._FINAL_RECORD_NAME,
                sha256=record_sha256,
                byte_count=len(serialized_record),
            )

    @contextmanager
    def _locked_run(
        self, handle: RunHandle, *, require_mutable: bool
    ) -> Iterator[_OpenRun]:
        self._require_handle(handle)
        lock_path = self._root_directory / handle.run_id / self._RUN_LOCK_NAME
        with self._process_lock(
            lock_path,
            self._process_run_locks,
            self._process_run_locks_guard,
        ):
            opened = self._open_authenticated_run(handle)
            lock_fd: int | None = None
            try:
                lock_fd = _open_regular_at(
                    opened.run_fd,
                    self._RUN_LOCK_NAME,
                    os.O_RDWR,
                )
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                self._verify_marker(opened.run_fd, handle)
                if require_mutable:
                    self._assert_mutable(opened.run_fd)
                yield opened
            finally:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                os.close(opened.run_fd)
                os.close(opened.root_fd)

    @staticmethod
    @contextmanager
    def _process_lock(
        path: Path,
        registry: dict[Path, RLock],
        guard: Lock,
    ) -> Iterator[None]:
        with guard:
            lock = registry.setdefault(path, RLock())
        with lock:
            yield

    def _open_authenticated_run(self, handle: RunHandle) -> _OpenRun:
        root_fd = self._open_root(create=False)
        try:
            run_fd = _open_directory_at(root_fd, handle.run_id)
        except BaseException:
            os.close(root_fd)
            raise
        try:
            self._verify_marker(run_fd, handle)
        except BaseException:
            os.close(run_fd)
            os.close(root_fd)
            raise
        return _OpenRun(
            root_fd=root_fd,
            run_fd=run_fd,
            run_directory=self._root_directory / handle.run_id,
        )

    def _verify_marker(self, run_fd: int, handle: RunHandle) -> None:
        try:
            marker_bytes = _read_regular_at(run_fd, self._OWNERSHIP_MARKER_NAME)
            actual = RunOwnership.model_validate_json(marker_bytes)
        except UnsafeRecorderPathError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError("run ownership marker is invalid") from error
        if marker_bytes != actual.canonical_bytes():
            raise ValueError("run ownership marker is not canonical")
        if actual != handle.ownership:
            raise ValueError("run ownership marker does not match handle ownership")

    def _assert_mutable(self, run_fd: int) -> None:
        events = self._read_event_dicts_locked(run_fd)
        completed = self._finalization_payloads(
            events, LifecycleEventType.FINALIZATION_COMPLETED
        )
        if completed:
            self._verified_terminal(run_fd, events)
            raise RunSealedError("run is sealed by terminal finalization")
        if self._finalization_payloads(events, LifecycleEventType.FINALIZATION_STARTED):
            raise RunSealedError("run is sealed while finalization is pending")

    def _verified_terminal(
        self,
        run_fd: int,
        events: list[dict[str, object]],
    ) -> tuple[str, bytes]:
        started = self._finalization_payloads(
            events, LifecycleEventType.FINALIZATION_STARTED
        )
        completed = self._finalization_payloads(
            events, LifecycleEventType.FINALIZATION_COMPLETED
        )
        if len(started) != 1 or len(completed) != 1 or started[0] != completed[0]:
            raise RecorderCorruptionError(
                "terminal finalization journal is inconsistent"
            )
        digest = completed[0].get("record_sha256")
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise RecorderCorruptionError("terminal finalization digest is invalid")
        try:
            record_bytes = _read_regular_at(run_fd, self._FINAL_RECORD_NAME)
        except (FileNotFoundError, UnsafeRecorderPathError) as error:
            raise RecorderCorruptionError(
                "terminal final record is unavailable"
            ) from error
        if hashlib.sha256(record_bytes).hexdigest() != digest:
            raise RecorderCorruptionError(
                "terminal final record does not match its digest"
            )
        return digest, record_bytes

    @staticmethod
    def _finalization_payloads(
        events: list[dict[str, object]], event_type: LifecycleEventType
    ) -> list[dict[str, object]]:
        return [
            item["payload"] for item in events if item["event_type"] == event_type.value
        ]

    def _append_event_locked(
        self,
        run_fd: int,
        event_type: str,
        payload: Mapping[str, object],
    ) -> RecordedEvent:
        if not event_type:
            raise ValueError("event_type must not be empty")
        existing = self._read_event_dicts_locked(run_fd)
        sequence = len(existing) + 1
        timestamp = datetime.now(UTC).isoformat()
        event_payload = dict(payload)
        line = (
            json.dumps(
                {
                    "sequence": sequence,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "payload": event_payload,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        created = False
        try:
            descriptor = _open_regular_at(
                run_fd,
                self._EVENTS_NAME,
                os.O_WRONLY | os.O_APPEND,
            )
        except FileNotFoundError:
            descriptor = _open_regular_at(
                run_fd,
                self._EVENTS_NAME,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                mode=0o600,
            )
            created = True
        try:
            _write_all(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            _fsync_fd(run_fd)
        return RecordedEvent(sequence, timestamp, event_type, event_payload)

    def _read_event_dicts_locked(self, run_fd: int) -> list[dict[str, object]]:
        serialized = _read_regular_optional(run_fd, self._EVENTS_NAME)
        if serialized is None:
            return []
        if serialized and not serialized.endswith(b"\n"):
            raise RecorderCorruptionError("event log has an incomplete trailing record")
        events: list[dict[str, object]] = []
        for expected_sequence, line in enumerate(serialized.splitlines(), start=1):
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise RecorderCorruptionError(
                    "event log contains invalid JSON"
                ) from error
            if not isinstance(item, dict) or set(item) != {
                "sequence",
                "timestamp",
                "event_type",
                "payload",
            }:
                raise RecorderCorruptionError("event log record shape is invalid")
            if item["sequence"] != expected_sequence:
                raise RecorderCorruptionError("event log sequence is not contiguous")
            if not isinstance(item["timestamp"], str) or not isinstance(
                item["event_type"], str
            ):
                raise RecorderCorruptionError("event log metadata is invalid")
            if not isinstance(item["payload"], dict):
                raise RecorderCorruptionError("event log payload is invalid")
            events.append(item)
        return events

    def _open_artifact_parent(
        self,
        run_fd: int,
        directory_parts: tuple[str, ...],
        *,
        create: bool,
    ) -> int:
        current = os.dup(run_fd)
        try:
            for part in directory_parts:
                try:
                    child = _open_directory_at(current, part)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(part, mode=0o700, dir_fd=current)
                    _fsync_fd(current)
                    child = _open_directory_at(current, part)
                os.close(current)
                current = child
            return current
        except BaseException:
            os.close(current)
            raise

    @staticmethod
    def _provenance_for_artifact(
        provenance: TransformationEvent, artifact: ArtifactMetadata
    ) -> TransformationEvent:
        expected_hash = provenance.output_hashes.get(artifact.name)
        if expected_hash is not None and expected_hash != artifact.sha256:
            raise ValueError(
                "artifact content does not match the declared output SHA-256"
            )
        output = provenance.outputs.get(artifact.name)
        if output is not None and output != artifact.name:
            raise ValueError(
                "artifact output must identify its normalized artifact name"
            )
        return TransformationEvent.model_validate(
            {
                **provenance.model_dump(),
                "outputs": {**provenance.outputs, artifact.name: artifact.name},
                "output_hashes": {
                    **provenance.output_hashes,
                    artifact.name: artifact.sha256,
                },
            }
        )

    @staticmethod
    def _atomic_write_file(
        directory_fd: int,
        name: str,
        content: bytes,
        *,
        mode: int = 0o600,
    ) -> None:
        """Create and fsync one regular file without ever replacing a destination."""
        _validate_leaf_name(name, label="file name")
        _reject_existing_destination(directory_fd, name)
        temporary = f".tmp-{secrets.token_hex(16)}"
        descriptor: int | None = None
        temporary_exists = False
        try:
            descriptor = _open_regular_at(
                directory_fd,
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode=mode,
            )
            temporary_exists = True
            _write_all(descriptor, content)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                # A same-directory hard link is an atomic create-if-absent
                # operation. Unlike rename(2), it cannot replace a destination
                # installed after the earlier diagnostic check.
                os.link(
                    temporary,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                # Preserve the stronger unsafe-path diagnosis for a racing link
                # or non-regular object while still failing closed for a file.
                _reject_existing_destination(directory_fd, name)
                raise
            os.unlink(temporary, dir_fd=directory_fd)
            temporary_exists = False
            _fsync_fd(directory_fd)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_exists:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass

    def _artifact_parts(
        self, name: str, *, allow_final_record: bool = False
    ) -> tuple[str, ...]:
        if not isinstance(name, str) or not name:
            raise ValueError("artifact name must not be empty")
        candidate = Path(name)
        parts = candidate.parts
        if (
            candidate.is_absolute()
            or not parts
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("artifact path must be a normalized relative file path")
        if (
            len(parts) == 1
            and parts[0] == self._FINAL_RECORD_NAME
            and allow_final_record
        ):
            return parts
        if parts[0] in self._RESERVED_TOP_LEVEL:
            raise ValueError("artifact name is reserved for recorder-owned state")
        for part in parts:
            _validate_leaf_name(part, label="artifact path component")
        return parts

    def _open_root(self, *, create: bool) -> int:
        return _open_absolute_directory(self._root_directory, create=create)

    @staticmethod
    def _validate_run_ownership(run_id: str, ownership: RunOwnership) -> None:
        if not isinstance(ownership, RunOwnership):
            raise TypeError("ownership must be a RunOwnership")
        _validate_leaf_name(run_id, label="run_id")
        if ownership.run_id != run_id:
            raise ValueError("ownership.run_id must match run_id")

    @staticmethod
    def _require_handle(handle: RunHandle) -> None:
        if not isinstance(handle, RunHandle):
            raise TypeError("every recorder mutation requires a verified RunHandle")


def _open_absolute_directory(path: Path, *, create: bool) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = os.open(
        os.sep,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for part in absolute.parts[1:]:
            try:
                child = _open_directory_at(current, part)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=current)
                _fsync_fd(current)
                child = _open_directory_at(current, part)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_directory_at(parent_fd: int, name: str) -> int:
    _validate_leaf_name(name, label="directory name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafeRecorderPathError(
                f"recorder directory is symlinked or not a directory: {name}"
            ) from error
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise UnsafeRecorderPathError(f"recorder path is not a directory: {name}")
    return descriptor


def _open_regular_at(
    parent_fd: int,
    name: str,
    flags: int,
    *,
    mode: int = 0o600,
) -> int:
    _validate_leaf_name(name, label="file name")
    safe_flags = flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, safe_flags, mode, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafeRecorderPathError(
                f"recorder file is symlinked or has an unsafe parent: {name}"
            ) from error
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise UnsafeRecorderPathError(f"recorder path is not a regular file: {name}")
    return descriptor


def _read_regular_at(parent_fd: int, name: str) -> bytes:
    descriptor = _open_regular_at(parent_fd, name, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_regular_optional(parent_fd: int, name: str) -> bytes | None:
    try:
        return _read_regular_at(parent_fd, name)
    except FileNotFoundError:
        return None


def _reject_existing_destination(parent_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise UnsafeRecorderPathError(f"recorder destination is a symlink: {name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafeRecorderPathError(
            f"recorder destination is not a regular file: {name}"
        )
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), name)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short recorder write")
        view = view[written:]


def _fsync_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _validate_leaf_name(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be one normalized path component")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
