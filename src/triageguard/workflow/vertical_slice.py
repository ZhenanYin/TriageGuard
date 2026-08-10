"""One attributable replay workflow from approved contract to evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from triageguard.config import MAX_REPEAT_COUNT, MIN_REPEAT_COUNT, Settings
from triageguard.contracts import render_gherkin, validate_gherkin_alignment
from triageguard.domain import (
    CvssProfile,
    DifferentialEvidence,
    EnvironmentKind,
    ExecutionFile,
    ExecutionManifest,
    RiskContract,
    RunRecord,
    RuntimeObservation,
    TestPlan,
    WorkflowStatus,
)
from triageguard.evidence import classify_differential
from triageguard.execution import (
    ControlledAuthorizationServer,
    ExecutionArtifacts,
    ExecutionTarget,
    ExecutionTimeoutError,
    InvalidObservationError,
    MissingObservationError,
    PytestRunner,
    UnexpectedPytestOutcomeError,
)
from triageguard.generation import (
    CodeValidationReport,
    GeneratedCodeArtifact,
    PlanValidationError,
    create_test_plan,
    generate_pytest,
    validate_generated_code,
)
from triageguard.llm import (
    ModelGatewayError,
    ModelOutputInvalid,
    ModelRequest,
    ModelResponse,
    ReplayGateway,
    ReplayResponseMissing,
    StructuredModelGateway,
)
from triageguard.provenance import canonical_json, canonical_sha256
from triageguard.research import ArtifactRecorder, RunHandle, RunOwnership
from triageguard.research.recorder import (
    ArtifactWriteJournal,
    LifecycleEvent,
    LifecycleEventType,
    TransformationEvent,
)
from triageguard.runtime import RuntimeObservationEnvelope
from triageguard.severity import (
    CvssAssessmentError,
    assess_differential_severity,
    calculate_cvss4,
)

_CONTROLLED_FIXTURE_WARNING = (
    "Controlled fixture only; this run is not evidence about a real OpenMRS revision."
)
_ADMINISTRATOR = "fixture-administrator"
_CLERK = "clerk"
_PASSWORD = "fixture-password-not-for-logs"
_BASE_REVISION = "base-revision"
_CANDIDATE_REVISION = "candidate-revision"


def _read_fixture_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    """Return the fields that bind a directory entry to one object."""
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _file_snapshot_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    """Return mutation-sensitive file metadata, excluding read-updated atime."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_beneath_directory(
    root_directory: Path,
    source_path: Path,
) -> bytes:
    """Read one regular file through one descriptor-rooted no-follow chain."""
    root = Path(os.path.abspath(os.fspath(root_directory)))
    source = Path(os.path.abspath(os.fspath(source_path)))
    relative = source.relative_to(root)
    if not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("source must be a normalized file beneath its run directory")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fds: list[int] = []
    directory_bindings: list[tuple[int, str, int, os.stat_result]] = []
    file_fd: int | None = None
    try:
        current = os.open(
            os.sep,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        directory_fds.append(current)

        for component in (*root.parts[1:], *relative.parts[:-1]):
            child = os.open(component, directory_flags, dir_fd=current)
            try:
                child_metadata = os.fstat(child)
                if not stat.S_ISDIR(child_metadata.st_mode):
                    raise ValueError("source parent is not a directory")
            except BaseException:
                os.close(child)
                raise
            directory_fds.append(child)
            directory_bindings.append(
                (current, component, child, child_metadata)
            )
            current = child

        file_name = relative.parts[-1]
        file_fd = os.open(file_name, file_flags, dir_fd=current)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source is not a regular file")

        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)

        after = os.fstat(file_fd)
        entry = os.stat(file_name, dir_fd=current, follow_symlinks=False)
        if (
            _file_snapshot_identity(before) != _file_snapshot_identity(after)
            or _file_snapshot_identity(after) != _file_snapshot_identity(entry)
            or len(content) != after.st_size
        ):
            raise ValueError("source changed while it was being read")

        for parent_fd, name, child_fd, opened in directory_bindings:
            opened_fd = os.fstat(child_fd)
            current_entry = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(opened_fd.st_mode)
                or not stat.S_ISDIR(current_entry.st_mode)
                or _stat_identity(opened) != _stat_identity(opened_fd)
                or _stat_identity(opened_fd) != _stat_identity(current_entry)
            ):
                raise ValueError("source directory changed while it was being read")
        return content
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


class WorkflowTransitionError(RuntimeError):
    """A method was called outside its one legal workflow state."""


class ContractApprovalError(ValueError):
    """The proposed approval did not preserve the prepared contract and Gherkin."""


class UnsafeGeneratedCodeError(ValueError):
    """Generated source did not pass the deterministic full-code gate."""

    def __init__(self, report: CodeValidationReport) -> None:
        self.report = report
        super().__init__("generated code failed deterministic validation")


class InterruptedExternalOperationError(RuntimeError):
    """A durable intent has no durable result, so its outcome is unknowable."""

    def __init__(self, operation_kind: str, operation_id: str) -> None:
        self.operation_kind = operation_kind
        self.operation_id = operation_id
        super().__init__(
            f"{operation_kind} operation {operation_id} has an interrupted unknown outcome"
        )


class OperationJournalInterruptedError(OSError):
    """An external result is known in memory but its durable commit was interrupted."""


class InvalidClassifierResultError(TypeError):
    """The injected classifier did not return coherent typed evidence."""

    def __init__(self) -> None:
        super().__init__("classifier must return a coherent DifferentialEvidence")


class ExecutionManifestError(RuntimeError):
    """A completed experiment lacks an exact immutable file inventory."""


class _WorkflowState(str, Enum):
    NEW = "new"
    PREPARED = "prepared"
    APPROVED = "approved"
    GENERATED = "generated"
    VALIDATED = "validated"
    EXECUTED = "executed"
    FINALIZED = "finalized"


class ControlledImpactReport(BaseModel):
    """Typed description of the fixture evidence entering Milestone 1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str
    classification: str
    purpose: str
    affected_workflow: str
    publication_status: str
    evidence_scope: str


class PreparedWorkflow(BaseModel):
    """Immutable, inspectable inputs proposed for human approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    environment_kind: EnvironmentKind
    base_revision: str
    candidate_revision: str
    environment_warning: str
    impact_report: ControlledImpactReport
    cvss_profile: CvssProfile
    contract: RiskContract
    gherkin: str
    contract_sha256: str
    cvss_profile_sha256: str
    gherkin_sha256: str


class GeneratedWorkflow(BaseModel):
    """Validated generated artifacts available to the execution view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: TestPlan
    generated: GeneratedCodeArtifact
    validation: CodeValidationReport


class _ApprovedInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: RiskContract
    gherkin: str
    contract_sha256: str
    gherkin_sha256: str


@dataclass(frozen=True)
class _ModelCall:
    request: ModelRequest
    response: ModelResponse | None
    failure_type: str | None
    failure_provenance: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.model_dump(mode="json"),
            "response": (
                self.response.model_dump(mode="json")
                if self.response is not None
                else None
            ),
            "failure_type": self.failure_type,
            "failure_provenance": self.failure_provenance,
        }


@dataclass(frozen=True)
class _TransitionSnapshot:
    """Exact recorder inputs reused until one logical transition is durable."""

    event: TransformationEvent
    artifact_name: str
    digest: str
    content: bytes


@dataclass(frozen=True)
class _PreparationSnapshot:
    """Complete immutable prepare result created before recorder I/O."""

    prepared: PreparedWorkflow
    transition: _TransitionSnapshot
    ownership: RunOwnership
    run_started_event: LifecycleEvent
    contract_bytes: bytes
    impact_bytes: bytes
    profile_bytes: bytes


class _CapturingGateway:
    """Retain exact request/response provenance without changing the gateway."""

    def __init__(
        self,
        delegate: StructuredModelGateway,
        *,
        before_call: Callable[[ModelRequest], ModelResponse | None],
        after_call: Callable[
            [ModelRequest, ModelResponse | None, Exception | None], None
        ],
    ) -> None:
        self._delegate = delegate
        self._before_call = before_call
        self._after_call = after_call
        self.calls: list[_ModelCall] = []

    @property
    def provider(self) -> str | None:
        value = getattr(self._delegate, "provider", None)
        return value if isinstance(value, str) and value else None

    @property
    def model(self) -> str | None:
        value = getattr(self._delegate, "model", None)
        return value if isinstance(value, str) and value else None

    def generate(self, request: ModelRequest) -> ModelResponse:
        recovered = self._before_call(request)
        if recovered is not None:
            self.calls.append(
                _ModelCall(
                    request=request,
                    response=recovered,
                    failure_type=None,
                    failure_provenance=None,
                )
            )
            return recovered
        try:
            response = self._delegate.generate(request)
        except Exception as error:
            provenance = getattr(error, "provenance", None)
            self.calls.append(
                _ModelCall(
                    request=request,
                    response=None,
                    failure_type=type(error).__name__,
                    failure_provenance=(
                        provenance.model_dump(mode="json")
                        if provenance is not None
                        else None
                    ),
                )
            )
            self._after_call(request, None, error)
            raise
        self._after_call(request, response, None)
        self.calls.append(
            _ModelCall(
                request=request,
                response=response,
                failure_type=None,
                failure_provenance=None,
            )
        )
        return response


class MilestoneOneWorkflow:
    """Enforce the one-way, human-approved Milestone 1 experiment."""

    def __init__(
        self,
        *,
        fixture_directory: str | Path,
        settings: Settings,
        gateway: StructuredModelGateway,
        recorder: ArtifactRecorder,
        fixture_reader: Callable[[Path], bytes] = _read_fixture_bytes,
        run_id: str | None = None,
        planner: Callable[[RiskContract, str, StructuredModelGateway], TestPlan] = create_test_plan,
        generator: Callable[
            [RiskContract, str, TestPlan, StructuredModelGateway],
            GeneratedCodeArtifact,
        ] = generate_pytest,
        validator: Callable[
            [str, RiskContract, TestPlan, str], CodeValidationReport
        ] = validate_generated_code,
        runner_factory: Callable[..., PytestRunner] = PytestRunner,
        classifier: Callable[
            [list[RuntimeObservation], list[RuntimeObservation], RiskContract],
            DifferentialEvidence,
        ] = classify_differential,
        server_factory: Callable[..., ControlledAuthorizationServer] = ControlledAuthorizationServer,
        base_revision: str = _BASE_REVISION,
        candidate_revision: str = _CANDIDATE_REVISION,
    ) -> None:
        if settings.environment_kind is not EnvironmentKind.CONTROLLED_FIXTURE:
            raise ValueError("Milestone 1 supports only the controlled fixture")
        self._validate_revision(base_revision)
        self._validate_revision(candidate_revision)
        if base_revision == candidate_revision:
            raise ValueError("base and candidate revisions must differ")
        self._fixture_directory = Path(fixture_directory)
        # The gateway is constructed before the workflow; retain only public
        # configuration in the long-lived workflow/session object.
        self._settings = settings.public_view()
        self._recorder = recorder
        self._operation_mutex = RLock()
        self._pending_model_results: dict[str, _TransitionSnapshot] = {}
        self._gateway = _CapturingGateway(
            gateway,
            before_call=self._before_model_call,
            after_call=self._after_model_call,
        )
        self._fixture_reader = fixture_reader
        self._run_id = run_id or f"run-{uuid4().hex}"
        self._planner = planner
        self._generator = generator
        self._validator = validator
        self._runner_factory = runner_factory
        self._classifier = classifier
        self._server_factory = server_factory
        self._base_revision = base_revision
        self._candidate_revision = candidate_revision
        self._state = _WorkflowState.NEW
        self._started_at: datetime | None = None
        self._run_directory: Path | None = None
        self._run_handle: RunHandle | None = None
        self._refresh_durable_state = False
        self._prepared: PreparedWorkflow | None = None
        self._approved: _ApprovedInputs | None = None
        self._generated: GeneratedWorkflow | None = None
        self._result: RunRecord | None = None
        self._preparation_snapshot: _PreparationSnapshot | None = None
        self._prepare_start_ambiguous = False
        self._prepare_start_collision: FileExistsError | None = None
        self._approval_snapshot: tuple[_ApprovedInputs, _TransitionSnapshot] | None = None
        self._generation_snapshot: tuple[
            TestPlan, GeneratedCodeArtifact, _TransitionSnapshot
        ] | None = None
        self._validation_snapshot: tuple[
            GeneratedWorkflow, _TransitionSnapshot
        ] | None = None
        self._execution_snapshot: tuple[
            int, RunRecord, _TransitionSnapshot
        ] | None = None
        self._pending_final_record: RunRecord | None = None
        self._pending_final_payload: dict[str, Any] | None = None
        self._finalization_intent: _TransitionSnapshot | None = None
        self._pending_failure_error: Exception | None = None
        self._pending_failure_operation: str | None = None
        self._completed_base: list[RuntimeObservation] = []
        self._completed_candidate: list[RuntimeObservation] = []
        self._execution_paths: list[dict[str, str]] = []
        self._execution_manifests: list[dict[str, Any]] = []

    @property
    def model_gateway_provider(self) -> str:
        return self._gateway.provider or (
            "replay" if self._settings.llm_mode == "replay" else "groq"
        )

    @property
    def model_gateway_model(self) -> str:
        return self._gateway.model or (
            "replay/openai-gpt-oss-120b"
            if self._settings.llm_mode == "replay"
            else self._settings.llm_model
        )

    @property
    def run_handle(self) -> RunHandle:
        """Return the frozen capability required to resume this local run."""
        return RunHandle.model_validate(
            self._require_run_handle().model_dump(mode="json")
        )

    def prepare(self) -> PreparedWorkflow:
        with self._operation_mutex:
            handle = self._run_handle
            if handle is None:
                return self._prepare_impl()
            try:
                self._recorder.verify_run_handle(handle)
            except (OSError, ValueError) as error:
                raise WorkflowTransitionError(
                    "located run identity cannot be verified for preparation recovery"
                ) from error
            with self._recorder.workflow_lease(handle):
                return self._prepare_impl()

    def approve_contract(self, contract: RiskContract, gherkin: str) -> None:
        with self._public_operation():
            self._approve_contract_impl(contract, gherkin)

    def generate(self) -> GeneratedWorkflow:
        with self._public_operation():
            return self._generate_impl()

    def execute(self, *, repeat_count: int) -> RunRecord:
        with self._public_operation():
            return self._execute_impl(repeat_count=repeat_count)

    def result(self) -> RunRecord:
        with self._public_operation():
            return self._result_impl()

    @contextmanager
    def _public_operation(self) -> Iterator[None]:
        """Serialize one public operation in-object and across local processes."""
        with self._operation_mutex:
            handle = self._run_handle
            if handle is None:
                yield
                return
            with self._recorder.workflow_lease(handle):
                if self._refresh_durable_state:
                    self._hydrate_durable_state()
                yield

    def _prepare_impl(self) -> PreparedWorkflow:
        """Load and type the controlled fixture before any model operation."""
        self._guard_recovery_owner("prepare")
        self._require_state(_WorkflowState.NEW, "prepare")
        if self._prepare_start_collision is not None:
            raise self._prepare_start_collision
        if self._preparation_snapshot is None:
            transition_started_at = datetime.now(UTC)
            self._started_at = transition_started_at
            contract_path = self._fixture_directory / "approved_contract.json"
            impact_path = self._fixture_directory / "impact_report.json"
            profile_path = self._fixture_directory / "cvss_profile.json"
            contract_bytes = self._fixture_reader(contract_path)
            impact_bytes = self._fixture_reader(impact_path)
            profile_bytes = self._fixture_reader(profile_path)
            contract = RiskContract.model_validate_json(contract_bytes)
            impact_report = ControlledImpactReport.model_validate_json(impact_bytes)
            cvss_profile = CvssProfile.model_validate_json(profile_bytes)
            calculate_cvss4(cvss_profile)
            gherkin = render_gherkin(contract)
            prepared = PreparedWorkflow(
                run_id=self._run_id,
                environment_kind=self._settings.environment_kind,
                base_revision=self._base_revision,
                candidate_revision=self._candidate_revision,
                environment_warning=_CONTROLLED_FIXTURE_WARNING,
                impact_report=impact_report,
                cvss_profile=cvss_profile,
                contract=contract,
                gherkin=gherkin,
                contract_sha256=canonical_sha256(
                    contract.model_dump(mode="json")
                ),
                cvss_profile_sha256=canonical_sha256(
                    cvss_profile.model_dump(mode="json")
                ),
                gherkin_sha256=_text_sha256(gherkin),
            )
            transition = self._make_transition(
                "prepared",
                prepared.model_dump(mode="json"),
                input_hashes={
                    "approved_contract_fixture": hashlib.sha256(
                        contract_bytes
                    ).hexdigest(),
                    "impact_report_fixture": hashlib.sha256(
                        impact_bytes
                    ).hexdigest(),
                    "cvss_profile_fixture": hashlib.sha256(
                        profile_bytes
                    ).hexdigest(),
                },
                reason_code="controlled_fixture_prepared",
                started_at=transition_started_at,
                finished_at=datetime.now(UTC),
            )
            ownership = RunOwnership.issue(self._run_id)
            self._preparation_snapshot = _PreparationSnapshot(
                prepared=PreparedWorkflow.model_validate(
                    prepared.model_dump(mode="json")
                ),
                transition=transition,
                ownership=ownership,
                run_started_event=LifecycleEvent(
                    event_type=LifecycleEventType.RUN_STARTED,
                    payload={
                        "id": self._run_id,
                        "ownership_token": ownership.ownership_token,
                    },
                ),
                contract_bytes=bytes(contract_bytes),
                impact_bytes=bytes(impact_bytes),
                profile_bytes=bytes(profile_bytes),
            )

        snapshot = self._preparation_snapshot
        if self._run_directory is None:
            if self._prepare_start_ambiguous:
                try:
                    located = self._recorder.locate_run(self._run_id)
                except FileNotFoundError:
                    self._start_preparation_run(snapshot.ownership)
                else:
                    self._run_directory = located
                    self._prepare_start_ambiguous = False
            else:
                self._start_preparation_run(snapshot.ownership)
        self._verify_located_run_identity(
            self._require_run_directory(), snapshot.ownership
        )
        self._record_lifecycle_once(snapshot.run_started_event)
        self._commit_transition(snapshot.transition)
        prepared = snapshot.prepared
        self._prepared = PreparedWorkflow.model_validate(
            prepared.model_dump(mode="json")
        )
        self._state = _WorkflowState.PREPARED
        return PreparedWorkflow.model_validate(prepared.model_dump(mode="json"))

    def _start_preparation_run(self, ownership: RunOwnership) -> None:
        """Start once, distinguishing a known collision from ambiguous I/O."""
        try:
            handle = self._recorder.start_run(self._run_id, ownership)
        except FileExistsError as error:
            self._prepare_start_collision = error
            self._prepare_start_ambiguous = False
            raise
        except OSError:
            self._prepare_start_ambiguous = True
            raise
        self._run_handle = handle
        self._run_directory = self._recorder.verify_run_handle(handle)
        self._prepare_start_ambiguous = False

    def _verify_located_run_identity(
        self,
        run_directory: Path,
        expected: RunOwnership,
    ) -> None:
        """Require Task 2's exact caller-bound proof after an ambiguous start."""
        try:
            handle = self._recorder.resume_run(self._run_id, expected)
            verified = self._recorder.verify_run_handle(handle)
        except (OSError, ValueError) as error:
            raise WorkflowTransitionError(
                "located run identity cannot be verified for preparation recovery"
            ) from error
        if verified != run_directory:
            raise WorkflowTransitionError(
                "located run identity cannot be verified for preparation recovery"
            )
        self._run_handle = handle

    def _approve_contract_impl(self, contract: RiskContract, gherkin: str) -> None:
        """Freeze the prepared contract and an executable-step-aligned Gherkin view."""
        self._guard_recovery_owner("approve_contract")
        self._resume_pending_failure("approve_contract")
        self._require_state(_WorkflowState.PREPARED, "approve_contract")
        assert self._prepared is not None
        if self._approval_snapshot is None:
            transition_started_at = datetime.now(UTC)
            try:
                submitted_contract = RiskContract.model_validate(
                    contract.model_dump(mode="json")
                )
            except (AttributeError, TypeError, ValueError) as error:
                self._finalize_failure(
                    WorkflowStatus.AWAITING_HUMAN_APPROVAL,
                    "contract_not_approved",
                    "The submitted contract was not a valid typed risk contract.",
                    operation="approve_contract",
                    error=error,
                )
                raise ContractApprovalError("contract was not approved") from error
            try:
                alignment = validate_gherkin_alignment(submitted_contract, gherkin)
            except ValueError as error:
                self._finalize_failure(
                    WorkflowStatus.AWAITING_HUMAN_APPROVAL,
                    "gherkin_alignment_failed",
                    "The submitted contract could not define the required security oracles.",
                    operation="approve_contract",
                    error=error,
                )
                raise ContractApprovalError("gherkin_alignment_failed") from error
            reason_code: str | None = None
            explanation: str | None = None
            if submitted_contract != self._prepared.contract:
                reason_code = "contract_not_approved"
                explanation = (
                    "The submitted contract changed the prepared fixture contract."
                )
            elif not alignment.approved:
                reason_code = "gherkin_alignment_failed"
                explanation = (
                    "The submitted Gherkin did not preserve the contract meaning."
                )
            if reason_code is not None:
                error = ContractApprovalError(reason_code)
                self._finalize_failure(
                    WorkflowStatus.AWAITING_HUMAN_APPROVAL,
                    reason_code,
                    explanation or "The contract was not approved.",
                    operation="approve_contract",
                    error=error,
                    details={"alignment_reason_codes": alignment.reason_codes},
                )
                raise error

            approved = _ApprovedInputs(
                contract=submitted_contract,
                gherkin=str(gherkin),
                contract_sha256=canonical_sha256(
                    submitted_contract.model_dump(mode="json")
                ),
                gherkin_sha256=_text_sha256(gherkin),
            )
            transition = self._make_transition(
                "approved",
                {
                    "prepared": {
                        "contract": self._prepared.contract.model_dump(mode="json"),
                        "gherkin": self._prepared.gherkin,
                        "contract_sha256": self._prepared.contract_sha256,
                        "gherkin_sha256": self._prepared.gherkin_sha256,
                    },
                    "approved": approved.model_dump(mode="json"),
                },
                input_hashes={
                    "prepared_contract": self._prepared.contract_sha256,
                    "prepared_gherkin": self._prepared.gherkin_sha256,
                },
                reason_code="contract_and_gherkin_approved",
                started_at=transition_started_at,
                finished_at=datetime.now(UTC),
            )
            self._approval_snapshot = (
                _ApprovedInputs.model_validate(approved.model_dump(mode="json")),
                transition,
            )
        else:
            submitted = RiskContract.model_validate(contract.model_dump(mode="json"))
            approved = self._approval_snapshot[0]
            if submitted != approved.contract or gherkin != approved.gherkin:
                raise WorkflowTransitionError(
                    "approval recovery requires the exact pending contract and Gherkin"
                )

        approved, transition = self._approval_snapshot
        self._record_lifecycle_once(
            LifecycleEvent(
                event_type=LifecycleEventType.CONTRACT_APPROVED,
                payload={"id": approved.contract.contract_id},
            )
        )
        self._commit_transition(transition)
        self._approved = _ApprovedInputs.model_validate(
            approved.model_dump(mode="json")
        )
        self._state = _WorkflowState.APPROVED

    def _generate_impl(self) -> GeneratedWorkflow:
        """Run the two bounded model calls and deterministic full-code validator."""
        self._guard_recovery_owner("generate")
        self._resume_pending_failure("generate")
        if self._state not in {_WorkflowState.APPROVED, _WorkflowState.GENERATED}:
            raise WorkflowTransitionError(
                "generate requires approved or recoverable generated state; "
                f"current state is {self._state.value}"
            )
        assert self._approved is not None
        if self._state is _WorkflowState.APPROVED:
            if self._generation_snapshot is None:
                generation_started_at = datetime.now(UTC)
                try:
                    plan = self._planner(
                        self._approved.contract,
                        self._approved.gherkin,
                        self._gateway,
                    )
                    generated_artifact = self._generator(
                        self._approved.contract,
                        self._approved.gherkin,
                        plan,
                        self._gateway,
                    )
                except OperationJournalInterruptedError:
                    raise
                except Exception as error:
                    status, reason_code, explanation = self._generation_failure(error)
                    self._finalize_failure(
                        status,
                        reason_code,
                        explanation,
                        operation="generate",
                        error=error,
                        details={"llm_calls": self._model_calls()},
                    )
                    raise
                plan = TestPlan.model_validate(plan.model_dump(mode="json"))
                generated_artifact = GeneratedCodeArtifact.model_validate(
                    generated_artifact.model_dump(mode="json")
                )
                transition = self._make_transition(
                    "generated",
                    {
                        "plan": plan.model_dump(mode="json"),
                        "generated": generated_artifact.model_dump(mode="json"),
                        "llm_calls": self._model_calls(),
                    },
                    input_hashes={
                        "approved_contract": self._approved.contract_sha256,
                        "approved_gherkin": self._approved.gherkin_sha256,
                    },
                    reason_code="replay_plan_and_code_generated",
                    started_at=generation_started_at,
                    finished_at=datetime.now(UTC),
                )
                self._generation_snapshot = (plan, generated_artifact, transition)
            self._commit_transition(self._generation_snapshot[2])
            self._state = _WorkflowState.GENERATED

        plan, generated_artifact, _ = self._generation_snapshot
        if self._validation_snapshot is None:
            validation_started_at = datetime.now(UTC)
            try:
                report = self._validator(
                    generated_artifact.code,
                    self._approved.contract,
                    plan,
                    self._approved.gherkin,
                )
                if not isinstance(report, CodeValidationReport):
                    raise TypeError("validator must return a CodeValidationReport")
            except Exception as error:
                self._finalize_failure(
                    WorkflowStatus.VALIDATION_FAILED,
                    "validator_invocation_failed",
                    "Deterministic generated-code validation could not complete.",
                    operation="generate",
                    error=error,
                    details={
                        "generated_code_sha256": _text_sha256(
                            generated_artifact.code
                        ),
                        "llm_calls": self._model_calls(),
                    },
                )
                raise
            metadata_matches = (
                generated_artifact.implemented_steps == report.implemented_steps
                and generated_artifact.used_primitives == report.used_primitives
                and generated_artifact.contract_id
                == self._approved.contract.contract_id
            )
            if not report.approved or not metadata_matches:
                error = UnsafeGeneratedCodeError(report)
                reason_code = (
                    "unsafe_generated_code"
                    if not report.approved
                    else "generated_metadata_mismatch"
                )
                self._finalize_failure(
                    WorkflowStatus.VALIDATION_FAILED,
                    reason_code,
                    "Generated code was rejected by deterministic validation.",
                    operation="generate",
                    error=error,
                    details={
                        "validation": report.model_dump(mode="json"),
                        "generated_code_sha256": _text_sha256(
                            generated_artifact.code
                        ),
                        "llm_calls": self._model_calls(),
                    },
                )
                raise error

            generated = GeneratedWorkflow(
                plan=plan,
                generated=generated_artifact,
                validation=report,
            )
            transition = self._make_transition(
                "validated",
                generated.model_dump(mode="json"),
                input_hashes={
                    "generated_code": report.code_sha256,
                    "test_plan": canonical_sha256(plan.model_dump(mode="json")),
                },
                reason_code="generated_code_validated",
                started_at=validation_started_at,
                finished_at=datetime.now(UTC),
            )
            self._validation_snapshot = (
                GeneratedWorkflow.model_validate(generated.model_dump(mode="json")),
                transition,
            )
        generated, transition = self._validation_snapshot
        self._commit_transition(transition)
        self._generated = GeneratedWorkflow.model_validate(
            generated.model_dump(mode="json")
        )
        self._state = _WorkflowState.VALIDATED
        return GeneratedWorkflow.model_validate(generated.model_dump(mode="json"))

    def _execute_impl(self, *, repeat_count: int) -> RunRecord:
        """Execute fresh base/candidate pairs, then classify the complete set once."""
        self._guard_recovery_owner("execute")
        self._resume_pending_failure("execute")
        if (
            isinstance(repeat_count, bool)
            or not isinstance(repeat_count, int)
            or not MIN_REPEAT_COUNT <= repeat_count <= MAX_REPEAT_COUNT
        ):
            raise ValueError(
                f"repeat_count must be an integer from {MIN_REPEAT_COUNT} "
                f"to {MAX_REPEAT_COUNT}"
            )
        if self._state is _WorkflowState.EXECUTED:
            if self._execution_snapshot is None:
                raise WorkflowTransitionError("executed state has no recovery snapshot")
            pending_count, record, _ = self._execution_snapshot
            if repeat_count != pending_count:
                raise WorkflowTransitionError(
                    "finalization recovery requires the original repeat_count"
                )
            self._finalize(record)
            return self._result_impl()
        self._require_state(_WorkflowState.VALIDATED, "execute")
        assert self._approved is not None
        assert self._generated is not None
        if self._execution_snapshot is None:
            execution_started_at = datetime.now(UTC)
            runner: Any | None = None
            try:
                recovered_pairs: list[
                    tuple[RuntimeObservation | None, RuntimeObservation | None]
                ] = []
                for repetition_index in range(1, repeat_count + 1):
                    base = self._recover_experiment_result(
                        repetition_index=repetition_index,
                        side="base",
                        revision=self._base_revision,
                        repeat_count=repeat_count,
                    )
                    if base is not None:
                        self._append_observation(
                            base,
                            self._base_revision,
                            self._completed_base,
                            repetition_index=repetition_index,
                        )
                    candidate = self._recover_experiment_result(
                        repetition_index=repetition_index,
                        side="candidate",
                        revision=self._candidate_revision,
                        repeat_count=repeat_count,
                    )
                    if candidate is not None:
                        self._append_observation(
                            candidate,
                            self._candidate_revision,
                            self._completed_candidate,
                            repetition_index=repetition_index,
                        )
                    recovered_pairs.append((base, candidate))
                if any(
                    base is None or candidate is None
                    for base, candidate in recovered_pairs
                ):
                    try:
                        runner = self._runner_factory(
                            generated_code=self._generated.generated.code,
                            contract=self._approved.contract,
                            plan=self._generated.plan,
                            gherkin=self._approved.gherkin,
                            artifact_root=self._require_run_directory() / "executions",
                        )
                    except Exception as error:
                        self._finalize_failure(
                            WorkflowStatus.EXECUTION_INCONCLUSIVE,
                            "execution_runner_construction_failed",
                            "The isolated execution runner could not be constructed.",
                            operation="execute",
                            error=error,
                            details=self._execution_details(),
                        )
                        raise

                for repetition_index, (base, candidate) in enumerate(
                    recovered_pairs, start=1
                ):
                    if base is not None and candidate is not None:
                        continue
                    assert runner is not None
                    with self._new_server("secure") as base_server, self._new_server(
                        "vulnerable"
                    ) as candidate_server:
                        if base is None:
                            base = self._run_experiment_operation(
                                runner=runner,
                                target=ExecutionTarget(
                                    base_url=base_server.base_url,
                                    username=_ADMINISTRATOR,
                                    password=_PASSWORD,
                                    revision=self._base_revision,
                                ),
                                side="base",
                                repetition_index=repetition_index,
                                repeat_count=repeat_count,
                            )
                        self._append_observation(
                            base,
                            self._base_revision,
                            self._completed_base,
                            repetition_index=repetition_index,
                        )
                        if candidate is None:
                            candidate = self._run_experiment_operation(
                                runner=runner,
                                target=ExecutionTarget(
                                    base_url=candidate_server.base_url,
                                    username=_ADMINISTRATOR,
                                    password=_PASSWORD,
                                    revision=self._candidate_revision,
                                ),
                                side="candidate",
                                repetition_index=repetition_index,
                                repeat_count=repeat_count,
                            )
                        self._append_observation(
                            candidate,
                            self._candidate_revision,
                            self._completed_candidate,
                            repetition_index=repetition_index,
                        )
                classifier_result = self._classifier(
                    list(self._completed_base),
                    list(self._completed_candidate),
                    self._approved.contract,
                )
                if not isinstance(classifier_result, DifferentialEvidence):
                    raise InvalidClassifierResultError()
                try:
                    expected_evidence = classify_differential(
                        list(self._completed_base),
                        list(self._completed_candidate),
                        self._approved.contract,
                    )
                    if classifier_result != expected_evidence:
                        raise InvalidClassifierResultError()
                    evidence = DifferentialEvidence.model_validate(
                        {
                            **classifier_result.model_dump(mode="json"),
                            "execution_manifest_sha256s": [
                                item["sha256"]
                                for item in self._execution_manifests
                            ],
                        }
                    )
                except (TypeError, ValueError) as error:
                    raise InvalidClassifierResultError() from error
                assert self._prepared is not None
                severity_assessment = assess_differential_severity(
                    evidence,
                    self._prepared.cvss_profile,
                )
            except Exception as error:
                if runner is not None:
                    self._capture_execution_path(runner, "failed")
                status, reason_code, explanation = self._execution_failure(error)
                self._finalize_failure(
                    status,
                    reason_code,
                    explanation,
                    operation="execute",
                    error=error,
                    details=self._execution_details(),
                )
                raise

            record = RunRecord(
                run_id=self._run_id,
                environment_kind=self._settings.environment_kind,
                base_revision=self._base_revision,
                candidate_revision=self._candidate_revision,
                status=evidence.status,
                reason_code=evidence.reason_code,
                explanation=evidence.explanation,
                started_at=self._require_started_at(),
                finished_at=datetime.now(UTC),
                differential_evidence=evidence,
                severity_assessment=severity_assessment,
                execution_manifest_sha256s=[
                    item["sha256"] for item in self._execution_manifests
                ],
            )
            transition = self._make_transition(
                "executed",
                {
                    **self._execution_details(),
                    "environment_kind": self._settings.environment_kind.value,
                    "environment_warning": _CONTROLLED_FIXTURE_WARNING,
                    "evidence": evidence.model_dump(mode="json"),
                    "run_record": record.model_dump(mode="json"),
                },
                input_hashes={
                    "approved_contract": self._approved.contract_sha256,
                    "approved_gherkin": self._approved.gherkin_sha256,
                    "generated_code": self._generated.validation.code_sha256,
                    "cvss_profile": self._prepared.cvss_profile_sha256,
                },
                reason_code=evidence.reason_code,
                started_at=execution_started_at,
                finished_at=datetime.now(UTC),
            )
            self._execution_snapshot = (
                repeat_count,
                RunRecord.model_validate(record.model_dump(mode="json")),
                transition,
            )
        pending_count, record, transition = self._execution_snapshot
        if repeat_count != pending_count:
            raise WorkflowTransitionError(
                "execution recovery requires the original repeat_count"
            )
        self._commit_transition(transition)
        self._state = _WorkflowState.EXECUTED
        self._finalize(record)
        return self._result_impl()

    def _result_impl(self) -> RunRecord:
        """Return a result only after recorder finalization completed."""
        self._guard_recovery_owner("result")
        if self._state is not _WorkflowState.FINALIZED or self._result is None:
            raise WorkflowTransitionError("result is unavailable before finalization")
        if not self._final_record_completed(self._result):
            raise WorkflowTransitionError("terminal record integrity check failed")
        self._verify_record_manifests(self._result)
        return RunRecord.model_validate(self._result.model_dump(mode="json"))

    def _new_server(self, behavior: str) -> ControlledAuthorizationServer:
        return self._server_factory(
            behavior=behavior,
            administrator_username=_ADMINISTRATOR,
            clerk_username=_CLERK,
            password=_PASSWORD,
        )

    @staticmethod
    def _append_observation(
        observation: RuntimeObservation,
        expected_revision: str,
        destination: list[RuntimeObservation],
        *,
        repetition_index: int | None = None,
    ) -> None:
        if not isinstance(observation, RuntimeObservation):
            raise TypeError("runner must return a RuntimeObservation")
        if observation.revision != expected_revision:
            raise ValueError("runner observation revision did not match its target")
        if repetition_index is not None:
            expected_offset = repetition_index - 1
            if expected_offset < 0 or len(destination) < expected_offset:
                raise WorkflowTransitionError(
                    "runtime observations must be registered in repetition order"
                )
            if len(destination) > expected_offset:
                if destination[expected_offset] != observation:
                    raise WorkflowTransitionError(
                        "runtime observation conflicts with durable repetition state"
                    )
                return
        destination.append(observation)

    def _capture_execution_path(self, runner: Any, side: str) -> None:
        artifacts = getattr(runner, "last_artifacts", None)
        if artifacts is None:
            return
        item = {"side": side}
        for name in (
            "run_directory",
            "pytest_config_path",
            "feature_path",
            "test_path",
            "observation_path",
            "pytest_outcome_path",
            "stdout_path",
            "stderr_path",
        ):
            value = getattr(artifacts, name, None)
            if value is not None:
                item[name] = str(value)
        if item not in self._execution_paths:
            self._execution_paths.append(item)

    def _execution_details(self) -> dict[str, Any]:
        return {
            "base_observations": [
                item.model_dump(mode="json") for item in self._completed_base
            ],
            "candidate_observations": [
                item.model_dump(mode="json") for item in self._completed_candidate
            ],
            "execution_artifacts": self._execution_paths,
            "execution_manifests": self._execution_manifests,
        }

    def _recover_experiment_result(
        self,
        *,
        repetition_index: int,
        side: str,
        revision: str,
        repeat_count: int,
    ) -> RuntimeObservation | None:
        """Return a durable result, detect an unknown intent, or request a new call."""
        intent_name = self._experiment_operation_name(
            repetition_index, side, "intent"
        )
        result_name = self._experiment_operation_name(
            repetition_index, side, "result"
        )
        expected_intent = self._experiment_intent_payload(
            repetition_index=repetition_index,
            side=side,
            revision=revision,
            repeat_count=repeat_count,
        )
        intent = self._load_named_payload(intent_name)
        if intent is None:
            return None
        if intent != expected_intent:
            raise WorkflowTransitionError(
                f"experiment {repetition_index}:{side} conflicts with durable intent"
            )
        result = self._load_named_payload(result_name)
        if result is None:
            raise InterruptedExternalOperationError(
                "experiment", f"{repetition_index}:{side}"
            )
        if result.get("intent_sha256") != canonical_sha256(expected_intent):
            raise WorkflowTransitionError(
                f"experiment {repetition_index}:{side} result intent mismatch"
            )
        if result.get("outcome") == "succeeded":
            observation = RuntimeObservation.model_validate(
                result.get("observation")
            )
            if observation.revision != revision:
                raise WorkflowTransitionError(
                    f"experiment {repetition_index}:{side} revision mismatch"
                )
            manifest_path = result.get("manifest_path")
            manifest_sha256 = result.get("manifest_sha256")
            if not isinstance(manifest_path, str) or not isinstance(
                manifest_sha256, str
            ):
                raise WorkflowTransitionError(
                    f"experiment {repetition_index}:{side} lacks a manifest binding"
                )
            self._verify_execution_manifest(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                repetition_index=repetition_index,
                side=side,
                revision=revision,
                expected_observation=observation,
            )
            self._register_execution_manifest(
                repetition_index=repetition_index,
                side=side,
                path=manifest_path,
                sha256=manifest_sha256,
            )
            return observation
        if result.get("outcome") == "failed":
            raise WorkflowTransitionError(
                f"experiment {repetition_index}:{side} previously failed as "
                f"{result.get('failure_type')}"
            )
        raise WorkflowTransitionError(
            f"experiment {repetition_index}:{side} durable result is invalid"
        )

    def _run_experiment_operation(
        self,
        *,
        runner: Any,
        target: ExecutionTarget,
        side: str,
        repetition_index: int,
        repeat_count: int,
    ) -> RuntimeObservation:
        """Commit intent, invoke once, then commit the exact returned observation."""
        intent_name = self._experiment_operation_name(
            repetition_index, side, "intent"
        )
        result_name = self._experiment_operation_name(
            repetition_index, side, "result"
        )
        intent = self._experiment_intent_payload(
            repetition_index=repetition_index,
            side=side,
            revision=target.revision,
            repeat_count=repeat_count,
        )
        self._commit_named_payload(
            event_name=f"experiment_{repetition_index}_{side}_intent",
            artifact_name=intent_name,
            payload=intent,
            input_hashes={
                "approved_contract": self._approved.contract_sha256,
                "generated_code": self._generated.validation.code_sha256,
            },
            reason_code="experiment_operation_intent_committed",
        )
        operation_started_at = datetime.now(UTC)
        try:
            observation = runner.run(target)
            self._append_observation(observation, target.revision, [])
        except Exception as error:
            self._capture_execution_path(runner, side)
            failure_payload = {
                "operation_kind": "experiment",
                "operation_id": f"{repetition_index}:{side}",
                "intent_sha256": canonical_sha256(intent),
                "outcome": "failed",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "execution_artifacts": self._execution_paths[-1:] if self._execution_paths else [],
            }
            self._commit_named_payload(
                event_name=f"experiment_{repetition_index}_{side}_result",
                artifact_name=result_name,
                payload=failure_payload,
                input_hashes={"experiment_intent": canonical_sha256(intent)},
                reason_code="experiment_operation_failure_committed",
            )
            raise
        self._capture_execution_path(runner, side)
        operation_finished_at = datetime.now(UTC)
        manifest_path, manifest_sha256 = self._snapshot_execution_manifest(
            runner=runner,
            observation=observation,
            revision=target.revision,
            repetition_index=repetition_index,
            side=side,
            started_at=operation_started_at,
            finished_at=operation_finished_at,
            intent_sha256=canonical_sha256(intent),
        )
        self._register_execution_manifest(
            repetition_index=repetition_index,
            side=side,
            path=manifest_path,
            sha256=manifest_sha256,
        )
        success_payload = {
            "operation_kind": "experiment",
            "operation_id": f"{repetition_index}:{side}",
            "intent_sha256": canonical_sha256(intent),
            "outcome": "succeeded",
            "observation": observation.model_dump(mode="json"),
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
            "execution_artifacts": self._execution_paths[-1:] if self._execution_paths else [],
        }
        self._commit_named_payload(
            event_name=f"experiment_{repetition_index}_{side}_result",
            artifact_name=result_name,
            payload=success_payload,
            input_hashes={"experiment_intent": canonical_sha256(intent)},
            reason_code="experiment_operation_result_committed",
        )
        return observation

    def _snapshot_execution_manifest(
        self,
        *,
        runner: Any,
        observation: RuntimeObservation,
        revision: str,
        repetition_index: int,
        side: str,
        started_at: datetime,
        finished_at: datetime,
        intent_sha256: str,
    ) -> tuple[str, str]:
        """Copy every completed runner file through the authenticated recorder."""
        artifacts = getattr(runner, "last_artifacts", None)
        if not isinstance(artifacts, ExecutionArtifacts):
            raise ExecutionManifestError(
                "completed runner did not expose typed ExecutionArtifacts"
            )
        source_paths = {
            "feature": artifacts.feature_path,
            "generated_test": artifacts.test_path,
            "pytest_config": artifacts.pytest_config_path,
            "raw_event_sidecar": Path(
                f"{artifacts.observation_path}.events.jsonl"
            ),
            "final_observation": artifacts.observation_path,
            "structured_pytest_outcome": artifacts.pytest_outcome_path,
            "stdout": artifacts.stdout_path,
            "stderr": artifacts.stderr_path,
        }
        filenames = {
            "feature": "authorization.feature",
            "generated_test": "test_authorization.py",
            "pytest_config": "pytest.ini",
            "raw_event_sidecar": "observation.events.jsonl",
            "final_observation": "observation.json",
            "structured_pytest_outcome": "pytest-outcome.json",
            "stdout": "pytest.stdout.txt",
            "stderr": "pytest.stderr.txt",
        }
        file_records: dict[str, ExecutionFile] = {}
        prefix = f"artifacts/executions/{repetition_index:04d}-{side}/files"
        for kind, source_path in source_paths.items():
            try:
                content = _read_regular_beneath_directory(
                    artifacts.run_directory,
                    source_path,
                )
            except (OSError, TypeError, ValueError) as error:
                raise ExecutionManifestError(
                    f"completed runner file is unavailable or unsafe: {kind}"
                ) from error
            relative_path = f"{prefix}/{filenames[kind]}"
            digest = hashlib.sha256(content).hexdigest()
            event = TransformationEvent(
                event_type=(
                    f"execution_{repetition_index}_{side}_{kind}_snapshotted"
                ),
                inputs={"experiment_intent": "experiment_intent"},
                outputs={relative_path: relative_path},
                input_hashes={"experiment_intent": intent_sha256},
                output_hashes={relative_path: digest},
                versions={"triageguard": "2.0.0", "workflow": "milestone_one"},
                started_at=started_at,
                finished_at=max(
                    finished_at,
                    started_at + timedelta(microseconds=1),
                ),
                reason_code="execution_file_snapshotted",
            )
            self._commit_transition(
                _TransitionSnapshot(
                    event=event,
                    artifact_name=relative_path,
                    digest=digest,
                    content=content,
                )
            )
            file_records[kind] = ExecutionFile(
                relative_path=relative_path,
                sha256=digest,
                byte_count=len(content),
            )

        manifest = ExecutionManifest(
            side=side,
            revision=revision,
            repetition_index=repetition_index,
            started_at=started_at,
            finished_at=max(
                finished_at,
                started_at + timedelta(microseconds=1),
            ),
            files=file_records,
        )
        manifest_path = (
            f"artifacts/executions/{repetition_index:04d}-{side}/manifest.json"
        )
        snapshot = self._commit_named_payload(
            event_name=f"execution_{repetition_index}_{side}_manifest",
            artifact_name=manifest_path,
            payload=manifest.model_dump(mode="json"),
            input_hashes={
                "experiment_intent": intent_sha256,
                **{
                    kind: record.sha256
                    for kind, record in file_records.items()
                },
            },
            reason_code="execution_manifest_committed",
        )
        self._verify_execution_manifest(
            manifest_path=manifest_path,
            manifest_sha256=snapshot.digest,
            repetition_index=repetition_index,
            side=side,
            revision=revision,
            expected_observation=observation,
        )
        return manifest_path, snapshot.digest

    def _register_execution_manifest(
        self,
        *,
        repetition_index: int,
        side: str,
        path: str,
        sha256: str,
    ) -> None:
        binding = {
            "repetition_index": repetition_index,
            "side": side,
            "path": path,
            "sha256": sha256,
        }
        for existing in self._execution_manifests:
            if (
                existing["repetition_index"] == repetition_index
                and existing["side"] == side
            ):
                if existing != binding:
                    raise WorkflowTransitionError(
                        "execution manifest binding conflicts with durable state"
                    )
                return
        self._execution_manifests.append(binding)

    def _verify_execution_manifest(
        self,
        *,
        manifest_path: str,
        manifest_sha256: str,
        repetition_index: int,
        side: str,
        revision: str,
        expected_observation: RuntimeObservation,
    ) -> None:
        """Verify a manifest and every recorder-owned file it transitively binds."""
        try:
            manifest_bytes = self._recorder.read_artifact(
                self._require_run_handle(), manifest_path
            )
            if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
                raise ValueError("manifest digest mismatch")
            manifest = ExecutionManifest.model_validate_json(manifest_bytes)
            if (
                manifest.repetition_index != repetition_index
                or manifest.side != side
                or manifest.revision != revision
            ):
                raise ValueError("manifest identity mismatch")
            final_observation_bytes: bytes | None = None
            for kind, file_record in manifest.files.items():
                content = self._recorder.read_artifact(
                    self._require_run_handle(), file_record.relative_path
                )
                if (
                    len(content) != file_record.byte_count
                    or hashlib.sha256(content).hexdigest() != file_record.sha256
                ):
                    raise ValueError("manifest file digest mismatch")
                if kind == "final_observation":
                    final_observation_bytes = content
            if final_observation_bytes is None:
                raise ValueError("manifest lacks its final observation")
            envelope = RuntimeObservationEnvelope.model_validate_json(
                final_observation_bytes
            )
            persisted_observation = RuntimeObservation.model_validate(
                envelope.model_dump(
                    mode="json", exclude={"contract_sha256"}
                )
            )
            if persisted_observation != expected_observation:
                raise ValueError(
                    "manifest final observation contradicts the operation result"
                )
            if (
                self._approved is None
                or envelope.contract_sha256 != self._approved.contract_sha256
            ):
                raise ValueError(
                    "manifest final observation is bound to another contract"
                )
        except Exception as error:
            if isinstance(error, WorkflowTransitionError):
                raise
            raise WorkflowTransitionError(
                f"execution manifest integrity check failed: {manifest_path}"
            ) from error

    def _experiment_intent_payload(
        self,
        *,
        repetition_index: int,
        side: str,
        revision: str,
        repeat_count: int,
    ) -> dict[str, Any]:
        assert self._approved is not None
        assert self._generated is not None
        return {
            "operation_kind": "experiment",
            "operation_id": f"{repetition_index}:{side}",
            "side": side,
            "revision": revision,
            "repetition_index": repetition_index,
            "repeat_count": repeat_count,
            "contract_sha256": self._approved.contract_sha256,
            "gherkin_sha256": self._approved.gherkin_sha256,
            "generated_code_sha256": self._generated.validation.code_sha256,
        }

    @staticmethod
    def _experiment_operation_name(
        repetition_index: int, side: str, phase: str
    ) -> str:
        return (
            "artifacts/operations/experiment/"
            f"{repetition_index:04d}-{side}/{phase}.json"
        )

    def _finalize_failure(
        self,
        status: WorkflowStatus,
        reason_code: str,
        explanation: str,
        *,
        operation: str,
        error: Exception,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Best-effort idempotent finalization that never replaces the root error."""
        if self._state is _WorkflowState.FINALIZED:
            return
        self._pending_failure_error = error
        self._pending_failure_operation = operation
        if (
            self._pending_final_record is not None
            and self._pending_final_record.status is status
            and self._pending_final_record.reason_code == reason_code
        ):
            record = self._pending_final_record
        else:
            record = RunRecord(
                run_id=self._run_id,
                environment_kind=self._settings.environment_kind,
                base_revision=self._base_revision,
                candidate_revision=self._candidate_revision,
                status=status,
                reason_code=reason_code,
                explanation=explanation,
                started_at=self._require_started_at(),
                finished_at=datetime.now(UTC),
                differential_evidence=None,
                execution_manifest_sha256s=[
                    item["sha256"] for item in self._execution_manifests
                ],
            )
        failure_payload = {
            "status": status.value,
            "reason_code": reason_code,
            "operation": operation,
            "exception_type": type(error).__name__,
            "details": dict(details or {}),
        }
        try:
            self._finalize(record, extra_payload=failure_payload)
        except Exception:  # noqa: BLE001 - failure finalization cannot mask root error
            # The original stage failure remains the public failure. Recorder journal
            # events and every earlier immutable artifact remain readable for recovery.
            return
        self._pending_failure_error = None
        self._pending_failure_operation = None

    def _resume_pending_failure(self, operation: str) -> None:
        """Retry only finalization for a previously failed stage, then re-raise it."""
        error = self._pending_failure_error
        if error is None:
            return
        if self._pending_failure_operation != operation:
            raise WorkflowTransitionError(
                f"pending failure belongs to {self._pending_failure_operation}"
            )
        assert self._pending_final_record is not None
        try:
            self._finalize(
                self._pending_final_record,
                extra_payload=self._pending_final_payload,
            )
        except Exception:  # noqa: BLE001 - the original stage error remains primary
            raise error
        self._pending_failure_error = None
        self._pending_failure_operation = None
        raise error

    def _guard_recovery_owner(self, operation: str) -> None:
        """Reject alternate methods before they can mutate pending recovery state."""
        if (
            self._state is _WorkflowState.NEW
            and self._preparation_snapshot is not None
            and operation != "prepare"
        ):
            raise WorkflowTransitionError("pending preparation belongs to prepare")
        if (
            self._pending_failure_error is not None
            and self._pending_failure_operation != operation
        ):
            prefix = (
                "result is unavailable before finalization; "
                if operation == "result"
                else ""
            )
            raise WorkflowTransitionError(
                f"{prefix}pending failure belongs to "
                f"{self._pending_failure_operation}"
            )

    def _restore_run(self, handle: RunHandle) -> None:
        """Attach an authenticated existing run and reconstruct its durable state."""
        if self._state is not _WorkflowState.NEW or self._run_handle is not None:
            raise WorkflowTransitionError("resume requires a fresh workflow object")
        self._run_handle = self._recorder.resume_run(
            handle.run_id, handle.ownership
        )
        self._run_directory = self._recorder.verify_run_handle(self._run_handle)
        self._run_id = handle.run_id
        with self._recorder.workflow_lease(self._run_handle):
            self._hydrate_durable_state()
        self._refresh_durable_state = True
        if self._prepared is None:
            raise WorkflowTransitionError(
                "existing run has no complete prepared workflow snapshot"
            )

    def _hydrate_durable_state(self) -> None:
        """Advance in-memory state only from verified completed stage artifacts."""
        if self._run_handle is None:
            return
        prepared = self._load_workflow_transition("prepared")
        if prepared is not None:
            payload, snapshot = prepared
            prepared_model = PreparedWorkflow.model_validate(payload)
            if prepared_model.run_id != self._run_id:
                raise WorkflowTransitionError("prepared snapshot run_id mismatch")
            self._prepared = prepared_model
            self._base_revision = prepared_model.base_revision
            self._candidate_revision = prepared_model.candidate_revision
            self._preparation_snapshot = self._preparation_snapshot or _PreparationSnapshot(
                prepared=prepared_model,
                transition=snapshot,
                ownership=self._run_handle.ownership,
                run_started_event=LifecycleEvent(
                    event_type=LifecycleEventType.RUN_STARTED,
                    payload={
                        "id": self._run_id,
                        "ownership_token": self._run_handle.ownership.ownership_token,
                    },
                ),
                contract_bytes=b"",
                impact_bytes=b"",
                profile_bytes=b"",
            )
            self._started_at = snapshot.event.started_at
            if self._state is _WorkflowState.NEW:
                self._state = _WorkflowState.PREPARED

        approved = self._load_workflow_transition("approved")
        if approved is not None:
            payload, transition = approved
            approved_model = _ApprovedInputs.model_validate(payload["approved"])
            self._approved = approved_model
            self._approval_snapshot = (approved_model, transition)
            if self._state in {_WorkflowState.NEW, _WorkflowState.PREPARED}:
                self._state = _WorkflowState.APPROVED

        generated = self._load_workflow_transition("generated")
        if generated is not None:
            payload, transition = generated
            plan = TestPlan.model_validate(payload["plan"])
            artifact = GeneratedCodeArtifact.model_validate(payload["generated"])
            self._generation_snapshot = (plan, artifact, transition)
            if self._state in {
                _WorkflowState.NEW,
                _WorkflowState.PREPARED,
                _WorkflowState.APPROVED,
            }:
                self._state = _WorkflowState.GENERATED

        validated = self._load_workflow_transition("validated")
        if validated is not None:
            payload, transition = validated
            generated_model = GeneratedWorkflow.model_validate(payload)
            self._generated = generated_model
            self._validation_snapshot = (generated_model, transition)
            if self._state is not _WorkflowState.FINALIZED:
                self._state = _WorkflowState.VALIDATED

        executed = self._load_workflow_transition("executed")
        if executed is not None:
            payload, transition = executed
            if self._prepared is None:
                raise WorkflowTransitionError(
                    "executed snapshot lacks its prepared CVSS profile"
                )
            if transition.event.input_hashes.get("cvss_profile") != (
                self._prepared.cvss_profile_sha256
            ):
                raise WorkflowTransitionError(
                    "executed severity is bound to another CVSS profile"
                )
            record_payload = payload.get("run_record")
            if not isinstance(record_payload, dict):
                raise WorkflowTransitionError(
                    "executed snapshot lacks its exact terminal record"
                )
            try:
                record = RunRecord.model_validate(record_payload)
            except (TypeError, ValueError) as error:
                raise WorkflowTransitionError(
                    "executed severity record is invalid"
                ) from error
            base_payloads = payload.get("base_observations", [])
            candidate_payloads = payload.get("candidate_observations", [])
            self._completed_base = [
                RuntimeObservation.model_validate(item) for item in base_payloads
            ]
            self._completed_candidate = [
                RuntimeObservation.model_validate(item) for item in candidate_payloads
            ]
            execution_paths = payload.get("execution_artifacts", [])
            if not isinstance(execution_paths, list):
                raise WorkflowTransitionError("execution artifact paths are invalid")
            self._execution_paths = [dict(item) for item in execution_paths]
            execution_manifests = payload.get("execution_manifests", [])
            if not isinstance(execution_manifests, list):
                raise WorkflowTransitionError(
                    "execution manifest bindings are invalid"
                )
            self._execution_manifests = [
                dict(item) for item in execution_manifests
            ]
            self._verify_record_manifests(record)
            self._execution_snapshot = (
                len(self._completed_base),
                record,
                transition,
            )
            self._state = _WorkflowState.EXECUTED

        finalization = self._load_workflow_transition("finalization_intent")
        if finalization is not None:
            payload, transition = finalization
            try:
                record = RunRecord.model_validate(payload["run_record"])
            except (KeyError, TypeError, ValueError) as error:
                raise WorkflowTransitionError(
                    "finalization severity record is invalid"
                ) from error
            self._pending_final_record = record
            failure_payload = payload.get("failure", {})
            if not isinstance(failure_payload, dict):
                raise WorkflowTransitionError("finalization failure payload is invalid")
            self._pending_final_payload = failure_payload
            details = failure_payload.get("details", {})
            if isinstance(details, dict):
                if not self._completed_base:
                    base_payloads = details.get("base_observations", [])
                    if isinstance(base_payloads, list):
                        self._completed_base = [
                            RuntimeObservation.model_validate(item)
                            for item in base_payloads
                        ]
                if not self._completed_candidate:
                    candidate_payloads = details.get(
                        "candidate_observations", []
                    )
                    if isinstance(candidate_payloads, list):
                        self._completed_candidate = [
                            RuntimeObservation.model_validate(item)
                            for item in candidate_payloads
                        ]
                if not self._execution_manifests:
                    bindings = details.get("execution_manifests", [])
                    if isinstance(bindings, list):
                        self._execution_manifests = [
                            dict(item) for item in bindings
                        ]
            self._verify_record_manifests(record)
            self._finalization_intent = transition
            try:
                terminal_bytes = self._recorder.read_artifact(
                    self._run_handle, "run_record.json"
                )
            except FileNotFoundError:
                terminal_bytes = None
            if terminal_bytes is not None:
                try:
                    terminal = RunRecord.model_validate_json(terminal_bytes)
                except (TypeError, ValueError) as error:
                    raise WorkflowTransitionError(
                        "terminal severity record is invalid"
                    ) from error
                if terminal != record:
                    raise WorkflowTransitionError(
                        "terminal record contradicts finalization intent"
                    )
                self._verify_record_manifests(terminal)
                if not self._final_record_completed(terminal):
                    self._recorder.finalize_run(
                        self._require_run_handle(), terminal
                    )
                if not self._final_record_completed(terminal):
                    raise WorkflowTransitionError(
                        "terminal finalization could not be completed"
                    )
                self._result = terminal
                self._state = _WorkflowState.FINALIZED

    def _load_workflow_transition(
        self, name: str
    ) -> tuple[dict[str, Any], _TransitionSnapshot] | None:
        event_type = f"workflow_{name}"
        events = self._recorder.read_events(self._require_run_handle())
        matching = [
            event
            for event in events
            if event.event_type == event_type
        ]
        if len(matching) > 1:
            raise WorkflowTransitionError(
                f"durable workflow transition is duplicated: {name}"
            )
        started, completed = self._workflow_transition_journals(
            events, event_type=event_type, name=name
        )
        if matching:
            event = TransformationEvent.model_validate(matching[0].payload)
        else:
            if not started and not completed:
                return None
            if len(started) != 1 or len(completed) > 1:
                raise WorkflowTransitionError(
                    f"durable workflow transition journal is ambiguous: {name}"
                )
            event = started[0].provenance
        if len(event.outputs) != 1 or len(event.output_hashes) != 1:
            raise WorkflowTransitionError(
                f"durable workflow transition has invalid outputs: {name}"
            )
        artifact_name, output_name = next(iter(event.outputs.items()))
        if artifact_name != output_name:
            raise WorkflowTransitionError(
                f"durable workflow transition output name mismatch: {name}"
            )
        try:
            content = self._recorder.read_artifact(
                self._require_run_handle(), artifact_name
            )
        except FileNotFoundError as error:
            if matching:
                raise WorkflowTransitionError(
                    f"durable workflow transition event exists without its artifact: {name}"
                ) from error
            raise WorkflowTransitionError(
                f"durable workflow transition journal exists without its artifact: {name}"
            ) from error
        digest = hashlib.sha256(content).hexdigest()
        if event.output_hashes.get(artifact_name) != digest:
            raise WorkflowTransitionError(
                f"durable workflow transition digest mismatch: {name}"
            )
        matching_started = [
            journal for journal in started if journal.artifact_name == artifact_name
        ]
        matching_completed = [
            journal for journal in completed if journal.artifact_name == artifact_name
        ]
        if len(matching_started) != 1 or len(matching_completed) > 1:
            raise WorkflowTransitionError(
                f"durable workflow transition lacks one exact artifact intent: {name}"
            )
        journal = matching_started[0]
        if (
            journal.provenance != event
            or journal.artifact_sha256 != digest
            or journal.artifact_byte_count != len(content)
        ):
            raise WorkflowTransitionError(
                f"durable workflow transition journal mismatch: {name}"
            )
        if not matching_completed:
            self._recorder.write_artifact(
                self._require_run_handle(), artifact_name, content, event
            )
        elif matching_completed[0] != journal:
            raise WorkflowTransitionError(
                f"durable workflow transition completion mismatch: {name}"
            )
        if not matching:
            self._recorder.record_transformation(
                self._require_run_handle(), event
            )
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise WorkflowTransitionError(
                f"durable workflow transition JSON is invalid: {name}"
            ) from error
        if not isinstance(payload, dict):
            raise WorkflowTransitionError(
                f"durable workflow transition payload is invalid: {name}"
            )
        return payload, _TransitionSnapshot(
            event=event,
            artifact_name=artifact_name,
            digest=digest,
            content=content,
        )

    @staticmethod
    def _workflow_transition_journals(
        events: list[Any], *, event_type: str, name: str
    ) -> tuple[list[ArtifactWriteJournal], list[ArtifactWriteJournal]]:
        """Return exact artifact journals that claim one workflow transition."""
        journals: dict[str, list[ArtifactWriteJournal]] = {
            "artifact_write_started": [],
            "artifact_write_completed": [],
        }
        for recorded in events:
            if recorded.event_type not in journals:
                continue
            provenance = recorded.payload.get("provenance")
            if not isinstance(provenance, dict) or provenance.get(
                "event_type"
            ) != event_type:
                continue
            try:
                journal = ArtifactWriteJournal.model_validate(recorded.payload)
            except (TypeError, ValueError) as error:
                raise WorkflowTransitionError(
                    f"durable workflow transition journal is invalid: {name}"
                ) from error
            journals[recorded.event_type].append(journal)
        return (
            journals["artifact_write_started"],
            journals["artifact_write_completed"],
        )

    def _finalize(
        self,
        record: RunRecord,
        *,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self._verify_record_manifests(record)
        if self._state is _WorkflowState.FINALIZED:
            if self._result == record:
                return
            raise WorkflowTransitionError("workflow already finalized")
        if self._pending_final_record is None:
            self._pending_final_record = RunRecord.model_validate(
                record.model_dump(mode="json")
            )
        elif self._pending_final_record != record:
            raise WorkflowTransitionError("a different final record is already pending")
        record_payload = record.model_dump(mode="json")
        supplied_payload = dict(extra_payload or {})
        if self._pending_final_payload is None:
            self._pending_final_payload = supplied_payload
        elif self._pending_final_payload != supplied_payload:
            if self._pending_final_record != record:
                raise WorkflowTransitionError(
                    "a different finalization payload is already pending"
                )
            supplied_payload = self._pending_final_payload
        if self._finalization_intent is None:
            finalization_started_at = datetime.now(UTC)
            self._finalization_intent = self._make_transition(
                "finalization_intent",
                {
                    "run_record": record_payload,
                    "failure": supplied_payload,
                },
                input_hashes={
                    "run_record": canonical_sha256(record_payload),
                },
                reason_code=record.reason_code,
                started_at=finalization_started_at,
                finished_at=datetime.now(UTC),
            )
        self._commit_transition(self._finalization_intent)
        if self._final_record_completed(record):
            self._result = record
            self._state = _WorkflowState.FINALIZED
            return
        self._recorder.finalize_run(self._require_run_handle(), record)
        self._result = record
        self._state = _WorkflowState.FINALIZED

    def _verify_record_manifests(self, record: RunRecord) -> None:
        """Require every manifest digest in a record to resolve transitively."""
        self._verify_record_severity(record)
        digests = record.execution_manifest_sha256s
        if not digests:
            return
        bindings = self._execution_manifests
        if [item.get("sha256") for item in bindings] != digests:
            raise WorkflowTransitionError(
                "execution manifest bindings do not match the terminal record"
            )
        for item in bindings:
            try:
                repetition_index = int(item["repetition_index"])
                side = str(item["side"])
                manifest_path = str(item["path"])
                manifest_sha256 = str(item["sha256"])
            except (KeyError, TypeError, ValueError) as error:
                raise WorkflowTransitionError(
                    "execution manifest binding is malformed"
                ) from error
            revision = (
                record.base_revision if side == "base" else record.candidate_revision
            )
            if side not in {"base", "candidate"}:
                raise WorkflowTransitionError(
                    "execution manifest binding has an invalid side"
                )
            observations = (
                self._completed_base
                if side == "base"
                else self._completed_candidate
            )
            if repetition_index > len(observations):
                raise WorkflowTransitionError(
                    "execution manifest lacks its exact runtime observation"
                )
            self._verify_execution_manifest(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                repetition_index=repetition_index,
                side=side,
                revision=revision,
                expected_observation=observations[repetition_index - 1],
            )

    def _verify_record_severity(self, record: RunRecord) -> None:
        """Recalculate from persisted prepared inputs only as an integrity check."""
        evidence = record.differential_evidence
        if evidence is None:
            if record.severity_assessment is not None:
                raise WorkflowTransitionError(
                    "severity exists without differential evidence"
                )
            return
        if self._prepared is None:
            raise WorkflowTransitionError(
                "severity cannot be verified without the prepared CVSS profile"
            )
        try:
            expected = assess_differential_severity(
                evidence,
                self._prepared.cvss_profile,
            )
        except (CvssAssessmentError, ValueError) as error:
            raise WorkflowTransitionError(
                "persisted severity could not be deterministically verified"
            ) from error
        if record.severity_assessment != expected:
            raise WorkflowTransitionError(
                "persisted severity does not match deterministic calculation"
            )

    def _final_record_completed(self, record: RunRecord) -> bool:
        expected_bytes = (canonical_json(record.model_dump(mode="json")) + "\n").encode(
            "utf-8"
        )
        expected_digest = hashlib.sha256(expected_bytes).hexdigest()
        try:
            record_bytes = self._recorder.read_artifact(
                self._require_run_handle(), "run_record.json"
            )
        except FileNotFoundError:
            return False
        if record_bytes != expected_bytes:
            return False
        return any(
            event.event_type == LifecycleEventType.FINALIZATION_COMPLETED.value
            and event.payload.get("record_sha256") == expected_digest
            for event in self._recorder.read_events(self._require_run_handle())
        )

    def _record_transition(
        self,
        transition: _WorkflowState,
        payload: Mapping[str, Any],
        *,
        input_hashes: Mapping[str, str],
        reason_code: str,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        snapshot = self._make_transition(
            transition.value,
            payload,
            input_hashes=input_hashes,
            reason_code=reason_code,
            started_at=started_at,
            finished_at=finished_at,
        )
        self._commit_transition(snapshot)

    def _make_transition(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        input_hashes: Mapping[str, str],
        reason_code: str,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> _TransitionSnapshot:
        """Freeze exact recorder bytes and provenance before the first write attempt."""
        content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        artifact_name = f"artifacts/{name}/{digest}.json"
        provenance_started_at = started_at or datetime.now(UTC)
        provenance_finished_at = finished_at or datetime.now(UTC)
        if provenance_finished_at <= provenance_started_at:
            provenance_finished_at = provenance_started_at + timedelta(microseconds=1)
        event = TransformationEvent(
            event_type=f"workflow_{name}",
            inputs={name: name for name in input_hashes},
            outputs={artifact_name: artifact_name},
            input_hashes=dict(input_hashes),
            output_hashes={artifact_name: digest},
            versions={"triageguard": "2.0.0", "workflow": "milestone_one"},
            started_at=provenance_started_at,
            finished_at=provenance_finished_at,
            reason_code=reason_code,
        )
        return _TransitionSnapshot(
            event=event,
            artifact_name=artifact_name,
            digest=digest,
            content=content,
        )

    def _commit_transition(self, snapshot: _TransitionSnapshot) -> None:
        """Reconcile Task 2 journals and commit one frozen logical transition."""
        if not self._transition_artifact_completed(snapshot):
            self._recorder.write_artifact(
                self._require_run_handle(),
                snapshot.artifact_name,
                snapshot.content,
                snapshot.event,
            )
        matching_events = [
            recorded
            for recorded in self._recorder.read_events(self._require_run_handle())
            if recorded.event_type == snapshot.event.event_type
        ]
        if not matching_events:
            self._recorder.record_transformation(
                self._require_run_handle(), snapshot.event
            )
            return
        if len(matching_events) != 1 or (
            matching_events[0].payload
            != snapshot.event.model_dump(mode="json")
        ):
            raise WorkflowTransitionError(
                "workflow transition event conflicts with its immutable artifact"
            )

    def _record_lifecycle_once(self, event: LifecycleEvent) -> None:
        """Append one typed lifecycle event, reconciling commit-then-raise."""
        if any(
            recorded.event_type == event.event_type.value
            and recorded.payload == event.payload
            for recorded in self._recorder.read_events(self._require_run_handle())
        ):
            return
        self._recorder.record_lifecycle_event(self._require_run_handle(), event)

    def _transition_artifact_completed(
        self,
        snapshot: _TransitionSnapshot,
    ) -> bool:
        events = self._recorder.read_events(self._require_run_handle())
        started = [
            recorded
            for recorded in events
            if recorded.event_type == "artifact_write_started"
            and recorded.payload.get("artifact_name") == snapshot.artifact_name
        ]
        completed = [
            recorded
            for recorded in events
            if recorded.event_type == "artifact_write_completed"
            and recorded.payload.get("artifact_name") == snapshot.artifact_name
        ]
        if not completed:
            return False
        if len(started) != 1 or len(completed) != 1:
            raise WorkflowTransitionError(
                "workflow transition artifact journal is duplicated or incomplete"
            )
        try:
            started_journal = ArtifactWriteJournal.model_validate(started[0].payload)
            completed_journal = ArtifactWriteJournal.model_validate(
                completed[0].payload
            )
        except (TypeError, ValueError) as error:
            raise WorkflowTransitionError(
                "workflow transition artifact journal is invalid"
            ) from error
        if (
            started_journal != completed_journal
            or completed_journal.provenance != snapshot.event
            or completed_journal.artifact_sha256 != snapshot.digest
            or completed_journal.artifact_byte_count != len(snapshot.content)
        ):
            raise WorkflowTransitionError(
                "workflow transition artifact journal conflicts with its snapshot"
            )
        try:
            artifact_bytes = self._recorder.read_artifact(
                self._require_run_handle(), snapshot.artifact_name
            )
        except FileNotFoundError:
            artifact_bytes = None
        if artifact_bytes != snapshot.content:
            raise WorkflowTransitionError(
                "completed transition artifact bytes do not match its digest"
            )
        return True

    def _before_model_call(self, request: ModelRequest) -> ModelResponse | None:
        """Commit exact call intent and reconcile an already durable result."""
        operation_id = request.purpose
        intent_name = self._model_operation_name(operation_id, "intent")
        result_name = self._model_operation_name(operation_id, "result")
        intent_payload = {
            "operation_kind": "model",
            "operation_id": operation_id,
            "request": request.model_dump(mode="json"),
            "request_sha256": canonical_sha256(request.model_dump(mode="json")),
        }
        intent_existed = self._load_named_payload(intent_name)
        if intent_existed is None:
            self._commit_named_payload(
                event_name=f"model_{operation_id}_intent",
                artifact_name=intent_name,
                payload=intent_payload,
                input_hashes={
                    "model_request": intent_payload["request_sha256"],
                },
                reason_code="model_operation_intent_committed",
            )
        elif intent_existed != intent_payload:
            raise WorkflowTransitionError(
                f"model operation {operation_id} conflicts with its durable intent"
            )

        result_payload = self._load_named_payload(result_name)
        pending = self._pending_model_results.get(operation_id)
        if result_payload is None and pending is not None:
            self._commit_transition(pending)
            result_payload = self._load_named_payload(result_name)
            self._pending_model_results.pop(operation_id, None)
        if result_payload is None:
            if intent_existed is None:
                return None
            raise InterruptedExternalOperationError("model", operation_id)
        if result_payload.get("request_sha256") != intent_payload["request_sha256"]:
            raise WorkflowTransitionError(
                f"model operation {operation_id} result is bound to another request"
            )
        outcome = result_payload.get("outcome")
        if outcome == "succeeded":
            return ModelResponse.model_validate(result_payload.get("response"))
        if outcome == "failed":
            failure_type = result_payload.get("failure_type")
            message = f"recovered durable model failure: {failure_type}"
            provenance_payload = result_payload.get("failure_provenance")
            provenance = None
            if isinstance(provenance_payload, dict):
                from triageguard.llm.gateway import ModelFailureProvenance

                provenance = ModelFailureProvenance.model_validate(
                    provenance_payload
                )
            error_type: type[ModelGatewayError]
            if failure_type == "ReplayResponseMissing":
                error_type = ReplayResponseMissing
            elif failure_type == "ModelOutputInvalid":
                error_type = ModelOutputInvalid
            else:
                error_type = ModelGatewayError
            raise error_type(message, provenance=provenance)
        raise WorkflowTransitionError(
            f"model operation {operation_id} has an invalid durable outcome"
        )

    def _after_model_call(
        self,
        request: ModelRequest,
        response: ModelResponse | None,
        error: Exception | None,
    ) -> None:
        """Persist the exact known call outcome before returning it to a stage."""
        operation_id = request.purpose
        result_name = self._model_operation_name(operation_id, "result")
        request_digest = canonical_sha256(request.model_dump(mode="json"))
        provenance = getattr(error, "provenance", None)
        payload: dict[str, Any] = {
            "operation_kind": "model",
            "operation_id": operation_id,
            "request_sha256": request_digest,
            "outcome": "succeeded" if error is None else "failed",
            "response": (
                response.model_dump(mode="json") if response is not None else None
            ),
            "failure_type": type(error).__name__ if error is not None else None,
            "failure_provenance": (
                provenance.model_dump(mode="json")
                if provenance is not None
                else None
            ),
        }
        snapshot = self._make_named_transition(
            event_name=f"model_{operation_id}_result",
            artifact_name=result_name,
            payload=payload,
            input_hashes={"model_request": request_digest},
            reason_code=(
                "model_operation_result_committed"
                if error is None
                else "model_operation_failure_committed"
            ),
        )
        self._pending_model_results[operation_id] = snapshot
        try:
            self._commit_transition(snapshot)
        except OSError as commit_error:
            raise OperationJournalInterruptedError(
                f"durable result commit interrupted for model {operation_id}"
            ) from commit_error
        self._pending_model_results.pop(operation_id, None)

    @staticmethod
    def _model_operation_name(operation_id: str, phase: str) -> str:
        return f"artifacts/operations/model/{operation_id}/{phase}.json"

    def _commit_named_payload(
        self,
        *,
        event_name: str,
        artifact_name: str,
        payload: Mapping[str, Any],
        input_hashes: Mapping[str, str],
        reason_code: str,
    ) -> _TransitionSnapshot:
        snapshot = self._make_named_transition(
            event_name=event_name,
            artifact_name=artifact_name,
            payload=payload,
            input_hashes=input_hashes,
            reason_code=reason_code,
        )
        self._commit_transition(snapshot)
        return snapshot

    def _make_named_transition(
        self,
        *,
        event_name: str,
        artifact_name: str,
        payload: Mapping[str, Any],
        input_hashes: Mapping[str, str],
        reason_code: str,
    ) -> _TransitionSnapshot:
        content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        started_at = datetime.now(UTC)
        event = TransformationEvent(
            event_type=f"external_{event_name}",
            inputs={name: name for name in input_hashes},
            outputs={artifact_name: artifact_name},
            input_hashes=dict(input_hashes),
            output_hashes={artifact_name: digest},
            versions={"triageguard": "2.0.0", "workflow": "milestone_one"},
            started_at=started_at,
            finished_at=started_at + timedelta(microseconds=1),
            reason_code=reason_code,
        )
        return _TransitionSnapshot(
            event=event,
            artifact_name=artifact_name,
            digest=digest,
            content=content,
        )

    def _load_named_payload(self, artifact_name: str) -> dict[str, Any] | None:
        events = self._recorder.read_events(self._require_run_handle())
        relevant_events = [
            event
            for event in events
            if isinstance(event.payload.get("outputs"), dict)
            and event.payload["outputs"].get(artifact_name) == artifact_name
        ]
        journal_events = {
            event_type: [
                event
                for event in events
                if event.event_type == event_type
                and event.payload.get("artifact_name") == artifact_name
            ]
            for event_type in (
                "artifact_write_started",
                "artifact_write_completed",
            )
        }
        try:
            content = self._recorder.read_artifact(
                self._require_run_handle(), artifact_name
            )
        except FileNotFoundError:
            if relevant_events or any(journal_events.values()):
                raise WorkflowTransitionError(
                    "durable operation event or journal exists without its artifact: "
                    f"{artifact_name}"
                )
            return None
        digest = hashlib.sha256(content).hexdigest()
        try:
            started_items = journal_events["artifact_write_started"]
            completed_items = journal_events["artifact_write_completed"]
            if len(started_items) != 1 or len(completed_items) > 1:
                raise ValueError("named artifact lacks one exact journal intent")
            started = ArtifactWriteJournal.model_validate(
                started_items[0].payload
            )
            if (
                started.artifact_sha256 != digest
                or started.artifact_byte_count != len(content)
            ):
                raise ValueError("named artifact bytes disagree with its journal")
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
            raise WorkflowTransitionError(
                "durable operation artifact integrity check failed: "
                f"{artifact_name}"
            ) from error

        # Artifact bytes and the frozen journal provenance are sufficient to
        # finish an interrupted artifact-first commit without rerunning its producer.
        self._commit_transition(
            _TransitionSnapshot(
                event=started.provenance,
                artifact_name=artifact_name,
                digest=digest,
                content=content,
            )
        )
        events = self._recorder.read_events(self._require_run_handle())
        try:
            started_items = [
                event
                for event in events
                if event.event_type == "artifact_write_started"
                and event.payload.get("artifact_name") == artifact_name
            ]
            completed_items = [
                event
                for event in events
                if event.event_type == "artifact_write_completed"
                and event.payload.get("artifact_name") == artifact_name
            ]
            if len(started_items) != 1 or len(completed_items) != 1:
                raise ValueError("named artifact lacks one exact journal pair")
            completed = ArtifactWriteJournal.model_validate(
                completed_items[0].payload
            )
            if started != completed:
                raise ValueError("named artifact journal entries disagree")
            exact_provenance = started.provenance.model_dump(mode="json")
            exact_events = [
                event
                for event in events
                if isinstance(event.payload.get("outputs"), dict)
                and event.payload["outputs"].get(artifact_name) == artifact_name
            ]
            if (
                len(exact_events) != 1
                or exact_events[0].event_type != started.provenance.event_type
                or exact_events[0].payload != exact_provenance
            ):
                raise ValueError(
                    "named artifact lacks one exact transformation event"
                )
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
            raise WorkflowTransitionError(
                "durable operation artifact integrity check failed: "
                f"{artifact_name}"
            ) from error
        if not isinstance(payload, dict):
            raise WorkflowTransitionError(
                f"durable operation artifact must be an object: {artifact_name}"
            )
        return payload

    def _model_calls(self) -> list[dict[str, Any]]:
        return [call.as_dict() for call in self._gateway.calls]

    @staticmethod
    def _generation_failure(
        error: Exception,
    ) -> tuple[WorkflowStatus, str, str]:
        if isinstance(error, InterruptedExternalOperationError):
            return (
                WorkflowStatus.GENERATION_ABSTAINED,
                "model_operation_interrupted_unknown_outcome",
                (
                    "A durable model-call intent had no durable result after process "
                    "interruption; the unknown call was not repeated."
                ),
            )
        if isinstance(error, ReplayResponseMissing):
            return (
                WorkflowStatus.GENERATION_ABSTAINED,
                "replay_response_missing",
                "A required prerecorded model response was unavailable; no fallback was used.",
            )
        if isinstance(error, PlanValidationError):
            return (
                WorkflowStatus.GENERATION_ABSTAINED,
                "invalid_model_plan",
                "The model plan failed deterministic contract and primitive validation.",
            )
        if isinstance(error, ModelGatewayError):
            return (
                WorkflowStatus.GENERATION_ABSTAINED,
                "model_generation_failed",
                "The bounded model operation failed; no fallback was used.",
            )
        return (
            WorkflowStatus.GENERATION_ABSTAINED,
            "generation_failed",
            "The generation stage failed before executable evidence existed.",
        )

    @staticmethod
    def _execution_failure(
        error: Exception,
    ) -> tuple[WorkflowStatus, str, str]:
        if isinstance(error, InvalidClassifierResultError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "invalid_classifier_result",
                "The classifier returned malformed or incoherent differential evidence.",
            )
        if isinstance(error, CvssAssessmentError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "cvss_assessment_failed",
                "Severity calculation failed; no vulnerability score was recorded.",
            )
        if isinstance(error, ExecutionManifestError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "execution_manifest_invalid",
                "A completed experiment lacked a complete immutable file manifest.",
            )
        if isinstance(error, InterruptedExternalOperationError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "experiment_interrupted_unknown_outcome",
                (
                    "A durable experiment intent had no durable result after process "
                    "interruption; the unknown experiment was not repeated."
                ),
            )
        if isinstance(error, ExecutionTimeoutError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "generated_test_timeout",
                "The approved generated test exceeded its bounded timeout.",
            )
        if isinstance(error, MissingObservationError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "missing_runtime_observation",
                "Execution ended without all five required raw observation facts.",
            )
        if isinstance(error, InvalidObservationError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                "invalid_runtime_observation",
                "Execution produced malformed or duplicate raw observation facts.",
            )
        if isinstance(error, UnexpectedPytestOutcomeError):
            return (
                WorkflowStatus.EXECUTION_INCONCLUSIVE,
                error.reason_code,
                "The isolated pytest outcome could not support security evidence.",
            )
        return (
            WorkflowStatus.EXECUTION_INCONCLUSIVE,
            "execution_failed",
            "Differential execution failed before all repetitions completed.",
        )

    def _require_state(self, expected: _WorkflowState, operation: str) -> None:
        if self._state is not expected:
            raise WorkflowTransitionError(
                f"{operation} requires {expected.value}; current state is {self._state.value}"
            )

    def _require_run_directory(self) -> Path:
        if self._run_directory is None:
            raise WorkflowTransitionError("run directory is unavailable before prepare")
        return self._run_directory

    def _require_run_handle(self) -> RunHandle:
        if self._run_handle is None:
            raise WorkflowTransitionError("run handle is unavailable before prepare")
        return self._run_handle

    def _require_started_at(self) -> datetime:
        if self._started_at is None:
            raise WorkflowTransitionError("run start time is unavailable before prepare")
        return self._started_at

    @staticmethod
    def _validate_revision(revision: str) -> None:
        RuntimeObservation(
            revision=revision,
            setup_succeeded=False,
            action_attempted=False,
            control_succeeded=None,
            control_request_status=None,
            control_resource_exists_before=None,
            control_resource_exists_after=None,
            request_status=None,
            resource_exists_after=None,
            pytest_exit_code=1,
            reason_code="revision_validation",
        )


def build_replay_workflow(
    *,
    artifact_root: str | Path,
    fixture_directory: str | Path,
    run_id: str | None = None,
    gateway: StructuredModelGateway | None = None,
    settings: Settings | None = None,
    **workflow_dependencies: Any,
) -> MilestoneOneWorkflow:
    """Build the real local replay slice without credentials or fallback output."""
    fixture_path = Path(fixture_directory)
    configured = settings or Settings(
        llm_mode="replay",
        artifacts_dir=Path(artifact_root),
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
    )
    if configured.llm_mode != "replay":
        raise ValueError("build_replay_workflow requires replay settings")
    if gateway is not None and type(gateway) is not ReplayGateway:
        raise ValueError("build_replay_workflow requires an exact ReplayGateway")
    if gateway is None:
        gateway = ReplayGateway(
            {
                "test_plan": _load_json(fixture_path / "planner_response.json"),
                "pytest_generation": _load_json(
                    fixture_path / "generator_response.json"
                ),
            }
        )
    return MilestoneOneWorkflow(
        fixture_directory=fixture_path,
        settings=configured,
        gateway=gateway,
        recorder=ArtifactRecorder(artifact_root),
        run_id=run_id,
        **workflow_dependencies,
    )


def resume_replay_workflow(
    *,
    artifact_root: str | Path,
    fixture_directory: str | Path,
    handle: RunHandle,
    gateway: StructuredModelGateway | None = None,
    settings: Settings | None = None,
    **workflow_dependencies: Any,
) -> MilestoneOneWorkflow:
    """Resume one authenticated local replay run from verified durable snapshots."""
    if not isinstance(handle, RunHandle):
        raise TypeError("resume_replay_workflow requires a RunHandle")
    workflow = build_replay_workflow(
        artifact_root=artifact_root,
        fixture_directory=fixture_directory,
        run_id=handle.run_id,
        gateway=gateway,
        settings=settings,
        **workflow_dependencies,
    )
    workflow._restore_run(handle)
    return workflow


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"fixture must contain a JSON object: {path.name}")
    return payload


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
