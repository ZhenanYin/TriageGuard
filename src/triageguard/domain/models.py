"""Immutable domain artifacts for the V2 security-research workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from triageguard.domain.statuses import EnvironmentKind, WorkflowStatus
from triageguard.provenance import canonical_sha256

_REVISION_PATTERN = (
    r"^(?:base|candidate|[a-z][a-z0-9]*(?:-[a-z0-9]+)+|[0-9a-f]{7,64})$"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CVSS4_BASE_METRICS = (
    "AV",
    "AC",
    "AT",
    "PR",
    "UI",
    "VC",
    "VI",
    "VA",
    "SC",
    "SI",
    "SA",
)
_EXECUTION_FILENAMES = {
    "feature": "authorization.feature",
    "generated_test": "test_authorization.py",
    "pytest_config": "pytest.ini",
    "raw_event_sidecar": "observation.events.jsonl",
    "final_observation": "observation.json",
    "structured_pytest_outcome": "pytest-outcome.json",
    "stdout": "pytest.stdout.txt",
    "stderr": "pytest.stderr.txt",
}


class ResearchArtifact(BaseModel):
    """Base class that prevents unreviewed data from entering a run record."""

    model_config = ConfigDict(extra="forbid", frozen=True)


CvssSourceCategory = Literal[
    "contract",
    "runtime_design",
    "deployment_assumption",
    "expert_judgment",
    "standard_interpretation",
]


class CvssMetricEvidence(ResearchArtifact):
    """Human-reviewable provenance for one CVSS v4.0 Base metric."""

    metric: StrictStr = Field(min_length=1)
    value: StrictStr = Field(min_length=1)
    rationale: StrictStr = Field(min_length=1)
    source_category: CvssSourceCategory
    source_references: list[StrictStr] = Field(min_length=1)


class CvssProfile(ResearchArtifact):
    """Complete expert-authored metric profile without a numeric score."""

    profile_id: StrictStr = Field(min_length=1)
    cvss_version: Literal["4.0"]
    vector: StrictStr = Field(min_length=1)
    metrics: list[CvssMetricEvidence] = Field(min_length=1)
    assessment_label: Literal["expert_authored_provisional"]

    @model_validator(mode="after")
    def validate_complete_base_profile(self) -> CvssProfile:
        metric_codes = tuple(item.metric for item in self.metrics)
        if metric_codes != _CVSS4_BASE_METRICS:
            raise ValueError(
                "profile must contain exactly the CVSS v4.0 Base metrics in order"
            )

        segments = self.vector.split("/")
        if not segments or segments[0] != "CVSS:4.0":
            raise ValueError("vector must use the CVSS:4.0 prefix")
        vector_items: list[tuple[str, str]] = []
        for segment in segments[1:]:
            code, separator, value = segment.partition(":")
            if not separator or not code or not value:
                raise ValueError("vector contains an invalid metric segment")
            if code not in _CVSS4_BASE_METRICS:
                raise ValueError("profile supports CVSS v4.0 Base metrics only")
            vector_items.append((code, value))
        if tuple(code for code, _ in vector_items) != _CVSS4_BASE_METRICS:
            raise ValueError(
                "vector must contain exactly the CVSS v4.0 Base metrics in order"
            )
        if tuple(vector_items) != tuple(
            (item.metric, item.value) for item in self.metrics
        ):
            raise ValueError("vector values must match the metric evidence records")
        return self


class VersionSeverityAssessment(ResearchArtifact):
    """Evidence-bound provisional score or explicit decision not to score."""

    revision: StrictStr = Field(max_length=128, pattern=_REVISION_PATTERN)
    status: Literal["provisional", "not_scored"]
    reason_code: StrictStr = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    profile_id: StrictStr | None = Field(default=None, min_length=1)
    profile_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    evidence_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    vector: StrictStr | None = Field(default=None, min_length=1)
    score: StrictFloat | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        allow_inf_nan=False,
    )
    severity: StrictStr | None = Field(default=None, min_length=1)
    metrics: list[CvssMetricEvidence] = Field(default_factory=list)
    calculator: StrictStr | None = Field(default=None, min_length=1)
    review_status: Literal[
        "expert_authored_provisional",
        "not_applicable",
    ]

    @model_validator(mode="after")
    def validate_scoring_claim(self) -> VersionSeverityAssessment:
        scoring_values = (
            self.profile_id,
            self.profile_sha256,
            self.vector,
            self.score,
            self.severity,
            self.calculator,
        )
        if self.status == "provisional":
            if any(value is None for value in scoring_values):
                raise ValueError(
                    "provisional assessment requires complete scoring provenance"
                )
            if self.reason_code != "tested_vulnerability_observed":
                raise ValueError(
                    "provisional assessment requires the vulnerability-observed reason"
                )
            if self.review_status != "expert_authored_provisional":
                raise ValueError(
                    "provisional assessment requires provisional expert review"
                )
            if tuple(item.metric for item in self.metrics) != _CVSS4_BASE_METRICS:
                raise ValueError(
                    "provisional assessment requires all CVSS v4.0 Base metrics"
                )
        else:
            if any(value is not None for value in scoring_values) or self.metrics:
                raise ValueError(
                    "not-scored assessment forbids score and profile claims"
                )
            if self.reason_code not in {
                "tested_vulnerability_not_observed",
                "insufficient_evidence_for_severity",
            }:
                raise ValueError("not-scored assessment has an unsupported reason")
            if self.review_status != "not_applicable":
                raise ValueError(
                    "not-scored assessment requires not-applicable review status"
                )
        return self


class DifferentialSeverityAssessment(ResearchArtifact):
    """Separate severity decisions for the compared base and candidate."""

    base: VersionSeverityAssessment
    candidate: VersionSeverityAssessment

    @model_validator(mode="after")
    def validate_distinct_revisions(self) -> DifferentialSeverityAssessment:
        if self.base.revision == self.candidate.revision:
            raise ValueError("base and candidate severity revisions must be distinct")
        return self


class RiskContract(ResearchArtifact):
    contract_id: str
    actor: str
    actor_privileges: list[str]
    missing_privileges: list[str]
    preconditions: list[str]
    action: str
    secure_expectation: str
    observable_evidence: list[str] = Field(min_length=1)
    base_expectation: str
    candidate_expectation: str
    cleanup: list[str]


class TestOperation(ResearchArtifact):
    primitive: str
    inputs: dict[str, str]
    captures: list[str] = Field(default_factory=list)


class TestAssertion(ResearchArtifact):
    primitive: str
    observed_field: str
    expected_value: str | int | bool


class TestControl(ResearchArtifact):
    name: str
    operations: list[TestOperation]
    assertions: list[TestAssertion]


class TestPlan(ResearchArtifact):
    __test__ = False

    plan_id: str
    contract_id: str
    givens: list[TestOperation]
    action: TestOperation
    post_action: list[TestOperation]
    assertions: list[TestAssertion] = Field(min_length=1)
    controls: list[TestControl]
    cleanup: list[TestOperation]


class RuntimeObservation(ResearchArtifact):
    """Raw execution facts, deliberately separate from final classification."""

    # Canonical controlled labels are lowercase hyphenated tokens. Lowercase
    # 7-64 character hexadecimal strings also support abbreviated/full Git SHAs.
    revision: StrictStr = Field(
        max_length=128,
        pattern=_REVISION_PATTERN,
    )
    setup_succeeded: StrictBool
    action_attempted: StrictBool
    control_succeeded: StrictBool | None
    control_request_status: StrictInt | None
    control_resource_exists_before: StrictBool | None
    control_resource_exists_after: StrictBool | None
    request_status: StrictInt | None
    resource_exists_after: StrictBool | None
    pytest_exit_code: StrictInt
    reason_code: StrictStr = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    @property
    def security_behavior(self) -> str | None:
        """Return only behavior directly supported by the observed HTTP facts."""
        if (
            not self.setup_succeeded
            or not self.action_attempted
            or self.control_succeeded is not True
            or self.control_request_status != 204
            or self.control_resource_exists_before is not True
            or self.control_resource_exists_after is not False
        ):
            return None
        if self.request_status == 403 and self.resource_exists_after is True:
            return "secure"
        if (
            self.request_status is not None
            and 200 <= self.request_status < 300
            and self.resource_exists_after is False
        ):
            return "vulnerable"
        return None


class ExecutionFile(ResearchArtifact):
    """One exact recorder-owned file transitively bound by an execution manifest."""

    relative_path: StrictStr = Field(min_length=1, max_length=512)
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    byte_count: StrictInt = Field(ge=0)


class ExecutionManifest(ResearchArtifact):
    """Immutable complete file inventory for one revision repetition."""

    side: Literal["base", "candidate"]
    revision: StrictStr = Field(max_length=128, pattern=_REVISION_PATTERN)
    repetition_index: StrictInt = Field(gt=0)
    started_at: datetime
    finished_at: datetime
    files: dict[str, ExecutionFile]

    @model_validator(mode="after")
    def validate_manifest(self) -> ExecutionManifest:
        if not _is_utc(self.started_at) or not _is_utc(self.finished_at):
            raise ValueError("execution manifest timestamps must be UTC")
        if self.finished_at < self.started_at:
            raise ValueError("execution manifest finished_at precedes started_at")
        if set(self.files) != set(_EXECUTION_FILENAMES):
            raise ValueError("execution manifest must bind every required file exactly")
        prefix = (
            f"artifacts/executions/{self.repetition_index:04d}-{self.side}/files/"
        )
        for kind, filename in _EXECUTION_FILENAMES.items():
            if self.files[kind].relative_path != f"{prefix}{filename}":
                raise ValueError(f"execution manifest path mismatch for {kind}")
        return self


class DifferentialEvidence(ResearchArtifact):
    base: RuntimeObservation
    candidate: RuntimeObservation
    base_revision: StrictStr = Field(max_length=128, pattern=_REVISION_PATTERN)
    candidate_revision: StrictStr = Field(max_length=128, pattern=_REVISION_PATTERN)
    repetitions: StrictInt
    stable: StrictBool
    status: WorkflowStatus
    reason_code: StrictStr = Field(min_length=1)
    explanation: StrictStr = Field(min_length=1)
    base_differing_run_indexes: list[StrictInt]
    candidate_differing_run_indexes: list[StrictInt]
    execution_manifest_sha256s: list[
        StrictStr
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_coherence(self) -> DifferentialEvidence:
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.base_revision != self.base.revision:
            raise ValueError("base_revision must match the base observation")
        if self.candidate_revision != self.candidate.revision:
            raise ValueError("candidate_revision must match the candidate observation")
        if self.base_revision == self.candidate_revision:
            raise ValueError("base and candidate revisions must be distinct")
        _validate_manifest_digests(
            self.execution_manifest_sha256s,
            expected_count=(2 * self.repetitions if self.execution_manifest_sha256s else 0),
        )
        for label, indexes in (
            ("base", self.base_differing_run_indexes),
            ("candidate", self.candidate_differing_run_indexes),
        ):
            if indexes != sorted(set(indexes)):
                raise ValueError(f"{label} differing indexes must be sorted and unique")
            if any(index < 2 or index > self.repetitions for index in indexes):
                raise ValueError(
                    f"{label} differing indexes must identify repetitions after run 1"
                )

        if self.status is WorkflowStatus.EXECUTION_INCONCLUSIVE:
            expected_explanation = _INCONCLUSIVE_CONCLUSIONS.get(self.reason_code)
            if expected_explanation is None:
                raise ValueError("reason_code does not agree with status")
            expected_reason = self.reason_code
        else:
            expected = _DIFFERENTIAL_CONCLUSIONS.get(self.status)
            if expected is None:
                raise ValueError("status cannot represent differential evidence")
            expected_reason, expected_explanation = expected
        if self.reason_code != expected_reason:
            raise ValueError("reason_code does not agree with status")
        if self.explanation != expected_explanation:
            raise ValueError("explanation does not agree with status and reason_code")
        expected_behaviors = _DIFFERENTIAL_BEHAVIORS.get(self.status)
        if expected_behaviors is not None and (
            self.base.security_behavior,
            self.candidate.security_behavior,
        ) != expected_behaviors:
            raise ValueError("status does not agree with the representative observations")
        if self.status is WorkflowStatus.UNSTABLE_RESULT and (
            self.base.security_behavior is None
            or self.candidate.security_behavior is None
        ):
            raise ValueError("unstable evidence requires supported representative facts")

        has_differences = bool(
            self.base_differing_run_indexes
            or self.candidate_differing_run_indexes
        )
        if self.status is WorkflowStatus.UNSTABLE_RESULT:
            if self.stable or not has_differences:
                raise ValueError("unstable evidence requires at least one differing run")
        elif self.status is WorkflowStatus.EXECUTION_INCONCLUSIVE:
            if self.stable or has_differences:
                raise ValueError("inconclusive evidence cannot claim stable differences")
        elif not self.stable or has_differences:
            raise ValueError("stable differential evidence cannot contain differing runs")
        return self


class RunRecord(ResearchArtifact):
    """Final, attributable summary persisted by the append-only recorder."""

    run_id: StrictStr = Field(min_length=1)
    environment_kind: EnvironmentKind
    base_revision: StrictStr = Field(max_length=128, pattern=_REVISION_PATTERN)
    candidate_revision: StrictStr = Field(max_length=128, pattern=_REVISION_PATTERN)
    status: WorkflowStatus
    reason_code: StrictStr = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    explanation: StrictStr = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    differential_evidence: DifferentialEvidence | None = None
    severity_assessment: DifferentialSeverityAssessment | None = None
    execution_manifest_sha256s: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_terminal_coherence(self) -> RunRecord:
        if not _is_utc(self.started_at):
            raise ValueError("started_at must be timezone-aware UTC")
        if self.base_revision == self.candidate_revision:
            raise ValueError("base and candidate revisions must be distinct")
        _validate_manifest_digests(self.execution_manifest_sha256s)
        if not _is_utc(self.finished_at):
            raise ValueError("finished_at must be timezone-aware UTC")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        evidence = self.differential_evidence
        classifier_conclusion = self.status in _DIFFERENTIAL_CONCLUSIONS or (
            self.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
            and self.reason_code in _INCONCLUSIVE_CONCLUSIONS
        )
        if evidence is None and classifier_conclusion:
            raise ValueError(
                "a differential terminal status requires differential evidence"
            )
        severity = self.severity_assessment
        if evidence is None and severity is not None:
            raise ValueError(
                "severity cannot be recorded without differential evidence"
            )
        if evidence is not None and severity is None:
            raise ValueError(
                "differential evidence requires severity assessment"
            )
        if evidence is not None:
            if (
                self.status is not evidence.status
                or self.reason_code != evidence.reason_code
                or self.explanation != evidence.explanation
            ):
                raise ValueError(
                    "RunRecord status, reason, and explanation must match its evidence"
                )
            if (
                self.base_revision != evidence.base_revision
                or self.candidate_revision != evidence.candidate_revision
            ):
                raise ValueError("RunRecord revisions must match its evidence")
            if self.execution_manifest_sha256s != (
                evidence.execution_manifest_sha256s
            ):
                raise ValueError("RunRecord manifest digests must match its evidence")
            if len(self.execution_manifest_sha256s) != 2 * evidence.repetitions:
                raise ValueError("terminal evidence must bind every execution manifest")
            if severity is not None:
                if (
                    severity.base.revision != evidence.base_revision
                    or severity.candidate.revision != evidence.candidate_revision
                ):
                    raise ValueError(
                        "severity revisions must match differential evidence"
                    )
                insufficient = evidence.status in {
                    WorkflowStatus.UNSTABLE_RESULT,
                    WorkflowStatus.EXECUTION_INCONCLUSIVE,
                }
                _validate_version_severity(
                    severity.base,
                    evidence.base,
                    insufficient=insufficient,
                )
                _validate_version_severity(
                    severity.candidate,
                    evidence.candidate,
                    insufficient=insufficient,
                )
        return self

    @property
    def stable(self) -> bool:
        """Expose repeatability without duplicating persisted evidence fields."""
        return (
            self.differential_evidence.stable
            if self.differential_evidence is not None
            else False
        )


_DIFFERENTIAL_CONCLUSIONS: dict[WorkflowStatus, tuple[str, str]] = {
    WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED: (
        "candidate_regression_observed",
        (
            "The base denied unauthorized deletion and preserved the patient, "
            "while the candidate allowed deletion and removed the patient."
        ),
    ),
    WorkflowStatus.CANDIDATE_FIX_OBSERVED: (
        "candidate_fix_observed",
        (
            "The base allowed unauthorized deletion and removed the patient, "
            "while the candidate denied deletion and preserved the patient."
        ),
    ),
    WorkflowStatus.NO_REGRESSION_OBSERVED: (
        "no_regression_observed",
        "Both revisions denied unauthorized deletion and preserved the patient.",
    ),
    WorkflowStatus.PRE_EXISTING_RISK_OBSERVED: (
        "pre_existing_risk_observed",
        "Both revisions allowed unauthorized deletion and removed the patient.",
    ),
    WorkflowStatus.UNSTABLE_RESULT: (
        "security_relevant_tuple_unstable",
        (
            "Repeated security-relevant facts differed from run 1; one-based "
            "differing run indexes are reported separately for base and candidate."
        ),
    ),
}


_DIFFERENTIAL_BEHAVIORS = {
    WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED: ("secure", "vulnerable"),
    WorkflowStatus.CANDIDATE_FIX_OBSERVED: ("vulnerable", "secure"),
    WorkflowStatus.NO_REGRESSION_OBSERVED: ("secure", "secure"),
    WorkflowStatus.PRE_EXISTING_RISK_OBSERVED: ("vulnerable", "vulnerable"),
}


_INCONCLUSIVE_CONCLUSIONS = {
    "base_setup_failed": (
        "At least one base run did not complete setup; differential evidence is "
        "inconclusive."
    ),
    "candidate_setup_failed": (
        "At least one candidate run did not complete setup; differential evidence "
        "is inconclusive."
    ),
    "base_action_not_attempted": (
        "At least one base run did not attempt the approved action; differential "
        "evidence is inconclusive."
    ),
    "candidate_action_not_attempted": (
        "At least one candidate run did not attempt the approved action; "
        "differential evidence is inconclusive."
    ),
    "base_control_failed_or_missing": (
        "At least one base run lacks a successful authorized control; differential "
        "evidence is inconclusive."
    ),
    "candidate_control_failed_or_missing": (
        "At least one candidate run lacks a successful authorized control; "
        "differential evidence is inconclusive."
    ),
    "base_raw_facts_missing_or_unsupported": (
        "At least one base run has missing or unsupported HTTP/state facts; "
        "differential evidence is inconclusive."
    ),
    "candidate_raw_facts_missing_or_unsupported": (
        "At least one candidate run has missing or unsupported HTTP/state facts; "
        "differential evidence is inconclusive."
    ),
}


def _is_utc(value: datetime) -> bool:
    offset = value.utcoffset()
    return value.tzinfo is not None and offset is not None and offset.total_seconds() == 0


def _validate_manifest_digests(
    digests: list[str], *, expected_count: int | None = None
) -> None:
    if expected_count is not None and len(digests) != expected_count:
        raise ValueError("execution manifest digest count is incoherent")
    if len(digests) != len(set(digests)):
        raise ValueError("execution manifest digests must be unique")
    if any(
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        raise ValueError("execution manifest digests must be canonical SHA-256 values")


def _validate_version_severity(
    assessment: VersionSeverityAssessment,
    observation: RuntimeObservation,
    *,
    insufficient: bool,
) -> None:
    expected_evidence_sha256 = canonical_sha256(
        observation.model_dump(mode="json")
    )
    if assessment.evidence_sha256 != expected_evidence_sha256:
        raise ValueError("severity observation hash does not match runtime evidence")
    if insufficient:
        if (
            assessment.status != "not_scored"
            or assessment.reason_code != "insufficient_evidence_for_severity"
        ):
            raise ValueError("incomplete evidence must remain not scored")
        return
    if observation.security_behavior == "secure":
        if (
            assessment.status != "not_scored"
            or assessment.reason_code != "tested_vulnerability_not_observed"
        ):
            raise ValueError("secure runtime evidence must remain not scored")
        return
    if observation.security_behavior != "vulnerable":
        raise ValueError("unsupported runtime evidence cannot receive severity")
    if (
        assessment.status != "provisional"
        or assessment.reason_code != "tested_vulnerability_observed"
    ):
        raise ValueError("vulnerable runtime evidence requires provisional severity")
    profile = CvssProfile(
        profile_id=assessment.profile_id,
        cvss_version="4.0",
        vector=assessment.vector,
        metrics=assessment.metrics,
        assessment_label="expert_authored_provisional",
    )
    if assessment.profile_sha256 != canonical_sha256(
        profile.model_dump(mode="json")
    ):
        raise ValueError("severity profile hash does not match its metric profile")
