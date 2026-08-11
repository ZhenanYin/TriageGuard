"""Immutable, typed artifacts used for real pull-request security analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from triageguard.domain.models import ResearchArtifact
from triageguard.domain.statuses import MilestoneTwoStatus

FullCommitSha = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
ReasonCode = Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_]*$")]


def _is_utc(value: datetime) -> bool:
    offset = value.utcoffset()
    return value.tzinfo is not None and offset is not None and offset.total_seconds() == 0


class PullRequestSnapshot(ResearchArtifact):
    """The exact GitHub and local-Git identity frozen before analysis begins."""

    snapshot_key: Sha256
    repository: Literal["openmrs/openmrs-core"]
    pull_number: StrictInt = Field(gt=0)
    pull_url: StrictStr
    state: Literal["open"]
    default_branch: StrictStr = Field(min_length=1)
    base_branch: StrictStr = Field(min_length=1)
    merge_base_sha: FullCommitSha
    base_sha: FullCommitSha
    head_sha: FullCommitSha
    candidate_sha: FullCommitSha
    merge_base_tree_sha: FullCommitSha
    base_tree_sha: FullCommitSha
    head_tree_sha: FullCommitSha
    candidate_tree_sha: FullCommitSha
    acquired_at: datetime
    github_api_version: StrictStr = Field(min_length=1)
    git_version: StrictStr = Field(min_length=1)
    analysis_config_sha256: Sha256

    @field_validator(
        "merge_base_sha",
        "base_sha",
        "head_sha",
        "candidate_sha",
        "merge_base_tree_sha",
        "base_tree_sha",
        "head_tree_sha",
        "candidate_tree_sha",
        mode="before",
    )
    @classmethod
    def validate_full_commit_sha(cls, value: object) -> object:
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("must be a full 40-character lowercase commit SHA")
        return value

    @model_validator(mode="after")
    def validate_snapshot_coherence(self) -> PullRequestSnapshot:
        if not _is_utc(self.acquired_at):
            raise ValueError("acquired_at must be timezone-aware UTC")
        revisions = (
            self.merge_base_sha,
            self.base_sha,
            self.head_sha,
            self.candidate_sha,
        )
        if len(set(revisions)) != len(revisions):
            raise ValueError("merge-base, base, head, and candidate revisions must be distinct")
        return self


class SnapshotFreshness(ResearchArtifact):
    """A non-mutating recheck of a frozen snapshot's currentness."""

    status: Literal["current", "stale", "unknown"]
    reason_code: ReasonCode
    checked_at: datetime
    observed_base_sha: FullCommitSha | None
    observed_head_sha: FullCommitSha | None
    observed_candidate_sha: FullCommitSha | None

    @model_validator(mode="after")
    def validate_freshness(self) -> SnapshotFreshness:
        if not _is_utc(self.checked_at):
            raise ValueError("checked_at must be timezone-aware UTC")
        observed = (
            self.observed_base_sha,
            self.observed_head_sha,
            self.observed_candidate_sha,
        )
        if self.status == "current" and any(value is None for value in observed):
            raise ValueError("current freshness requires every observed revision")
        if self.status == "unknown" and any(value is not None for value in observed):
            raise ValueError("unknown freshness cannot claim observed revisions")
        return self


class DiffHunk(ResearchArtifact):
    """One exact line range within a locally computed Git diff."""

    old_start: StrictInt = Field(ge=0)
    old_count: StrictInt = Field(ge=0)
    new_start: StrictInt = Field(ge=0)
    new_count: StrictInt = Field(ge=0)


class DiffFile(ResearchArtifact):
    """File-level metadata and hunk inventory for a diff artifact."""

    status: Literal["added", "modified", "deleted", "renamed", "copied", "type_changed"]
    old_path: StrictStr | None
    new_path: StrictStr | None
    binary: StrictBool
    additions: StrictInt = Field(ge=0)
    deletions: StrictInt = Field(ge=0)
    hunks: list[DiffHunk] = Field(default_factory=list)
    content_sha256: Sha256 | None

    @model_validator(mode="after")
    def validate_file_paths(self) -> DiffFile:
        if self.status == "added" and self.old_path is not None:
            raise ValueError("an added file cannot have an old path")
        if self.status == "deleted" and self.new_path is not None:
            raise ValueError("a deleted file cannot have a new path")
        if self.status not in {"added", "deleted"} and (
            self.old_path is None or self.new_path is None
        ):
            raise ValueError("a changed file requires old and new paths")
        if self.binary and self.hunks:
            raise ValueError("a binary file cannot contain text hunks")
        return self


class DiffArtifact(ResearchArtifact):
    """One reproducible locally generated author, integration, or drift diff."""

    kind: Literal["author_diff", "integration_diff", "base_drift_diff"]
    old_revision: FullCommitSha
    new_revision: FullCommitSha
    git_arguments: list[StrictStr] = Field(min_length=1)
    git_version: StrictStr = Field(min_length=1)
    files: list[DiffFile]
    patch_sha256: Sha256
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_diff_revisions(self) -> DiffArtifact:
        if self.old_revision == self.new_revision:
            raise ValueError("diff revisions must be distinct")
        return self


class ContextScoreComponent(ResearchArtifact):
    """A named, inspectable component of deterministic context ranking."""

    name: StrictStr = Field(min_length=1)
    value: StrictFloat


class ContextAnchor(ResearchArtifact):
    """An exact, immutable excerpt that a model may cite by ID only."""

    anchor_id: StrictStr = Field(min_length=1)
    revision_role: Literal["merge_base", "base", "head", "candidate"]
    commit_sha: FullCommitSha
    path: StrictStr = Field(min_length=1)
    java_symbol: StrictStr | None
    start_line: StrictInt = Field(gt=0)
    end_line: StrictInt = Field(gt=0)
    text_sha256: Sha256
    selection_reason: StrictStr = Field(min_length=1)
    score_components: list[ContextScoreComponent]
    change_relation: Literal[
        "author_change", "integration_change", "base_drift_change", "repository_context"
    ]
    truncated: StrictBool

    @model_validator(mode="after")
    def validate_line_range(self) -> ContextAnchor:
        if self.end_line < self.start_line:
            raise ValueError("anchor end_line must not precede start_line")
        names = [component.name for component in self.score_components]
        if len(names) != len(set(names)):
            raise ValueError("anchor score component names must be unique")
        return self


class ContextBundle(ResearchArtifact):
    """The bounded evidence catalog supplied to a single model request."""

    snapshot_key: Sha256
    anchors: list[ContextAnchor]
    selected_file_count: StrictInt = Field(ge=0)
    selected_anchor_count: StrictInt = Field(ge=0)
    selected_bytes: StrictInt = Field(ge=0)
    max_files: StrictInt = Field(gt=0)
    max_anchors: StrictInt = Field(gt=0)
    max_bytes: StrictInt = Field(gt=0)
    excluded_paths: list[StrictStr] = Field(default_factory=list)
    binary_paths: list[StrictStr] = Field(default_factory=list)
    truncated_anchor_ids: list[StrictStr] = Field(default_factory=list)
    primary_change_represented: StrictBool
    context_sha256: Sha256

    @model_validator(mode="after")
    def validate_bundle_coherence(self) -> ContextBundle:
        anchor_ids = [anchor.anchor_id for anchor in self.anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("context anchor IDs must be unique")
        if self.selected_anchor_count != len(self.anchors):
            raise ValueError("selected anchor count must match anchors")
        if self.selected_file_count != len({anchor.path for anchor in self.anchors}):
            raise ValueError("selected file count must match anchors")
        if self.selected_file_count > self.max_files or self.selected_anchor_count > self.max_anchors:
            raise ValueError("context selection exceeds its configured limits")
        if self.selected_bytes > self.max_bytes:
            raise ValueError("context selection exceeds its byte limit")
        if set(self.truncated_anchor_ids) - set(anchor_ids):
            raise ValueError("truncated anchor IDs must be present in the bundle")
        return self


class ClaimEvidenceBinding(ResearchArtifact):
    """Citations supporting one claim field in an unconfirmed risk hypothesis."""

    claim_field: Literal[
        "actor", "action", "expected_secure_behavior", "possible_failure", "observable"
    ]
    observable_index: StrictInt | None
    anchor_ids: list[StrictStr] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_claim_binding(self) -> ClaimEvidenceBinding:
        if self.claim_field == "observable" and self.observable_index is None:
            raise ValueError("an observable binding requires its observable index")
        if self.claim_field != "observable" and self.observable_index is not None:
            raise ValueError("only an observable binding may have an observable index")
        if self.observable_index is not None and self.observable_index < 0:
            raise ValueError("observable index must not be negative")
        if len(self.anchor_ids) != len(set(self.anchor_ids)):
            raise ValueError("citation anchor IDs must be unique within a binding")
        return self


class RiskHypothesisDraft(ResearchArtifact):
    """Model-proposed, explicitly unconfirmed explanation of a possible risk."""

    claim_status: Literal["unconfirmed_risk_hypothesis"]
    title: StrictStr = Field(min_length=1)
    explanation: StrictStr = Field(min_length=1)
    actor: StrictStr = Field(min_length=1)
    preconditions: list[StrictStr]
    action: StrictStr = Field(min_length=1)
    protected_asset: StrictStr = Field(min_length=1)
    security_property: StrictStr = Field(min_length=1)
    expected_secure_behavior: StrictStr = Field(min_length=1)
    possible_failure: StrictStr = Field(min_length=1)
    observables: list[StrictStr] = Field(min_length=1)
    code_identifiers: list[StrictStr]
    evidence_bindings: list[ClaimEvidenceBinding]
    limitations: list[StrictStr] = Field(min_length=1)
    missing_evidence: list[StrictStr]
    priority_rationale: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_bindings(self) -> RiskHypothesisDraft:
        seen: set[tuple[str, int | None]] = set()
        for binding in self.evidence_bindings:
            key = (binding.claim_field, binding.observable_index)
            if key in seen:
                raise ValueError("claim evidence bindings must not duplicate a claim")
            seen.add(key)
        required = {
            ("actor", None),
            ("action", None),
            ("expected_secure_behavior", None),
            ("possible_failure", None),
        }
        required.update(("observable", index) for index in range(len(self.observables)))
        if seen != required:
            raise ValueError("claim evidence bindings must cover every required claim")
        return self


class RiskHypothesis(RiskHypothesisDraft):
    """A validated hypothesis with a deterministic workflow-assigned identity."""

    hypothesis_id: StrictStr = Field(min_length=1)

    @property
    def citation_anchor_ids(self) -> list[str]:
        """Return the stable first-seen union without persisting a duplicate list."""
        return list(
            dict.fromkeys(
                anchor_id
                for binding in self.evidence_bindings
                for anchor_id in binding.anchor_ids
            )
        )


RiskAssessmentOutcome = Literal[
    "risks_proposed", "no_meaningful_security_risk_found", "insufficient_context_to_assess"
]


class RiskAssessmentDraft(ResearchArtifact):
    """One raw structured outcome from the risk-proposal model call."""

    snapshot_key: Sha256
    context_sha256: Sha256
    outcome: RiskAssessmentOutcome
    hypotheses: list[RiskHypothesisDraft] = Field(default_factory=list)
    rationale: StrictStr | None = None
    security_relevant_areas: list[StrictStr] = Field(default_factory=list)
    supporting_anchor_ids: list[StrictStr] = Field(default_factory=list)
    coverage_limitations: list[StrictStr] = Field(default_factory=list)
    reason_code: ReasonCode | None = None
    missing_evidence: list[StrictStr] = Field(default_factory=list)
    needed_evidence: list[StrictStr] = Field(default_factory=list)
    generated_at: datetime

    @model_validator(mode="after")
    def validate_outcome_coherence(self) -> RiskAssessmentDraft:
        if not _is_utc(self.generated_at):
            raise ValueError("generated_at must be timezone-aware UTC")
        if len(self.supporting_anchor_ids) != len(set(self.supporting_anchor_ids)):
            raise ValueError("supporting citation anchor IDs must be unique")
        if self.outcome == "risks_proposed":
            if not 1 <= len(self.hypotheses) <= 5:
                raise ValueError("risks_proposed requires one to five hypotheses")
        elif self.hypotheses:
            raise ValueError("an abstention outcome cannot contain hypotheses")
        if self.outcome == "no_meaningful_security_risk_found" and (
            not self.rationale
            or not self.security_relevant_areas
            or not self.coverage_limitations
        ):
            raise ValueError("a no-risk outcome requires rationale, areas, and limitations")
        if self.outcome == "insufficient_context_to_assess" and (
            not self.reason_code or not self.missing_evidence or not self.needed_evidence
        ):
            raise ValueError("an insufficient-context outcome requires a reason and evidence gap")
        return self


class RiskAssessment(RiskAssessmentDraft):
    """A locally validated risk assessment whose risks have stable IDs."""

    hypotheses: list[RiskHypothesis] = Field(default_factory=list)
    assessment_sha256: Sha256
    validated_at: datetime

    @model_validator(mode="after")
    def validate_assessment_time(self) -> RiskAssessment:
        if not _is_utc(self.validated_at):
            raise ValueError("validated_at must be timezone-aware UTC")
        if self.validated_at < self.generated_at:
            raise ValueError("validated_at must not precede generated_at")
        return self


class ReviewedFieldChange(ResearchArtifact):
    """An explicit before/after record for a reviewer edit."""

    field_name: StrictStr = Field(min_length=1)
    before: StrictStr
    after: StrictStr


class HumanReviewedRisk(ResearchArtifact):
    """A reviewer-approved immutable successor to one model hypothesis."""

    snapshot_key: Sha256
    assessment_sha256: Sha256
    original_hypothesis_sha256: Sha256
    selected_hypothesis_id: StrictStr = Field(min_length=1)
    reviewed_risk: RiskHypothesis
    added_citation_anchor_ids: list[StrictStr] = Field(default_factory=list)
    removed_citation_anchor_ids: list[StrictStr] = Field(default_factory=list)
    field_changes: list[ReviewedFieldChange] = Field(default_factory=list)
    approved_at: datetime

    @model_validator(mode="after")
    def validate_review_coherence(self) -> HumanReviewedRisk:
        if not _is_utc(self.approved_at):
            raise ValueError("approved_at must be timezone-aware UTC")
        if self.selected_hypothesis_id != self.reviewed_risk.hypothesis_id:
            raise ValueError("selected hypothesis ID must match the reviewed risk")
        added = set(self.added_citation_anchor_ids)
        removed = set(self.removed_citation_anchor_ids)
        if len(added) != len(self.added_citation_anchor_ids) or len(removed) != len(self.removed_citation_anchor_ids):
            raise ValueError("review citation changes must be unique")
        if added & removed:
            raise ValueError("a review citation cannot be both added and removed")
        fields = [change.field_name for change in self.field_changes]
        if len(fields) != len(set(fields)):
            raise ValueError("reviewed field changes must be unique")
        return self


class GherkinStep(ResearchArtifact):
    """One ordered structured step matching the rendered Gherkin text."""

    number: StrictInt = Field(gt=0)
    keyword: Literal["Given", "When", "Then", "And"]
    text: StrictStr = Field(min_length=1)


class GherkinStepBinding(ResearchArtifact):
    """Traceability from one approved-risk claim to concrete step numbers."""

    claim_field: Literal[
        "actor", "precondition", "action", "expected_secure_behavior", "possible_failure", "observable"
    ]
    source_index: StrictInt | None
    step_numbers: list[StrictInt] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binding_index(self) -> GherkinStepBinding:
        indexed = self.claim_field in {"precondition", "observable"}
        if indexed != (self.source_index is not None):
            raise ValueError("only precondition and observable bindings require an index")
        if self.source_index is not None and self.source_index < 0:
            raise ValueError("binding source index must not be negative")
        if len(self.step_numbers) != len(set(self.step_numbers)):
            raise ValueError("bound step numbers must be unique")
        return self


class GherkinCandidate(ResearchArtifact):
    """A generated, editable but not yet human-approved scenario candidate."""

    candidate_id: StrictStr = Field(min_length=1)
    snapshot_key: Sha256
    reviewed_risk_sha256: Sha256
    feature_title: StrictStr = Field(min_length=1)
    scenario_title: StrictStr = Field(min_length=1)
    steps: list[GherkinStep] = Field(min_length=1)
    gherkin_text: StrictStr = Field(min_length=1)
    bindings: list[GherkinStepBinding]
    testability_notes: list[StrictStr]
    setup_gaps: list[StrictStr]
    generated_at: datetime

    @model_validator(mode="after")
    def validate_candidate_coherence(self) -> GherkinCandidate:
        if not _is_utc(self.generated_at):
            raise ValueError("generated_at must be timezone-aware UTC")
        if [step.number for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("Gherkin steps must have consecutive numbers starting at one")
        step_numbers = {step.number for step in self.steps}
        if any(number not in step_numbers for binding in self.bindings for number in binding.step_numbers):
            raise ValueError("a Gherkin binding must reference an existing step")
        keys = [(binding.claim_field, binding.source_index) for binding in self.bindings]
        if len(keys) != len(set(keys)):
            raise ValueError("Gherkin bindings must not duplicate a source claim")
        return self


class GherkinApproval(ResearchArtifact):
    """The terminal human approval of a validated scenario candidate."""

    snapshot_key: Sha256
    candidate_id: StrictStr = Field(min_length=1)
    candidate_sha256: Sha256
    reviewed_risk_sha256: Sha256
    approved_at: datetime

    @model_validator(mode="after")
    def validate_approval_time(self) -> GherkinApproval:
        if not _is_utc(self.approved_at):
            raise ValueError("approved_at must be timezone-aware UTC")
        return self


class MilestoneTwoRunRecord(ResearchArtifact):
    """Terminal immutable record for the one-way real-PR analysis workflow."""

    run_id: StrictStr = Field(min_length=1)
    snapshot: PullRequestSnapshot
    status: MilestoneTwoStatus
    reason_code: ReasonCode
    explanation: StrictStr = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    freshness: SnapshotFreshness | None = None
    risk_assessment: RiskAssessment | None = None
    human_reviewed_risk: HumanReviewedRisk | None = None
    gherkin_candidate: GherkinCandidate | None = None
    gherkin_approval: GherkinApproval | None = None

    @model_validator(mode="after")
    def validate_terminal_coherence(self) -> MilestoneTwoRunRecord:
        if not _is_utc(self.started_at) or not _is_utc(self.finished_at):
            raise ValueError("run timestamps must be timezone-aware UTC")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.status is MilestoneTwoStatus.APPROVED_GHERKIN and (
            self.gherkin_approval is None
            or self.gherkin_candidate is None
            or self.human_reviewed_risk is None
        ):
            raise ValueError("approved Gherkin requires a candidate, reviewed risk, and approval")
        if self.status is MilestoneTwoStatus.NO_MEANINGFUL_SECURITY_RISK_FOUND and (
            self.risk_assessment is None
            or self.risk_assessment.outcome != "no_meaningful_security_risk_found"
        ):
            raise ValueError("no-risk terminal status requires a matching assessment")
        if self.status is MilestoneTwoStatus.INSUFFICIENT_CONTEXT_TO_ASSESS and (
            self.risk_assessment is None
            or self.risk_assessment.outcome != "insufficient_context_to_assess"
        ):
            raise ValueError("insufficient-context status requires a matching assessment")
        if self.status is MilestoneTwoStatus.STALE and (
            self.freshness is None or self.freshness.status != "stale"
        ):
            raise ValueError("stale terminal status requires a stale freshness check")
        return self
