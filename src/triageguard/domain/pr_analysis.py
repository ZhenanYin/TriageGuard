"""Immutable, typed artifacts used for real pull-request security analysis."""

from __future__ import annotations

import hashlib
import re
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
from triageguard.provenance import canonical_json, canonical_sha256

FullCommitSha = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
ReasonCode = Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_]*$")]
InsufficientContextReason = Literal[
    "mergeability_unknown",
    "merge_conflict",
    "candidate_ref_missing",
    "candidate_parent_mismatch",
    "snapshot_changed_during_acquisition",
    "github_recheck_unavailable",
    "analysis_limit_exceeded",
    "primary_change_not_represented",
    "insufficient_context_to_assess",
]
_PROHIBITED_MODEL_CLAIM_PATTERNS = (
    r"\bcvss(?:\s*[:v]?\s*\d)?\b",
    r"\bconfirmed\s+(?:vulnerability|vulnerable|security\s+issue)\b",
    r"\b(?:vulnerability|security\s+issue)\s+(?:is|was)\s+confirmed\b",
    r"\bconfirmed\s+(?:safe|safety|secure)\b",
    r"\b(?:safe|secure)\s+(?:is|was)\s+confirmed\b",
)
_FAILED_REASON_CODES = {
    "unsupported_pr_url",
    "unsupported_repository",
    "pr_not_open",
    "non_default_base_branch",
    "mergeability_unknown",
    "merge_conflict",
    "candidate_ref_missing",
    "candidate_parent_mismatch",
    "snapshot_changed_during_acquisition",
    "github_recheck_unavailable",
    "analysis_limit_exceeded",
    "primary_change_not_represented",
    "model_generation_failed",
    "model_output_invalid",
    "risk_grounding_failed",
    "risk_not_approved",
    "gherkin_alignment_failed",
    "gherkin_not_approved",
}
_EDITABLE_RISK_FIELDS = {
    "actor",
    "preconditions",
    "action",
    "protected_asset",
    "expected_secure_behavior",
    "possible_failure",
    "observables",
    "limitations",
}


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
    acquisition_tool_version: StrictStr = Field(min_length=1)
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

    snapshot_key: Sha256
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
        if self.status == "stale" and any(value is None for value in observed):
            raise ValueError("stale freshness requires every observed revision")
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
    hunks: tuple[DiffHunk, ...] = Field(default_factory=tuple)
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
    git_arguments: tuple[StrictStr, ...] = Field(min_length=1)
    git_version: StrictStr = Field(min_length=1)
    files: tuple[DiffFile, ...]
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
    text: StrictStr = Field(min_length=1)
    text_sha256: Sha256
    selection_reason: StrictStr = Field(min_length=1)
    score_components: tuple[ContextScoreComponent, ...]
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
        if self.text_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("anchor text SHA-256 must match its exact text")
        return self


class ContextBundle(ResearchArtifact):
    """The bounded evidence catalog supplied to a single model request."""

    snapshot_key: Sha256
    anchors: tuple[ContextAnchor, ...]
    selected_file_count: StrictInt = Field(ge=0)
    selected_anchor_count: StrictInt = Field(ge=0)
    selected_bytes: StrictInt = Field(ge=0)
    max_files: StrictInt = Field(gt=0)
    max_anchors: StrictInt = Field(gt=0)
    max_bytes: StrictInt = Field(gt=0)
    max_anchor_lines: StrictInt = Field(gt=0)
    max_blob_bytes: StrictInt = Field(gt=0)
    max_search_identifiers: StrictInt = Field(gt=0)
    max_hits_per_identifier: StrictInt = Field(gt=0)
    excluded_paths: tuple[StrictStr, ...] = Field(default_factory=tuple)
    binary_paths: tuple[StrictStr, ...] = Field(default_factory=tuple)
    truncated_anchor_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)
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
        if any(
            anchor.end_line - anchor.start_line + 1 > self.max_anchor_lines
            for anchor in self.anchors
        ):
            raise ValueError("context anchor exceeds max_anchor_lines")
        if set(self.truncated_anchor_ids) != {
            anchor.anchor_id for anchor in self.anchors if anchor.truncated
        }:
            raise ValueError("truncated anchor IDs must exactly match truncated anchors")
        return self


class ClaimEvidenceBinding(ResearchArtifact):
    """Citations supporting one claim field in an unconfirmed risk hypothesis."""

    claim_field: Literal[
        "actor", "action", "expected_secure_behavior", "possible_failure", "observable"
    ]
    observable_index: StrictInt | None
    anchor_ids: tuple[StrictStr, ...] = Field(min_length=1)

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
    preconditions: tuple[StrictStr, ...]
    action: StrictStr = Field(min_length=1)
    protected_asset: StrictStr = Field(min_length=1)
    security_property: StrictStr = Field(min_length=1)
    expected_secure_behavior: StrictStr = Field(min_length=1)
    possible_failure: StrictStr = Field(min_length=1)
    observables: tuple[StrictStr, ...] = Field(min_length=1)
    code_identifiers: tuple[StrictStr, ...]
    evidence_bindings: tuple[ClaimEvidenceBinding, ...]
    limitations: tuple[StrictStr, ...] = Field(min_length=1)
    missing_evidence: tuple[StrictStr, ...]
    priority_rationale: StrictStr = Field(min_length=1)

    @field_validator(
        "title",
        "explanation",
        "actor",
        "preconditions",
        "action",
        "protected_asset",
        "security_property",
        "expected_secure_behavior",
        "possible_failure",
        "observables",
        "code_identifiers",
        "limitations",
        "missing_evidence",
        "priority_rationale",
    )
    @classmethod
    def reject_prohibited_model_claims(cls, value: object) -> object:
        texts = value if isinstance(value, tuple) else (value,)
        if any(
            isinstance(text, str)
            and any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _PROHIBITED_MODEL_CLAIM_PATTERNS)
            for text in texts
        ):
            raise ValueError("model-originated text contains a prohibited claim")
        return value

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

    @property
    def citation_anchor_ids(self) -> list[str]:
        return list(
            dict.fromkeys(
                anchor_id
                for binding in self.evidence_bindings
                for anchor_id in binding.anchor_ids
            )
        )


class RiskHypothesis(RiskHypothesisDraft):
    """A validated hypothesis with a deterministic workflow-assigned identity."""

    hypothesis_id: StrictStr = Field(default="", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def reject_model_supplied_id(cls, value: object) -> object:
        if isinstance(value, dict) and "hypothesis_id" in value:
            raise ValueError("hypothesis_id is locally derived and cannot be model supplied")
        return value

    @model_validator(mode="after")
    def derive_hypothesis_id(self) -> RiskHypothesis:
        content = self.model_dump(mode="json", exclude={"hypothesis_id"})
        object.__setattr__(self, "hypothesis_id", f"risk-{canonical_sha256(content)}")
        return self

    @classmethod
    def from_draft(cls, draft: RiskHypothesisDraft) -> RiskHypothesis:
        """Assign a stable local identity only after validating provider content."""
        return cls.model_validate(draft.model_dump(mode="json"))

    @classmethod
    def from_persisted(cls, value: dict[str, object]) -> RiskHypothesis:
        """Read a durable artifact only when its recorded ID equals the local derivation."""
        supplied_id = value.get("hypothesis_id")
        draft = RiskHypothesisDraft.model_validate(
            {key: item for key, item in value.items() if key != "hypothesis_id"}
        )
        result = cls.from_draft(draft)
        if supplied_id != result.hypothesis_id:
            raise ValueError("persisted hypothesis ID does not match local derivation")
        return result



RiskAssessmentOutcome = Literal[
    "risks_proposed", "no_meaningful_security_risk_found", "insufficient_context_to_assess"
]


class RiskAssessmentDraft(ResearchArtifact):
    """One raw structured outcome from the risk-proposal model call."""

    snapshot_key: Sha256
    context_sha256: Sha256
    outcome: RiskAssessmentOutcome
    hypotheses: tuple[RiskHypothesisDraft, ...] = Field(default_factory=tuple)
    rationale: StrictStr | None = None
    security_relevant_areas: tuple[StrictStr, ...] = Field(default_factory=tuple)
    supporting_anchor_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)
    coverage_limitations: tuple[StrictStr, ...] = Field(default_factory=tuple)
    reason_code: InsufficientContextReason | None = None
    missing_evidence: tuple[StrictStr, ...] = Field(default_factory=tuple)
    needed_evidence: tuple[StrictStr, ...] = Field(default_factory=tuple)
    generated_at: datetime

    @field_validator(
        "rationale",
        "security_relevant_areas",
        "coverage_limitations",
        "missing_evidence",
        "needed_evidence",
    )
    @classmethod
    def reject_prohibited_assessment_claims(cls, value: object) -> object:
        texts = value if isinstance(value, tuple) else (value,)
        if any(
            isinstance(text, str)
            and any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _PROHIBITED_MODEL_CLAIM_PATTERNS)
            for text in texts
        ):
            raise ValueError("model-originated text contains a prohibited claim")
        return value

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
        if self.outcome == "no_meaningful_security_risk_found" and not any(
            "not proof" in limitation.lower() and "safety" in limitation.lower()
            for limitation in self.coverage_limitations
        ):
            raise ValueError("a no-risk outcome must state that it is not proof of safety")
        if self.outcome == "insufficient_context_to_assess" and (
            not self.reason_code or not self.missing_evidence or not self.needed_evidence
        ):
            raise ValueError("an insufficient-context outcome requires a reason and evidence gap")
        return self


class IdentifierEvidence(ResearchArtifact):
    """One locally checked identifier-to-anchor occurrence."""

    identifier: StrictStr = Field(min_length=1)
    anchor_ids: tuple[StrictStr, ...] = Field(min_length=1)


class GroundingReport(ResearchArtifact):
    """Immutable local attestation that a validated risk is grounded in context."""

    producer: Literal["local_grounding_validator"]
    snapshot_key: Sha256
    context_sha256: Sha256
    hypothesis_id: StrictStr = Field(min_length=1)
    hypothesis_sha256: Sha256
    cited_anchor_ids: tuple[StrictStr, ...] = Field(min_length=1)
    identifier_evidence: tuple[IdentifierEvidence, ...]


class RiskAssessment(RiskAssessmentDraft):
    """A locally validated risk assessment whose risks have stable IDs."""

    hypotheses: tuple[RiskHypothesis, ...] = Field(default_factory=tuple)
    assessment_sha256: Sha256
    validated_at: datetime
    context_bundle: ContextBundle
    grounding_reports: tuple[GroundingReport, ...]

    @model_validator(mode="after")
    def validate_assessment_time(self) -> RiskAssessment:
        if not _is_utc(self.validated_at):
            raise ValueError("validated_at must be timezone-aware UTC")
        if self.validated_at < self.generated_at:
            raise ValueError("validated_at must not precede generated_at")
        if self.context_bundle.snapshot_key != self.snapshot_key:
            raise ValueError("context bundle snapshot key must match the assessment")
        if self.context_bundle.context_sha256 != self.context_sha256:
            raise ValueError("context bundle hash must match the assessment")
        if not self.context_bundle.primary_change_represented:
            raise ValueError("validated assessment requires represented primary integration change")
        if self.outcome == "risks_proposed":
            if len(self.grounding_reports) != len(self.hypotheses):
                raise ValueError("every risk hypothesis requires one grounding report")
            anchors = {anchor.anchor_id: anchor for anchor in self.context_bundle.anchors}
            reports = {report.hypothesis_id: report for report in self.grounding_reports}
            if len(reports) != len(self.grounding_reports):
                raise ValueError("grounding reports must have unique hypothesis IDs")
            for hypothesis in self.hypotheses:
                report = reports.get(hypothesis.hypothesis_id)
                if report is None:
                    raise ValueError("risk hypothesis is missing its grounding report")
                if (
                    report.snapshot_key != self.snapshot_key
                    or report.context_sha256 != self.context_sha256
                    or report.hypothesis_sha256 != canonical_sha256(hypothesis.model_dump(mode="json"))
                ):
                    raise ValueError("grounding report does not bind the exact assessment inputs")
                citations = tuple(hypothesis.citation_anchor_ids)
                if report.cited_anchor_ids != citations or any(anchor_id not in anchors for anchor_id in citations):
                    raise ValueError("grounding report citations must resolve to the frozen context catalog")
                if not any(anchors[anchor_id].change_relation == "integration_change" for anchor_id in citations):
                    raise ValueError("risk hypothesis requires an integration-change citation")
                bindings = {binding.identifier: binding for binding in report.identifier_evidence}
                if set(bindings) != set(hypothesis.code_identifiers):
                    raise ValueError("grounding report must cover every declared code identifier")
                bound_ids = set(citations)
                for identifier, evidence in bindings.items():
                    if any(anchor_id not in bound_ids for anchor_id in evidence.anchor_ids) or not any(
                        identifier in anchors[anchor_id].text for anchor_id in evidence.anchor_ids
                    ):
                        raise ValueError("identifier evidence must occur in a bound context anchor")
        elif self.grounding_reports:
            raise ValueError("abstention assessments cannot contain grounding reports")
        return self


class ReviewedFieldChange(ResearchArtifact):
    """An explicit before/after record for a reviewer edit."""

    field_name: Literal[
        "actor",
        "preconditions",
        "action",
        "protected_asset",
        "expected_secure_behavior",
        "possible_failure",
        "observables",
        "limitations",
    ]
    before: StrictStr
    after: StrictStr


class HumanReviewedRisk(ResearchArtifact):
    """A reviewer-approved immutable successor to one model hypothesis."""

    snapshot_key: Sha256
    assessment_sha256: Sha256
    selected_hypothesis_id: StrictStr = Field(min_length=1)
    selected_hypothesis_sha256: Sha256
    reviewed_risk: RiskHypothesisDraft
    reviewed_content_sha256: Sha256
    added_citation_anchor_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)
    removed_citation_anchor_ids: tuple[StrictStr, ...] = Field(default_factory=tuple)
    field_changes: tuple[ReviewedFieldChange, ...] = Field(default_factory=tuple)
    approved_at: datetime

    @model_validator(mode="after")
    def validate_review_coherence(self) -> HumanReviewedRisk:
        if not _is_utc(self.approved_at):
            raise ValueError("approved_at must be timezone-aware UTC")
        if self.reviewed_content_sha256 != canonical_sha256(
            self.reviewed_risk.model_dump(mode="json")
        ):
            raise ValueError("reviewed content hash must match the reviewed risk")
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
    step_numbers: tuple[StrictInt, ...] = Field(min_length=1)

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


class GherkinCandidateDraft(ResearchArtifact):
    """Raw provider scenario output; it deliberately has no durable identity."""

    snapshot_key: Sha256
    reviewed_risk_sha256: Sha256
    approved_risk: RiskHypothesisDraft
    feature_title: StrictStr = Field(min_length=1)
    scenario_title: StrictStr = Field(min_length=1)
    steps: tuple[GherkinStep, ...] = Field(min_length=1)
    gherkin_text: StrictStr = Field(min_length=1)
    bindings: tuple[GherkinStepBinding, ...]
    testability_notes: tuple[StrictStr, ...]
    setup_gaps: tuple[StrictStr, ...]
    generated_at: datetime

    @field_validator("feature_title", "scenario_title", "gherkin_text", "testability_notes", "setup_gaps")
    @classmethod
    def reject_prohibited_gherkin_claims(cls, value: object) -> object:
        texts = value if isinstance(value, tuple) else (value,)
        if any(
            isinstance(text, str)
            and any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _PROHIBITED_MODEL_CLAIM_PATTERNS)
            for text in texts
        ):
            raise ValueError("model-originated text contains a prohibited claim")
        return value

    @model_validator(mode="after")
    def validate_candidate_coherence(self) -> GherkinCandidateDraft:
        if not _is_utc(self.generated_at):
            raise ValueError("generated_at must be timezone-aware UTC")
        if [step.number for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("Gherkin steps must have consecutive numbers starting at one")
        if self.reviewed_risk_sha256 != canonical_sha256(
            self.approved_risk.model_dump(mode="json")
        ):
            raise ValueError("reviewed risk hash must match the approved risk")
        parsed_steps = _parse_gherkin(
            self.gherkin_text, self.feature_title, self.scenario_title
        )
        if re.search(
            r"```|#|[(){}\[\];]|\b(?:def|class|import|os|sys|subprocess|python|bash|sh|curl|wget|rm|chmod|sudo|eval|exec)\b|/bin/",
            self.gherkin_text,
            flags=re.IGNORECASE,
        ):
            raise ValueError("Gherkin text cannot contain implementation code")
        if any(
            identifier not in self.gherkin_text
            for identifier in self.approved_risk.code_identifiers
        ):
            raise ValueError("Gherkin text must retain approved code identifiers")
        structured_steps = tuple((step.keyword, step.text) for step in self.steps)
        if parsed_steps != structured_steps:
            raise ValueError("Gherkin text steps must exactly match structured steps")
        phases = _gherkin_phases(self.steps)
        step_numbers = {step.number for step in self.steps}
        if any(number not in step_numbers for binding in self.bindings for number in binding.step_numbers):
            raise ValueError("a Gherkin binding must reference an existing step")
        keys = [(binding.claim_field, binding.source_index) for binding in self.bindings]
        if len(keys) != len(set(keys)):
            raise ValueError("Gherkin bindings must not duplicate a source claim")
        required = {("actor", None), ("action", None), ("expected_secure_behavior", None), ("possible_failure", None)}
        required.update(("precondition", index) for index in range(len(self.approved_risk.preconditions)))
        required.update(("observable", index) for index in range(len(self.approved_risk.observables)))
        if set(keys) != required:
            raise ValueError("Gherkin bindings must cover every approved-risk claim")
        allowed_phases = {
            "actor": {"Given"},
            "precondition": {"Given"},
            "action": {"When"},
            "expected_secure_behavior": {"Then"},
            "possible_failure": {"Then"},
            "observable": {"Then"},
        }
        for binding in self.bindings:
            if any(phases[number] not in allowed_phases[binding.claim_field] for number in binding.step_numbers):
                raise ValueError("Gherkin binding references a step in an invalid phase")
        return self


def _parse_gherkin(
    text: str, feature_title: str, scenario_title: str
) -> tuple[tuple[str, str], ...]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if sum(line.startswith("Feature:") for line in lines) != 1:
        raise ValueError("Gherkin text must contain exactly one Feature")
    if sum(line.startswith("Scenario:") for line in lines) != 1:
        raise ValueError("Gherkin text must contain exactly one Scenario")
    if lines[0] != f"Feature: {feature_title}" or lines[1] != f"Scenario: {scenario_title}":
        raise ValueError("Gherkin Feature and Scenario titles must match structured titles")
    parsed: list[tuple[str, str]] = []
    for line in lines[2:]:
        match = re.fullmatch(r"(Given|When|Then|And)\s+(.+)", line)
        if match is None:
            raise ValueError("Gherkin text contains non-step executable or free text")
        parsed.append((match.group(1), match.group(2)))
    if not parsed:
        raise ValueError("Gherkin text must contain steps")
    return tuple(parsed)


def _gherkin_phases(steps: tuple[GherkinStep, ...]) -> dict[int, str]:
    order = {"Given": 0, "When": 1, "Then": 2}
    current: str | None = None
    previous_order = -1
    phases: dict[int, str] = {}
    for step in steps:
        if step.keyword == "And":
            if current is None:
                raise ValueError("Gherkin cannot begin with And")
        else:
            if order[step.keyword] < previous_order:
                raise ValueError("Gherkin steps must follow Given/When/Then phase order")
            current = step.keyword
            previous_order = order[current]
        phases[step.number] = current
    if "Given" not in phases.values() or "When" not in phases.values() or "Then" not in phases.values():
        raise ValueError("Gherkin scenario requires Given, When, and Then phases")
    return phases


class GherkinCandidate(GherkinCandidateDraft):
    """Locally identified, validated successor to raw Gherkin provider output."""

    candidate_id: StrictStr = Field(default="", min_length=1)

    @model_validator(mode="before")
    @classmethod
    def reject_model_supplied_id(cls, value: object) -> object:
        if isinstance(value, dict) and "candidate_id" in value:
            raise ValueError("candidate_id is locally derived and cannot be model supplied")
        return value

    @model_validator(mode="after")
    def derive_candidate_id(self) -> GherkinCandidate:
        content = self.model_dump(mode="json", exclude={"candidate_id"})
        object.__setattr__(self, "candidate_id", f"gherkin-{canonical_sha256(content)}")
        return self

    @classmethod
    def from_draft(cls, draft: GherkinCandidateDraft) -> GherkinCandidate:
        return cls.model_validate(draft.model_dump(mode="json"))

    @classmethod
    def from_persisted(cls, value: dict[str, object]) -> GherkinCandidate:
        supplied_id = value.get("candidate_id")
        draft = GherkinCandidateDraft.model_validate(
            {key: item for key, item in value.items() if key != "candidate_id"}
        )
        result = cls.from_draft(draft)
        if supplied_id != result.candidate_id:
            raise ValueError("persisted candidate ID does not match local derivation")
        return result


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
        assessment = self.risk_assessment
        review = self.human_reviewed_risk
        candidate = self.gherkin_candidate
        approval = self.gherkin_approval
        if self.freshness is not None and self.freshness.snapshot_key != self.snapshot.snapshot_key:
            raise ValueError("freshness check snapshot key must match the terminal snapshot")
        if assessment is not None and assessment.snapshot_key != self.snapshot.snapshot_key:
            raise ValueError("risk assessment snapshot key must match the terminal snapshot")
        if review is not None:
            if assessment is None:
                raise ValueError("a human-reviewed risk requires its risk assessment")
            if review.snapshot_key != self.snapshot.snapshot_key:
                raise ValueError("reviewed risk snapshot key must match the terminal snapshot")
            if review.assessment_sha256 != assessment.assessment_sha256:
                raise ValueError("reviewed risk assessment hash must match the assessment")
            selected = next(
                (risk for risk in assessment.hypotheses if risk.hypothesis_id == review.selected_hypothesis_id),
                None,
            )
            if selected is None:
                raise ValueError("reviewed risk must select a hypothesis from the assessment")
            if review.selected_hypothesis_sha256 != canonical_sha256(selected.model_dump(mode="json")):
                raise ValueError("reviewed risk original hypothesis hash must match the assessment")
            expected_changes = {
                field_name: (canonical_json(getattr(selected, field_name)), canonical_json(getattr(review.reviewed_risk, field_name)))
                for field_name in _EDITABLE_RISK_FIELDS
                if getattr(selected, field_name) != getattr(review.reviewed_risk, field_name)
            }
            reported_changes = {
                change.field_name: (change.before, change.after)
                for change in review.field_changes
            }
            if reported_changes != expected_changes:
                raise ValueError("review field changes must exactly describe the reviewed content")
            original_citations = selected.citation_anchor_ids
            reviewed_citations = review.reviewed_risk.citation_anchor_ids
            expected_added = tuple(
                anchor_id for anchor_id in reviewed_citations if anchor_id not in original_citations
            )
            expected_removed = tuple(
                anchor_id for anchor_id in original_citations if anchor_id not in reviewed_citations
            )
            if (
                review.added_citation_anchor_ids != expected_added
                or review.removed_citation_anchor_ids != expected_removed
            ):
                raise ValueError("review citation deltas must match the reviewed evidence bindings")
        if candidate is not None:
            if review is None:
                raise ValueError("a Gherkin candidate requires a human-reviewed risk")
            if candidate.snapshot_key != self.snapshot.snapshot_key:
                raise ValueError("Gherkin candidate snapshot key must match the terminal snapshot")
            if candidate.reviewed_risk_sha256 != review.reviewed_content_sha256:
                raise ValueError("Gherkin candidate risk hash must match the reviewed risk")
            if candidate.approved_risk != review.reviewed_risk:
                raise ValueError("Gherkin candidate approved risk must match the reviewed risk")
        if approval is not None:
            if candidate is None:
                raise ValueError("a Gherkin approval requires its candidate")
            if approval.snapshot_key != self.snapshot.snapshot_key:
                raise ValueError("Gherkin approval snapshot key must match the terminal snapshot")
            if approval.candidate_id != candidate.candidate_id:
                raise ValueError("Gherkin approval candidate ID must match the candidate")
            if approval.candidate_sha256 != canonical_sha256(candidate.model_dump(mode="json")):
                raise ValueError("Gherkin approval candidate hash must match the candidate")
            if approval.reviewed_risk_sha256 != candidate.reviewed_risk_sha256:
                raise ValueError("Gherkin approval risk hash must match the candidate")
        if self.status is MilestoneTwoStatus.APPROVED_GHERKIN and (
            approval is None or candidate is None or review is None
        ):
            raise ValueError("approved Gherkin requires a candidate, reviewed risk, and approval")
        if self.status is MilestoneTwoStatus.APPROVED_GHERKIN and (
            self.freshness is None
            or self.freshness.status != "current"
            or (
                self.freshness.observed_base_sha,
                self.freshness.observed_head_sha,
                self.freshness.observed_candidate_sha,
            )
            != (self.snapshot.base_sha, self.snapshot.head_sha, self.snapshot.candidate_sha)
            or approval is None
            or self.freshness.checked_at > approval.approved_at
        ):
            raise ValueError("approved Gherkin requires a current final matching freshness check")
        if self.status is MilestoneTwoStatus.NO_MEANINGFUL_SECURITY_RISK_FOUND and (
            self.risk_assessment is None
            or self.risk_assessment.outcome != "no_meaningful_security_risk_found"
        ):
            raise ValueError("no-risk terminal status requires a matching assessment")
        if self.status is MilestoneTwoStatus.NO_MEANINGFUL_SECURITY_RISK_FOUND and (
            self.reason_code != "no_meaningful_security_risk_found"
        ):
            raise ValueError("no-risk terminal status requires its supported reason code")
        if self.status is MilestoneTwoStatus.INSUFFICIENT_CONTEXT_TO_ASSESS and (
            self.risk_assessment is None
            or self.risk_assessment.outcome != "insufficient_context_to_assess"
        ):
            raise ValueError("insufficient-context status requires a matching assessment")
        if self.status is MilestoneTwoStatus.INSUFFICIENT_CONTEXT_TO_ASSESS and (
            self.reason_code not in _insufficient_context_reason_codes()
        ):
            raise ValueError("insufficient-context terminal status requires a supported reason code")
        if self.status is MilestoneTwoStatus.STALE and (
            self.freshness is None or self.freshness.status != "stale"
        ):
            raise ValueError("stale terminal status requires a stale freshness check")
        if self.status is MilestoneTwoStatus.STALE and self.freshness is not None and (
            self.freshness.observed_base_sha == self.snapshot.base_sha
            and self.freshness.observed_head_sha == self.snapshot.head_sha
            and self.freshness.observed_candidate_sha == self.snapshot.candidate_sha
        ):
            raise ValueError("stale terminal status requires an observed revision divergence")
        if self.status is MilestoneTwoStatus.STALE and self.reason_code != "snapshot_stale":
            raise ValueError("stale terminal status requires the snapshot_stale reason code")
        if self.status is MilestoneTwoStatus.APPROVED_GHERKIN and self.reason_code != "gherkin_approved":
            raise ValueError("approved Gherkin requires the gherkin_approved reason code")
        if self.status is MilestoneTwoStatus.FAILED and self.reason_code not in _FAILED_REASON_CODES:
            raise ValueError("failed terminal status requires a supported reason code")
        return self


def _insufficient_context_reason_codes() -> set[str]:
    return set(InsufficientContextReason.__args__)
