"""Deterministic grounding checks for model-proposed risk hypotheses."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from triageguard.domain.pr_analysis import (
    ContextAnchor,
    ContextBundle,
    GroundingReport,
    HumanReviewedRisk,
    IdentifierEvidence,
    PullRequestSnapshot,
    ReviewedFieldChange,
    RiskAssessment,
    RiskAssessmentDraft,
    RiskHypothesis,
    RiskHypothesisDraft,
)
from triageguard.provenance import canonical_json, canonical_sha256

REQUIRED_CLAIM_FIELDS = frozenset(
    {
        "actor",
        "action",
        "expected_secure_behavior",
        "possible_failure",
        "observable",
    }
)

EDITABLE_REVIEW_FIELDS = frozenset(
    {
        "actor",
        "preconditions",
        "action",
        "protected_asset",
        "expected_secure_behavior",
        "possible_failure",
        "observables",
        "limitations",
    }
)

FORBIDDEN_CLAIM_PATTERNS = (
    re.compile(r"\bCVSS\s*:\s*\d", re.IGNORECASE),
    re.compile(
        r"\b(?:is|confirmed as)\s+(?:a\s+)?vulnerab(?:le|ility)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:is|confirmed as)\s+safe\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RiskGroundingReport:
    """A deterministic explanation of whether a draft can become an assessment."""

    approved: bool
    reason_codes: tuple[str, ...]
    validated_hypothesis_ids: tuple[str, ...]


def _add_reason(reason_codes: list[str], reason_code: str) -> None:
    """Record each reason once while preserving deterministic order."""
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)


def _hypothesis_texts(hypothesis: RiskHypothesisDraft) -> tuple[str, ...]:
    """Return every model-authored text field that must avoid decisive claims."""
    return (
        hypothesis.title,
        hypothesis.explanation,
        hypothesis.actor,
        *hypothesis.preconditions,
        hypothesis.action,
        hypothesis.protected_asset,
        hypothesis.security_property,
        hypothesis.expected_secure_behavior,
        hypothesis.possible_failure,
        *hypothesis.observables,
        *hypothesis.code_identifiers,
        *hypothesis.limitations,
        *hypothesis.missing_evidence,
        hypothesis.priority_rationale,
    )


def _has_forbidden_claim(hypothesis: RiskHypothesisDraft) -> bool:
    """Reject decisive vulnerability, safety, or CVSS statements."""
    return any(
        pattern.search(text) is not None
        for pattern in FORBIDDEN_CLAIM_PATTERNS
        for text in _hypothesis_texts(hypothesis)
    )


def _validate_claim_bindings(
    hypothesis: RiskHypothesis,
    anchors: dict[str, ContextAnchor],
    reason_codes: list[str],
) -> None:
    """Check all citations, integration evidence, and code identifiers locally."""
    seen_claims: set[tuple[str, int | None]] = set()

    for binding in hypothesis.evidence_bindings:
        claim_key = (binding.claim_field, binding.observable_index)
        if claim_key in seen_claims:
            _add_reason(reason_codes, "duplicate_claim_evidence_binding")
        seen_claims.add(claim_key)

        if len(binding.anchor_ids) != len(set(binding.anchor_ids)):
            _add_reason(reason_codes, "duplicate_evidence_citation")

        if any(anchor_id not in anchors for anchor_id in binding.anchor_ids):
            _add_reason(reason_codes, "unknown_evidence_anchor")

    required_claims = {
        ("actor", None),
        ("action", None),
        ("expected_secure_behavior", None),
        ("possible_failure", None),
        *(("observable", index) for index in range(len(hypothesis.observables))),
    }
    if (
        not REQUIRED_CLAIM_FIELDS.issubset(
            {claim_field for claim_field, _ in seen_claims}
        )
        or seen_claims != required_claims
    ):
        _add_reason(reason_codes, "missing_claim_evidence_binding")

    cited_anchor_ids = hypothesis.citation_anchor_ids
    known_citations = tuple(
        anchor_id for anchor_id in cited_anchor_ids if anchor_id in anchors
    )
    if known_citations and not any(
        anchors[anchor_id].change_relation == "integration_change"
        for anchor_id in known_citations
    ):
        _add_reason(reason_codes, "missing_integration_evidence")

    for identifier in hypothesis.code_identifiers:
        if not any(
            identifier in anchors[anchor_id].text for anchor_id in known_citations
        ):
            _add_reason(reason_codes, "identifier_not_in_bound_excerpt")


def _grounding_report(
    *,
    hypothesis: RiskHypothesis,
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    anchors: dict[str, ContextAnchor],
) -> GroundingReport:
    """Create the immutable local attestation for one fully grounded risk."""
    cited_anchor_ids = tuple(hypothesis.citation_anchor_ids)
    identifier_evidence = tuple(
        IdentifierEvidence(
            identifier=identifier,
            anchor_ids=tuple(
                anchor_id
                for anchor_id in cited_anchor_ids
                if identifier in anchors[anchor_id].text
            ),
        )
        for identifier in hypothesis.code_identifiers
    )
    return GroundingReport(
        producer="local_grounding_validator",
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_sha256=canonical_sha256(hypothesis.model_dump(mode="json")),
        cited_anchor_ids=cited_anchor_ids,
        identifier_evidence=identifier_evidence,
    )


def validate_risk_assessment(
    *,
    draft: RiskAssessmentDraft,
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
) -> tuple[RiskAssessment | None, RiskGroundingReport]:
    """Validate a draft against one exact frozen snapshot and context bundle."""
    reason_codes: list[str] = []

    try:
        snapshot = PullRequestSnapshot.model_validate(snapshot.model_dump(mode="json"))
    except ValidationError:
        report = RiskGroundingReport(
            approved=False,
            reason_codes=("invalid_snapshot",),
            validated_hypothesis_ids=(),
        )
        return None, report

    try:
        context = ContextBundle.model_validate(context.model_dump(mode="json"))
    except ValidationError:
        report = RiskGroundingReport(
            approved=False,
            reason_codes=("invalid_context_bundle",),
            validated_hypothesis_ids=(),
        )
        return None, report

    try:
        normalized_draft = RiskAssessmentDraft.model_validate(
            draft.model_dump(mode="json")
        )
    except ValidationError:
        report = RiskGroundingReport(
            approved=False,
            reason_codes=("invalid_risk_assessment_draft",),
            validated_hypothesis_ids=(),
        )
        return None, report

    if normalized_draft.snapshot_key != snapshot.snapshot_key:
        _add_reason(reason_codes, "draft_snapshot_mismatch")
    if normalized_draft.context_sha256 != context.context_sha256:
        _add_reason(reason_codes, "draft_context_mismatch")
    if context.snapshot_key != snapshot.snapshot_key:
        _add_reason(reason_codes, "context_snapshot_mismatch")
    if not context.primary_change_represented:
        _add_reason(reason_codes, "primary_change_not_represented")

    anchors = {anchor.anchor_id: anchor for anchor in context.anchors}

    validated_hypotheses: list[RiskHypothesis] = []
    grounding_reports: list[GroundingReport] = []

    if normalized_draft.outcome == "risks_proposed":
        for hypothesis_draft in normalized_draft.hypotheses:
            reasons_before_hypothesis = len(reason_codes)

            if _has_forbidden_claim(hypothesis_draft):
                _add_reason(reason_codes, "prohibited_model_claim")

            hypothesis = RiskHypothesis.from_draft(hypothesis_draft)
            _validate_claim_bindings(
                hypothesis,
                anchors,
                reason_codes,
            )

            if len(reason_codes) == reasons_before_hypothesis:
                validated_hypotheses.append(hypothesis)
                grounding_reports.append(
                    _grounding_report(
                        hypothesis=hypothesis,
                        snapshot=snapshot,
                        context=context,
                        anchors=anchors,
                    )
                )

    elif normalized_draft.outcome == "no_meaningful_security_risk_found":
        if any(
            anchor_id not in anchors
            for anchor_id in normalized_draft.supporting_anchor_ids
        ):
            _add_reason(reason_codes, "unknown_evidence_anchor")
        elif not any(
            anchors[anchor_id].change_relation == "integration_change"
            for anchor_id in normalized_draft.supporting_anchor_ids
        ):
            _add_reason(reason_codes, "missing_integration_evidence")

    if reason_codes:
        report = RiskGroundingReport(
            approved=False,
            reason_codes=tuple(reason_codes),
            validated_hypothesis_ids=(),
        )
        return None, report

    try:
        assessment_values = {
            **normalized_draft.model_dump(),
            "hypotheses": tuple(validated_hypotheses),
            "validated_at": normalized_draft.generated_at,
            "context_bundle": context,
            "grounding_reports": tuple(grounding_reports),
        }
        assessment = RiskAssessment.from_content(**assessment_values)
    except ValueError:
        report = RiskGroundingReport(
            approved=False,
            reason_codes=("assessment_validation_failed",),
            validated_hypothesis_ids=(),
        )
        return None, report

    report = RiskGroundingReport(
        approved=True,
        reason_codes=(),
        validated_hypothesis_ids=tuple(
            hypothesis.hypothesis_id for hypothesis in assessment.hypotheses
        ),
    )
    return assessment, report


def _review_value(value: object) -> str:
    """Render a field change without changing its underlying typed value."""
    if isinstance(value, str):
        return value
    return canonical_json(value)


def _reviewed_grounding_report(
    *,
    hypothesis: RiskHypothesis,
    assessment: RiskAssessment,
    anchors: dict[str, ContextAnchor],
    reviewed_content_sha256: str,
) -> GroundingReport:
    """Create grounding evidence for the immutable reviewed successor."""
    cited_anchor_ids = tuple(hypothesis.citation_anchor_ids)
    identifier_evidence = tuple(
        IdentifierEvidence(
            identifier=identifier,
            anchor_ids=tuple(
                anchor_id
                for anchor_id in cited_anchor_ids
                if identifier in anchors[anchor_id].text
            ),
        )
        for identifier in hypothesis.code_identifiers
    )
    return GroundingReport(
        producer="local_grounding_validator",
        snapshot_key=assessment.snapshot_key,
        context_sha256=assessment.context_sha256,
        hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_sha256=reviewed_content_sha256,
        cited_anchor_ids=cited_anchor_ids,
        identifier_evidence=identifier_evidence,
    )


def create_human_review(
    *,
    assessment: RiskAssessment,
    hypothesis_id: str,
    edits: Mapping[str, object],
    selected_anchor_ids: Sequence[str],
    reviewed_at: datetime,
) -> HumanReviewedRisk:
    """Create an immutable, locally re-grounded human review of one risk."""
    original = next(
        (
            hypothesis
            for hypothesis in assessment.hypotheses
            if hypothesis.hypothesis_id == hypothesis_id
        ),
        None,
    )
    if original is None:
        raise ValueError("selected hypothesis does not belong to the assessment")

    unapproved_fields = set(edits) - EDITABLE_REVIEW_FIELDS
    if unapproved_fields:
        raise ValueError("review edits may only change approved editable fields")

    selected = tuple(selected_anchor_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected review anchors must be a non-empty unique list")

    anchors = {anchor.anchor_id: anchor for anchor in assessment.context_bundle.anchors}
    if any(anchor_id not in anchors for anchor_id in selected):
        raise ValueError("selected review anchors must exist in frozen context")

    original_citations = tuple(original.citation_anchor_ids)
    if set(selected) != set(original_citations):
        raise ValueError(
            "selected review anchors must match the risk's existing citations"
        )

    reviewed_payload = {
        **original.model_dump(
            mode="python",
            exclude={"hypothesis_id"},
        ),
        **dict(edits),
    }
    reviewed_risk = RiskHypothesisDraft.model_validate(reviewed_payload)
    reviewed_hypothesis = RiskHypothesis.from_draft(reviewed_risk)

    grounding_reasons: list[str] = []
    if _has_forbidden_claim(reviewed_risk):
        _add_reason(grounding_reasons, "prohibited_model_claim")
    _validate_claim_bindings(
        reviewed_hypothesis,
        anchors,
        grounding_reasons,
    )
    if grounding_reasons:
        raise ValueError(
            "reviewed risk does not remain grounded: " + ", ".join(grounding_reasons)
        )

    reviewed_citations = tuple(reviewed_risk.citation_anchor_ids)
    field_changes = tuple(
        ReviewedFieldChange(
            field_name=field_name,
            before=_review_value(getattr(original, field_name)),
            after=_review_value(getattr(reviewed_risk, field_name)),
        )
        for field_name in sorted(EDITABLE_REVIEW_FIELDS)
        if field_name in edits
        and getattr(original, field_name) != getattr(reviewed_risk, field_name)
    )

    return HumanReviewedRisk(
        snapshot_key=assessment.snapshot_key,
        assessment_sha256=assessment.assessment_sha256,
        selected_hypothesis_id=original.hypothesis_id,
        selected_hypothesis_sha256=canonical_sha256(original.model_dump(mode="json")),
        reviewed_risk=reviewed_risk,
        reviewed_content_sha256=canonical_sha256(reviewed_risk.model_dump(mode="json")),
        reviewed_grounding=_reviewed_grounding_report(
            hypothesis=reviewed_hypothesis,
            assessment=assessment,
            anchors=anchors,
            reviewed_content_sha256=canonical_sha256(
                reviewed_risk.model_dump(mode="json")
            ),
        ),
        added_citation_anchor_ids=tuple(
            anchor_id
            for anchor_id in reviewed_citations
            if anchor_id not in original_citations
        ),
        removed_citation_anchor_ids=tuple(
            anchor_id
            for anchor_id in original_citations
            if anchor_id not in reviewed_citations
        ),
        field_changes=field_changes,
        approved_at=reviewed_at,
    )
