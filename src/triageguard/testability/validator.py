"""Local deterministic validation for frozen-evidence testability decisions."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from triageguard.domain import (
    ContextBundle,
    HumanReviewedRisk,
    TestabilityAssessment,
    TestabilityAssessmentDraft,
)


@dataclass(frozen=True)
class TestabilityValidationReport:
    """The local decision about whether a model draft may become durable evidence."""

    approved: bool
    reason_codes: tuple[str, ...]


def validate_testability_assessment(
    *,
    draft: TestabilityAssessmentDraft,
    human_review: HumanReviewedRisk,
    context: ContextBundle,
) -> tuple[TestabilityAssessment | None, TestabilityValidationReport]:
    """Validate one raw model decision against the saved review and context."""
    try:
        normalized_draft = TestabilityAssessmentDraft.model_validate(
            draft.model_dump(mode="json")
        )
    except ValidationError:
        return None, TestabilityValidationReport(
            approved=False,
            reason_codes=("invalid_testability_draft",),
        )

    try:
        reviewed = HumanReviewedRisk.model_validate(
            human_review.model_dump(mode="json")
        )
    except ValidationError:
        return None, TestabilityValidationReport(
            approved=False,
            reason_codes=("invalid_human_review",),
        )

    try:
        frozen_context = ContextBundle.model_validate(context.model_dump(mode="json"))
    except ValidationError:
        return None, TestabilityValidationReport(
            approved=False,
            reason_codes=("invalid_context_bundle",),
        )

    reason_codes: list[str] = []

    if reviewed.snapshot_key != frozen_context.snapshot_key:
        _add_reason(reason_codes, "review_snapshot_mismatch")
    if normalized_draft.snapshot_key != frozen_context.snapshot_key:
        _add_reason(reason_codes, "draft_snapshot_mismatch")
    if normalized_draft.context_sha256 != frozen_context.context_sha256:
        _add_reason(reason_codes, "draft_context_mismatch")
    if normalized_draft.reviewed_risk_sha256 != reviewed.reviewed_content_sha256:
        _add_reason(reason_codes, "draft_review_mismatch")

    anchors = {anchor.anchor_id: anchor for anchor in frozen_context.anchors}
    referenced_anchor_ids = tuple(
        anchor_id
        for binding in normalized_draft.bindings
        for anchor_id in binding.anchor_ids
    ) + tuple(
        anchor_id
        for need in normalized_draft.evidence_needs
        for anchor_id in need.supporting_anchor_ids
    )

    if any(anchor_id not in anchors for anchor_id in referenced_anchor_ids):
        _add_reason(reason_codes, "unknown_testability_anchor")

    known_references = tuple(
        anchor_id for anchor_id in referenced_anchor_ids if anchor_id in anchors
    )
    if normalized_draft.decision in {
        "testable_from_frozen_evidence",
        "needs_more_frozen_evidence",
    } and not any(
        anchors[anchor_id].change_relation == "integration_change"
        for anchor_id in known_references
    ):
        _add_reason(reason_codes, "missing_integration_testability_evidence")

    if reason_codes:
        return None, TestabilityValidationReport(
            approved=False,
            reason_codes=tuple(reason_codes),
        )

    try:
        assessment = TestabilityAssessment.from_content(
            **normalized_draft.model_dump(mode="python"),
            validated_at=normalized_draft.generated_at,
        )
    except ValueError:
        return None, TestabilityValidationReport(
            approved=False,
            reason_codes=("testability_assessment_validation_failed",),
        )

    return assessment, TestabilityValidationReport(
        approved=True,
        reason_codes=(),
    )


def _add_reason(reason_codes: list[str], reason_code: str) -> None:
    """Record each local validation reason only once in a stable order."""
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)
