"""Tests for immutable testability and frozen-evidence refinement artifacts."""

from datetime import UTC, datetime

import pytest

from triageguard.domain import (
    ContextRefinement as Refinement,
)
from triageguard.domain import (
    FrozenEvidenceNeed as EvidenceNeed,
)
from triageguard.domain import (
    TestabilityAssessment as Assessment,
)
from triageguard.domain import (
    TestabilityBinding as Binding,
)

NOW = datetime(2026, 8, 15, tzinfo=UTC)
SNAPSHOT_KEY = "a" * 64
CONTEXT_SHA256 = "b" * 64
REFINED_CONTEXT_SHA256 = "c" * 64
REVIEWED_RISK_SHA256 = "d" * 64


def _testable_values() -> dict[str, object]:
    """Return the smallest complete testable assessment payload."""
    return {
        "snapshot_key": SNAPSHOT_KEY,
        "context_sha256": CONTEXT_SHA256,
        "reviewed_risk_sha256": REVIEWED_RISK_SHA256,
        "evidence_envelope_sha256": "e" * 64,
        "decision": "testable_from_frozen_evidence",
        "bindings": (
            Binding(
                role="setup",
                anchor_ids=("anchor-setup",),
            ),
            Binding(
                role="action",
                anchor_ids=("anchor-action",),
            ),
            Binding(
                role="observable",
                anchor_ids=("anchor-observable",),
            ),
        ),
        "evidence_needs": (),
        "explanation": (
            "The frozen code identifies setup, action, and an observable outcome."
        ),
        "generated_at": NOW,
        "validated_at": NOW,
    }


def test_testable_assessment_derives_a_local_content_hash() -> None:
    """A valid testability result has all three evidence-bound test roles."""
    assessment = Assessment.from_content(**_testable_values())

    assert assessment.decision == "testable_from_frozen_evidence"
    assert len(assessment.assessment_sha256) == 64
    assert len(assessment.bindings) == 3


def test_testable_assessment_requires_setup_action_and_observable() -> None:
    """A model cannot call a risk testable while omitting a required test role."""
    values = _testable_values()
    values["bindings"] = (
        Binding(
            role="setup",
            anchor_ids=("anchor-setup",),
        ),
        Binding(
            role="action",
            anchor_ids=("anchor-action",),
        ),
    )

    with pytest.raises(ValueError, match="setup.*action.*observable"):
        Assessment.from_content(**values)


def test_context_refinement_records_a_distinct_successor_context() -> None:
    """A successful refinement records its parent, successor, and added anchors."""
    need = EvidenceNeed(
        need_id="need-delete-route",
        category="entry_point",
        search_terms=("deletePatient",),
        explanation="Find the frozen code route that reaches patient deletion.",
        supporting_anchor_ids=("anchor-action",),
    )

    refinement = Refinement.from_content(
        snapshot_key=SNAPSHOT_KEY,
        parent_context_sha256=CONTEXT_SHA256,
        refined_context_sha256=REFINED_CONTEXT_SHA256,
        evidence_need_ids=(need.need_id,),
        added_anchor_ids=("anchor-delete-route",),
        exhausted=False,
        created_at=NOW,
    )

    assert refinement.parent_context_sha256 == CONTEXT_SHA256
    assert refinement.refined_context_sha256 == REFINED_CONTEXT_SHA256
    assert refinement.parent_context_sha256 != refinement.refined_context_sha256
