"""Tests for LLM-generated Gherkin bound to a human-reviewed risk."""

from datetime import UTC, datetime

import pytest

from triageguard.contracts.gherkin_generation import (
    GherkinGenerationError,
    apply_gherkin_text_edit,
    approve_gherkin,
    build_gherkin_request,
    generate_gherkin,
    validate_gherkin_candidate,
)
from triageguard.domain.pr_analysis import (
    ClaimEvidenceBinding,
    HumanReviewedRisk,
    RiskHypothesisDraft,
)
from triageguard.llm.replay_gateway import ReplayGateway
from triageguard.provenance import canonical_sha256


def _human_review() -> HumanReviewedRisk:
    """Return one immutable human-approved unconfirmed risk."""
    reviewed_risk = RiskHypothesisDraft(
        claim_status="unconfirmed_risk_hypothesis",
        title="Patient deletion may bypass an expected authorization check",
        explanation=(
            "The changed deletion path needs an executable authorization check."
        ),
        actor="An authenticated OpenMRS user",
        preconditions=("The user can reach the patient-deletion service path.",),
        action="The user requests deletion of a patient record.",
        protected_asset="Patient records",
        security_property="Authorization",
        expected_secure_behavior=(
            "The API rejects deletion and the patient remains stored."
        ),
        possible_failure=(
            "The API deletes a patient record without the expected authorization."
        ),
        observables=(
            "The deletion request is rejected.",
            "The patient record remains stored.",
        ),
        code_identifiers=("purgePatient", "deletePatient"),
        evidence_bindings=(
            ClaimEvidenceBinding(
                claim_field="actor",
                observable_index=None,
                anchor_ids=("anchor-integration",),
            ),
            ClaimEvidenceBinding(
                claim_field="action",
                observable_index=None,
                anchor_ids=("anchor-integration",),
            ),
            ClaimEvidenceBinding(
                claim_field="expected_secure_behavior",
                observable_index=None,
                anchor_ids=("anchor-integration",),
            ),
            ClaimEvidenceBinding(
                claim_field="possible_failure",
                observable_index=None,
                anchor_ids=("anchor-integration",),
            ),
            ClaimEvidenceBinding(
                claim_field="observable",
                observable_index=0,
                anchor_ids=("anchor-integration",),
            ),
            ClaimEvidenceBinding(
                claim_field="observable",
                observable_index=1,
                anchor_ids=("anchor-integration",),
            ),
        ),
        limitations=(
            "The authorization implementation is outside the bounded context.",
        ),
        missing_evidence=(),
        priority_rationale=(
            "Patient deletion is security-relevant and needs an executable test."
        ),
    )
    return HumanReviewedRisk(
        snapshot_key="a" * 64,
        assessment_sha256="b" * 64,
        selected_hypothesis_id="risk-original",
        selected_hypothesis_sha256="c" * 64,
        reviewed_risk=reviewed_risk,
        reviewed_content_sha256=canonical_sha256(reviewed_risk.model_dump(mode="json")),
        approved_at=datetime(2026, 8, 12, tzinfo=UTC),
    )


def test_gherkin_request_is_bound_to_the_human_review() -> None:
    """The model request contains the exact approved risk, not a raw proposal."""
    human_review = _human_review()

    request = build_gherkin_request(human_review)

    assert request.purpose == "gherkin_generation"
    assert request.payload["snapshot_key"] == human_review.snapshot_key
    assert request.payload["reviewed_risk_sha256"] == canonical_sha256(
        human_review.reviewed_risk.model_dump(mode="json")
    )
    assert request.payload["approved_risk"] == human_review.reviewed_risk.model_dump(
        mode="json"
    )
    assert request.output_schema["additionalProperties"] is False


def _candidate_response(human_review: HumanReviewedRisk) -> dict[str, object]:
    """Return one locally replayed structured Gherkin response."""
    approved_risk = human_review.reviewed_risk
    steps = [
        {
            "number": 1,
            "keyword": "Given",
            "text": (
                "an authenticated OpenMRS user can reach the purgePatient "
                "patient-deletion service path"
            ),
        },
        {
            "number": 2,
            "keyword": "When",
            "text": ("the user requests deletePatient for a patient record"),
        },
        {
            "number": 3,
            "keyword": "Then",
            "text": approved_risk.expected_secure_behavior,
        },
        {
            "number": 4,
            "keyword": "And",
            "text": approved_risk.possible_failure,
        },
        {
            "number": 5,
            "keyword": "And",
            "text": approved_risk.observables[0],
        },
        {
            "number": 6,
            "keyword": "And",
            "text": approved_risk.observables[1],
        },
    ]
    feature_title = "Patient deletion authorization"
    scenario_title = "Unauthorized patient deletion is rejected"
    gherkin_text = "\n".join(
        [
            f"Feature: {feature_title}",
            "",
            f"Scenario: {scenario_title}",
            "",
            *(f"{step['keyword']} {step['text']}" for step in steps),
        ]
    )

    return {
        "snapshot_key": human_review.snapshot_key,
        "reviewed_risk_sha256": human_review.reviewed_content_sha256,
        "approved_risk": approved_risk.model_dump(mode="json"),
        "feature_title": feature_title,
        "scenario_title": scenario_title,
        "steps": steps,
        "gherkin_text": gherkin_text,
        "bindings": [
            {
                "claim_field": "actor",
                "source_index": None,
                "step_numbers": [1],
            },
            {
                "claim_field": "precondition",
                "source_index": 0,
                "step_numbers": [1],
            },
            {
                "claim_field": "action",
                "source_index": None,
                "step_numbers": [2],
            },
            {
                "claim_field": "expected_secure_behavior",
                "source_index": None,
                "step_numbers": [3],
            },
            {
                "claim_field": "possible_failure",
                "source_index": None,
                "step_numbers": [4],
            },
            {
                "claim_field": "observable",
                "source_index": 0,
                "step_numbers": [5],
            },
            {
                "claim_field": "observable",
                "source_index": 1,
                "step_numbers": [6],
            },
        ],
        "testability_notes": ["Run the scenario against an OpenMRS test environment."],
        "setup_gaps": [],
        "generated_at": "2026-08-12T00:00:00Z",
    }


def test_generate_gherkin_returns_a_locally_identified_candidate() -> None:
    """A schema-valid replay response becomes a deterministic candidate."""
    human_review = _human_review()
    gateway = ReplayGateway(
        {
            "gherkin_generation": _candidate_response(human_review),
        }
    )

    candidate, response = generate_gherkin(
        human_review=human_review,
        gateway=gateway,
    )

    assert candidate.snapshot_key == human_review.snapshot_key
    assert candidate.reviewed_risk_sha256 == human_review.reviewed_content_sha256
    assert candidate.approved_risk == human_review.reviewed_risk
    assert candidate.candidate_id.startswith("gherkin-")
    assert response.provider == "replay"


def _candidate(human_review: HumanReviewedRisk):
    """Return one locally validated candidate from the replay fixture."""
    candidate, _ = generate_gherkin(
        human_review=human_review,
        gateway=ReplayGateway(
            {
                "gherkin_generation": _candidate_response(human_review),
            }
        ),
    )
    return candidate


def test_edit_cannot_remove_the_failure_oracle() -> None:
    """A wording edit cannot erase the approved secure-behavior oracle."""
    human_review = _human_review()
    candidate = _candidate(human_review)
    edited_text = candidate.gherkin_text.replace(
        human_review.reviewed_risk.expected_secure_behavior,
        "the page loads",
    )

    with pytest.raises(
        GherkinGenerationError,
        match="bound_risk_term_removed",
    ):
        apply_gherkin_text_edit(
            candidate=candidate,
            text=edited_text,
            human_review=human_review,
        )


def test_approve_gherkin_binds_exact_candidate_and_review_hashes() -> None:
    """Human approval records immutable identities for later freshness checks."""
    human_review = _human_review()
    candidate = _candidate(human_review)

    approval = approve_gherkin(
        candidate=candidate,
        human_review=human_review,
        approved_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assert approval.snapshot_key == human_review.snapshot_key
    assert approval.candidate_id == candidate.candidate_id
    assert approval.candidate_sha256 == canonical_sha256(
        candidate.model_dump(mode="json")
    )
    assert approval.reviewed_risk_sha256 == human_review.reviewed_content_sha256


def test_candidate_validation_accepts_the_exact_reviewed_scenario() -> None:
    """The locally generated candidate remains tied to its human review."""
    human_review = _human_review()
    candidate = _candidate(human_review)

    report = validate_gherkin_candidate(
        candidate=candidate,
        human_review=human_review,
    )

    assert report.approved is True
    assert report.reason_codes == ()


def test_edit_rejects_step_insertion_and_reordering() -> None:
    """The first prototype permits wording changes, never scenario reshaping."""
    human_review = _human_review()
    candidate = _candidate(human_review)
    action_line = "When the user requests deletePatient for a patient record"

    inserted_step_text = candidate.gherkin_text.replace(
        action_line,
        "Given an unrelated setup step\n" + action_line,
    )
    with pytest.raises(
        GherkinGenerationError,
        match="gherkin_step_structure_changed",
    ):
        apply_gherkin_text_edit(
            candidate=candidate,
            text=inserted_step_text,
            human_review=human_review,
        )

    reordered_step_text = candidate.gherkin_text.replace(
        (
            "Then The API rejects deletion and the patient remains stored.\n"
            "And The API deletes a patient record without the expected authorization."
        ),
        (
            "And The API deletes a patient record without the expected authorization.\n"
            "Then The API rejects deletion and the patient remains stored."
        ),
    )
    with pytest.raises(
        GherkinGenerationError,
        match="gherkin_step_structure_changed",
    ):
        apply_gherkin_text_edit(
            candidate=candidate,
            text=reordered_step_text,
            human_review=human_review,
        )


def test_gherkin_request_rejects_a_tampered_human_review() -> None:
    """A changed review hash cannot be used to request a new scenario."""
    human_review = _human_review()
    tampered_review = human_review.model_copy(
        update={"reviewed_content_sha256": "f" * 64}
    )

    with pytest.raises(ValueError, match="human review"):
        build_gherkin_request(tampered_review)
