"""Tests for structured model requests about frozen-evidence testability."""

import hashlib
from datetime import UTC, datetime

import pytest

from triageguard.contracts.gherkin_generation import build_gherkin_request
from triageguard.domain import (
    ClaimEvidenceBinding,
    ContextAnchor,
    ContextBundle,
    HumanReviewedRisk,
    RiskHypothesisDraft,
)
from triageguard.domain import (
    TestabilityAssessment as Assessment,
)
from triageguard.domain import (
    TestabilityAssessmentDraft as RawAssessment,
)
from triageguard.llm import ModelOutputInvalid, ReplayGateway
from triageguard.provenance import canonical_json, canonical_sha256
from triageguard.testability.generator import (
    build_testability_request,
    generate_testability_assessment,
)
from triageguard.testability.validator import validate_testability_assessment

NOW = datetime(2026, 8, 16, tzinfo=UTC)
SNAPSHOT_KEY = "a" * 64


def _context() -> ContextBundle:
    """Return one saved integration-change anchor."""
    text = "void verifyDeleteAuthorization() {\n    requirePrivilege();\n}\n"
    anchor = ContextAnchor(
        anchor_id="anchor-authorization",
        revision_role="candidate",
        commit_sha="b" * 40,
        blob_sha="c" * 40,
        path="api/AuthorizationService.java",
        java_symbol="verifyDeleteAuthorization",
        start_line=3,
        end_line=5,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        selection_reason="integration change",
        score_components=(),
        change_relation="integration_change",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key=SNAPSHOT_KEY,
        anchors=(anchor,),
        selected_file_count=1,
        selected_anchor_count=1,
        selected_bytes=len(text.encode("utf-8")),
        max_files=10,
        max_anchors=20,
        max_bytes=40_000,
        max_anchor_lines=40,
        max_blob_bytes=40_000,
        max_search_identifiers=20,
        max_hits_per_identifier=10,
        primary_change_represented=True,
    )


def _human_review() -> HumanReviewedRisk:
    """Return one evidence-bound reviewer-approved risk hypothesis."""
    risk = RiskHypothesisDraft(
        claim_status="unconfirmed_risk_hypothesis",
        title="Patient deletion may bypass authorization",
        explanation=(
            "The changed deletion path needs an executable authorization check."
        ),
        actor="An authenticated OpenMRS user",
        preconditions=("The user can reach the patient-deletion service path.",),
        action="The user requests deletion of a patient record.",
        protected_asset="Patient records",
        security_property="Authorization",
        expected_secure_behavior=(
            "The deletion path enforces the required authorization."
        ),
        possible_failure=(
            "The changed path may delete a patient record without authorization."
        ),
        observables=("The deletion request is rejected.",),
        code_identifiers=("verifyDeleteAuthorization", "requirePrivilege"),
        evidence_bindings=(
            ClaimEvidenceBinding(
                claim_field="explanation",
                observable_index=None,
                anchor_ids=("anchor-authorization",),
            ),
            ClaimEvidenceBinding(
                claim_field="actor",
                observable_index=None,
                anchor_ids=("anchor-authorization",),
            ),
            ClaimEvidenceBinding(
                claim_field="action",
                observable_index=None,
                anchor_ids=("anchor-authorization",),
            ),
            ClaimEvidenceBinding(
                claim_field="expected_secure_behavior",
                observable_index=None,
                anchor_ids=("anchor-authorization",),
            ),
            ClaimEvidenceBinding(
                claim_field="possible_failure",
                observable_index=None,
                anchor_ids=("anchor-authorization",),
            ),
            ClaimEvidenceBinding(
                claim_field="observable",
                observable_index=0,
                anchor_ids=("anchor-authorization",),
            ),
        ),
        limitations=("Only bounded frozen code evidence was reviewed.",),
        missing_evidence=(),
        priority_rationale="Patient deletion is security-relevant.",
    )
    return HumanReviewedRisk(
        snapshot_key=SNAPSHOT_KEY,
        assessment_sha256="d" * 64,
        selected_hypothesis_id="risk-delete-patient",
        selected_hypothesis_sha256="e" * 64,
        reviewed_risk=risk,
        reviewed_content_sha256=canonical_sha256(risk.model_dump(mode="json")),
        approved_at=NOW,
    )


def test_testability_request_contains_only_reviewed_risk_and_frozen_context() -> None:
    """The model receives no credentials, live Git data, or unrestricted source."""
    review = _human_review()
    context = _context()

    request = build_testability_request(
        human_review=review,
        context=context,
    )

    serialized = canonical_json(request.payload)

    assert request.purpose == "testability_assessment"
    assert set(request.payload) == {
        "snapshot_key",
        "reviewed_risk_sha256",
        "reviewed_risk",
        "context_anchors",
        "context_limits",
        "output_rules",
    }
    assert request.payload["snapshot_key"] == SNAPSHOT_KEY
    assert request.payload["reviewed_risk_sha256"] == review.reviewed_content_sha256
    assert request.payload["context_anchors"] == [
        anchor.model_dump(mode="json") for anchor in context.anchors
    ]
    assert "GROQ_API_KEY" not in serialized
    assert "GITHUB_TOKEN" not in serialized
    assert request.output_schema["additionalProperties"] is False


def _model_response(
    review: HumanReviewedRisk,
    context: ContextBundle,
) -> dict[str, object]:
    """Return one schema-valid model decision based only on the saved anchor."""
    return {
        "snapshot_key": review.snapshot_key,
        "context_sha256": context.context_sha256,
        "reviewed_risk_sha256": review.reviewed_content_sha256,
        "decision": "testable_from_frozen_evidence",
        "bindings": [
            {
                "role": "setup",
                "anchor_ids": ["anchor-authorization"],
            },
            {
                "role": "action",
                "anchor_ids": ["anchor-authorization"],
            },
            {
                "role": "observable",
                "anchor_ids": ["anchor-authorization"],
            },
        ],
        "evidence_needs": [],
        "explanation": (
            "The saved authorization method provides setup, action, and an "
            "observable authorization outcome."
        ),
        "generated_at": NOW.isoformat(),
    }


def test_generator_accepts_a_testable_response_bound_to_frozen_anchors() -> None:
    """A replayed model response becomes a raw testability draft only when bound."""
    review = _human_review()
    context = _context()
    gateway = ReplayGateway(
        {
            "testability_assessment": _model_response(
                review,
                context,
            )
        }
    )

    assessment, response = generate_testability_assessment(
        human_review=review,
        context=context,
        gateway=gateway,
    )

    assert assessment.decision == "testable_from_frozen_evidence"
    assert assessment.context_sha256 == context.context_sha256
    assert assessment.reviewed_risk_sha256 == review.reviewed_content_sha256
    assert response.provider == "replay"


def test_generator_rejects_a_model_response_with_a_fabricated_anchor() -> None:
    """The model may not cite evidence outside the frozen context catalog."""
    review = _human_review()
    context = _context()
    response_data = _model_response(review, context)
    response_data["bindings"][0]["anchor_ids"] = ["anchor-invented"]
    gateway = ReplayGateway({"testability_assessment": response_data})

    with pytest.raises(ModelOutputInvalid, match="absent from the frozen context"):
        generate_testability_assessment(
            human_review=review,
            context=context,
            gateway=gateway,
        )


def test_local_validator_turns_a_grounded_model_draft_into_a_hashed_assessment() -> (
    None
):
    """Only locally checked testability output becomes a durable assessment."""
    review = _human_review()
    context = _context()
    draft = RawAssessment.model_validate(_model_response(review, context))

    assessment, report = validate_testability_assessment(
        draft=draft,
        human_review=review,
        context=context,
    )

    assert report.approved is True
    assert assessment is not None
    assert assessment.decision == "testable_from_frozen_evidence"
    assert len(assessment.assessment_sha256) == 64


def test_local_validator_rejects_a_draft_with_a_fabricated_anchor() -> None:
    """A raw model draft cannot bypass local frozen-context anchor checks."""
    review = _human_review()
    context = _context()
    response_data = _model_response(review, context)
    response_data["bindings"][0]["anchor_ids"] = ["anchor-invented"]
    draft = RawAssessment.model_validate(response_data)

    assessment, report = validate_testability_assessment(
        draft=draft,
        human_review=review,
        context=context,
    )

    assert assessment is None
    assert report.approved is False
    assert report.reason_codes == ("unknown_testability_anchor",)


def test_gherkin_request_requires_a_locally_validated_testability_assessment() -> None:
    """A scenario request is tied to the exact review, context, and testability gate."""
    review = _human_review()
    context = _context()
    draft = RawAssessment.model_validate(_model_response(review, context))
    assessment = Assessment.from_content(
        **draft.model_dump(mode="python"),
        validated_at=NOW,
    )

    request = build_gherkin_request(
        human_review=review,
        testability_assessment=assessment,
        context=context,
    )

    assert request.payload["snapshot_key"] == review.snapshot_key
    assert request.payload["reviewed_risk_sha256"] == review.reviewed_content_sha256
    assert request.payload["testability_assessment_sha256"] == (
        assessment.assessment_sha256
    )
    assert request.payload["context_sha256"] == context.context_sha256
    assert request.payload["context_anchors"] == [
        anchor.model_dump(mode="json") for anchor in context.anchors
    ]
