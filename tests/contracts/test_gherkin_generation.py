"""Tests for LLM-generated Gherkin bound to a human-reviewed risk."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from triageguard.contracts import gherkin_generation
from triageguard.contracts.gherkin_generation import (
    GherkinGenerationError,
    apply_gherkin_text_edit,
    approve_gherkin,
    build_gherkin_request,
    generate_gherkin,
    validate_edited_gherkin,
    validate_gherkin_candidate,
)
from triageguard.domain.pr_analysis import (
    ClaimEvidenceBinding,
    ContextAnchor,
    ContextBundle,
    GherkinCandidateDraft,
    GherkinStepEvidenceBinding,
    HumanReviewedRisk,
    RiskHypothesisDraft,
)
from triageguard.domain.pr_analysis import (
    TestabilityAssessment as ValidatedTestabilityAssessment,
)
from triageguard.domain.pr_analysis import (
    TestabilityBinding as FrozenTestabilityBinding,
)
from triageguard.evidence import EvidenceArtifactBinding, ModelEvidenceBudgetError
from triageguard.llm import ProviderRequestBudget
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
                claim_field="explanation",
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


def _comparison_bindings() -> tuple[EvidenceArtifactBinding, ...]:
    return (
        EvidenceArtifactBinding(name="author_diff", sha256="1" * 64),
        EvidenceArtifactBinding(name="integration_diff", sha256="2" * 64),
        EvidenceArtifactBinding(name="base_drift_diff", sha256="3" * 64),
    )


def _context() -> ContextBundle:
    """Return saved integration-change evidence for the approved risk."""
    text = "void purgePatient(Patient patient) {\n    deletePatient(patient);\n}\n"
    anchor = ContextAnchor(
        anchor_id="anchor-integration",
        revision_role="candidate",
        commit_sha="d" * 40,
        blob_sha="e" * 40,
        path="api/PatientService.java",
        java_symbol="purgePatient",
        start_line=1,
        end_line=3,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        selection_reason="primary integration change",
        score_components=(),
        change_relation="integration_change",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key="a" * 64,
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


def _context_with_hidden_integration_anchor() -> ContextBundle:
    """Return one required anchor and one oversized hidden integration anchor."""
    context = _context()
    text = "void hiddenAuthorizationPath() {}\n" * 300
    hidden = context.anchors[0].model_copy(
        update={
            "anchor_id": "anchor-hidden-integration",
            "blob_sha": "f" * 40,
            "path": "api/HiddenAuthorizationPath.java",
            "java_symbol": "hiddenAuthorizationPath",
            "start_line": 10,
            "end_line": 309,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    )
    return ContextBundle.from_content(
        **context.model_dump(
            mode="python",
            exclude={
                "anchors",
                "context_sha256",
                "selected_anchor_count",
                "selected_file_count",
                "selected_bytes",
                "max_anchor_lines",
            },
        ),
        anchors=(*context.anchors, hidden),
        selected_file_count=2,
        selected_anchor_count=2,
        selected_bytes=sum(
            len(anchor.text.encode("utf-8")) for anchor in (*context.anchors, hidden)
        ),
        max_anchor_lines=400,
    )


def _testability_assessment(
    human_review: HumanReviewedRisk,
    context: ContextBundle,
) -> ValidatedTestabilityAssessment:
    """Return one locally shaped testability approval for the frozen anchor."""
    return ValidatedTestabilityAssessment.from_content(
        snapshot_key=human_review.snapshot_key,
        context_sha256=context.context_sha256,
        reviewed_risk_sha256=human_review.reviewed_content_sha256,
        evidence_envelope_sha256="4" * 64,
        decision="testable_from_frozen_evidence",
        bindings=(
            FrozenTestabilityBinding(
                role="setup",
                anchor_ids=("anchor-integration",),
            ),
            FrozenTestabilityBinding(
                role="action",
                anchor_ids=("anchor-integration",),
            ),
            FrozenTestabilityBinding(
                role="observable",
                anchor_ids=("anchor-integration",),
            ),
        ),
        evidence_needs=(),
        explanation=(
            "The saved integration change supplies setup, action, and an "
            "observable authorization outcome."
        ),
        generated_at=datetime(2026, 8, 16, tzinfo=UTC),
        validated_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def _gherkin_envelope(
    human_review: HumanReviewedRisk,
    context: ContextBundle,
    assessment: ValidatedTestabilityAssessment,
):
    return gherkin_generation.build_gherkin_evidence(
        human_review=human_review,
        testability_assessment=assessment,
        context=context,
        comparison_bindings=_comparison_bindings(),
        budget=ProviderRequestBudget(
            provider="groq",
            model="openai/gpt-oss-120b",
            max_body_bytes=7_000,
        ),
    ).envelope


def _gherkin_boundary(
    human_review: HumanReviewedRisk,
    context: ContextBundle | None = None,
) -> dict[str, object]:
    frozen_context = context or _context()
    assessment = _testability_assessment(human_review, frozen_context)
    return {
        "testability_assessment": assessment,
        "context": frozen_context,
        "comparison_bindings": _comparison_bindings(),
        "evidence_envelope": _gherkin_envelope(
            human_review,
            frozen_context,
            assessment,
        ),
    }


def test_gherkin_stage_uses_the_union_of_required_visible_evidence() -> None:
    """Removing any reviewed or testability citation would weaken the scenario."""
    human_review = _human_review()
    context = _context()
    assessment = _testability_assessment(human_review, context)

    result = gherkin_generation.build_gherkin_evidence(
        human_review=human_review,
        testability_assessment=assessment,
        context=context,
        comparison_bindings=_comparison_bindings(),
        budget=ProviderRequestBudget(
            provider="groq",
            model="openai/gpt-oss-120b",
            max_body_bytes=7_000,
        ),
    )

    assert result.request_body_bytes <= result.envelope.max_request_body_bytes
    assert result.envelope.stage == "gherkin_generation"
    assert {anchor.anchor_id for anchor in result.envelope.visible_anchors} == {
        "anchor-integration"
    }
    assert result.request.payload["evidence_envelope"] == result.envelope.model_dump(
        mode="json"
    )


def test_gherkin_stage_fails_locally_when_required_whole_evidence_cannot_fit() -> None:
    """A required testability anchor may not be sliced or silently omitted."""
    human_review = _human_review()
    context = _context()
    text = "void purgePatient() { deletePatient(); } " + ("x" * 5_000)
    anchor = context.anchors[0].model_copy(
        update={
            "end_line": 1,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    )
    large_context = ContextBundle.from_content(
        **context.model_dump(
            mode="python",
            exclude={"anchors", "context_sha256", "selected_bytes"},
        ),
        anchors=(anchor,),
        selected_bytes=len(text.encode("utf-8")),
    )
    assessment = _testability_assessment(human_review, large_context)

    with pytest.raises(ModelEvidenceBudgetError, match="anchor-integration"):
        gherkin_generation.build_gherkin_evidence(
            human_review=human_review,
            testability_assessment=assessment,
            context=large_context,
            comparison_bindings=_comparison_bindings(),
            budget=ProviderRequestBudget(
                provider="groq",
                model="openai/gpt-oss-120b",
                max_body_bytes=7_000,
            ),
        )


def test_gherkin_draft_records_the_exact_visible_envelope_hash() -> None:
    """A candidate must identify the frozen evidence visible during generation."""
    payload = _candidate_response(_human_review())
    payload["evidence_envelope_sha256"] = "5" * 64

    draft = GherkinCandidateDraft.model_validate(payload)

    assert draft.evidence_envelope_sha256 == "5" * 64


def test_gherkin_request_is_bound_to_the_human_review() -> None:
    """The request uses one approved risk and locally testable frozen evidence."""
    human_review = _human_review()
    context = _context()
    assessment = _testability_assessment(human_review, context)
    envelope = _gherkin_envelope(human_review, context, assessment)

    request = build_gherkin_request(
        human_review=human_review,
        testability_assessment=assessment,
        context=context,
        comparison_bindings=_comparison_bindings(),
        evidence_envelope=envelope,
    )

    assert request.purpose == "gherkin_generation"
    assert request.payload["snapshot_key"] == human_review.snapshot_key
    assert request.payload["reviewed_risk_sha256"] == canonical_sha256(
        human_review.reviewed_risk.model_dump(mode="json")
    )
    assert request.payload["testability_assessment_sha256"] == (
        assessment.assessment_sha256
    )
    assert request.payload["evidence_envelope"] == envelope.model_dump(mode="json")
    assert request.output_schema["additionalProperties"] is False


def _candidate_response(
    human_review: HumanReviewedRisk,
    evidence_envelope_sha256: str = "5" * 64,
    *,
    include_approved_risk: bool = True,
) -> dict[str, object]:
    """Return one locally replayed structured Gherkin response."""
    approved_risk = human_review.reviewed_risk
    context = _context()
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

    result = {
        "snapshot_key": human_review.snapshot_key,
        "context_sha256": context.context_sha256,
        "reviewed_risk_sha256": human_review.reviewed_content_sha256,
        "evidence_envelope_sha256": evidence_envelope_sha256,
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
        "step_evidence_bindings": [
            {
                "step_number": number,
                "anchor_ids": ["anchor-integration"],
            }
            for number in range(1, 7)
        ],
        "testability_notes": ["Run the scenario against an OpenMRS test environment."],
        "setup_gaps": [],
        "generated_at": "2026-08-12T00:00:00Z",
    }
    if not include_approved_risk:
        result.pop("approved_risk")
    return result


def test_generate_gherkin_returns_a_locally_identified_candidate() -> None:
    """A schema-valid replay response becomes a deterministic candidate."""
    human_review = _human_review()
    context = _context()
    assessment = _testability_assessment(human_review, context)
    envelope = _gherkin_envelope(human_review, context, assessment)
    gateway = ReplayGateway(
        {
            "gherkin_generation": _candidate_response(
                human_review,
                envelope.envelope_sha256,
                include_approved_risk=False,
            ),
        }
    )

    candidate, response = generate_gherkin(
        human_review=human_review,
        testability_assessment=assessment,
        context=context,
        comparison_bindings=_comparison_bindings(),
        evidence_envelope=envelope,
        gateway=gateway,
    )

    assert candidate.snapshot_key == human_review.snapshot_key
    assert candidate.reviewed_risk_sha256 == human_review.reviewed_content_sha256
    assert candidate.approved_risk == human_review.reviewed_risk
    assert candidate.candidate_id.startswith("gherkin-")
    assert response.provider == "replay"


def _candidate(human_review: HumanReviewedRisk):
    """Return one locally validated candidate from the replay fixture."""
    context = _context()
    assessment = _testability_assessment(human_review, context)
    envelope = _gherkin_envelope(human_review, context, assessment)
    candidate, _ = generate_gherkin(
        human_review=human_review,
        testability_assessment=assessment,
        context=context,
        comparison_bindings=_comparison_bindings(),
        evidence_envelope=envelope,
        gateway=ReplayGateway(
            {
                "gherkin_generation": _candidate_response(
                    human_review,
                    envelope.envelope_sha256,
                    include_approved_risk=False,
                ),
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
            **_gherkin_boundary(human_review),
        )


def test_approve_gherkin_binds_exact_candidate_and_review_hashes() -> None:
    """Human approval records immutable identities for later freshness checks."""
    human_review = _human_review()
    candidate = _candidate(human_review)

    approval = approve_gherkin(
        candidate=candidate,
        human_review=human_review,
        **_gherkin_boundary(human_review),
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
        **_gherkin_boundary(human_review),
    )

    assert report.approved is True
    assert report.decision == "valid_evidence_bound_gherkin"
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
            **_gherkin_boundary(human_review),
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
            **_gherkin_boundary(human_review),
        )


def test_gherkin_request_rejects_a_tampered_human_review() -> None:
    """A changed review hash cannot be used to request a new scenario."""
    human_review = _human_review()
    context = _context()
    assessment = _testability_assessment(human_review, context)
    tampered_review = human_review.model_copy(
        update={"reviewed_content_sha256": "f" * 64}
    )

    with pytest.raises(ValueError, match="human review"):
        build_gherkin_request(
            human_review=tampered_review,
            testability_assessment=assessment,
            context=context,
            comparison_bindings=_comparison_bindings(),
            evidence_envelope=_gherkin_envelope(
                human_review,
                context,
                assessment,
            ),
        )


def test_candidate_records_current_context_and_evidence_for_every_step() -> None:
    """Every generated Gherkin step must name frozen evidence in its context."""
    human_review = _human_review()
    context = _context()
    response_data = _candidate_response(human_review)
    response_data["context_sha256"] = context.context_sha256
    response_data["step_evidence_bindings"] = [
        {
            "step_number": number,
            "anchor_ids": ["anchor-integration"],
        }
        for number in range(1, 7)
    ]

    candidate = GherkinCandidateDraft.model_validate(response_data)

    assert candidate.context_sha256 == context.context_sha256
    assert candidate.step_evidence_bindings == (
        GherkinStepEvidenceBinding(
            step_number=1,
            anchor_ids=("anchor-integration",),
        ),
        GherkinStepEvidenceBinding(
            step_number=2,
            anchor_ids=("anchor-integration",),
        ),
        GherkinStepEvidenceBinding(
            step_number=3,
            anchor_ids=("anchor-integration",),
        ),
        GherkinStepEvidenceBinding(
            step_number=4,
            anchor_ids=("anchor-integration",),
        ),
        GherkinStepEvidenceBinding(
            step_number=5,
            anchor_ids=("anchor-integration",),
        ),
        GherkinStepEvidenceBinding(
            step_number=6,
            anchor_ids=("anchor-integration",),
        ),
    )


def test_candidate_rejects_missing_step_evidence() -> None:
    """A scenario cannot leave one of its steps without frozen-code support."""
    human_review = _human_review()
    context = _context()
    response_data = _candidate_response(human_review)
    response_data["context_sha256"] = context.context_sha256
    response_data["step_evidence_bindings"] = [
        {
            "step_number": 1,
            "anchor_ids": ["anchor-integration"],
        }
    ]

    with pytest.raises(ValidationError, match="step evidence bindings"):
        GherkinCandidateDraft.model_validate(response_data)


def test_edited_gherkin_classifies_evidence_bound_scenario_as_valid() -> None:
    """An unchanged scenario remains valid for its exact frozen context."""
    human_review = _human_review()
    context = _context()
    candidate = _candidate(human_review)

    report = validate_edited_gherkin(
        candidate=candidate,
        text=candidate.gherkin_text,
        human_review=human_review,
        **_gherkin_boundary(human_review, context),
    )

    assert report.approved is True
    assert report.decision == "valid_evidence_bound_gherkin"
    assert report.reason_codes == ()


def test_gherkin_validation_rejects_a_catalog_anchor_hidden_from_the_model() -> None:
    """Step evidence must resolve inside the exact Gherkin visibility boundary."""
    human_review = _human_review()
    context = _context_with_hidden_integration_anchor()
    assessment = _testability_assessment(human_review, context)
    envelope = _gherkin_envelope(human_review, context, assessment)
    candidate = GherkinCandidateDraft.model_validate(
        {
            **_candidate_response(human_review, envelope.envelope_sha256),
            "context_sha256": context.context_sha256,
            "step_evidence_bindings": [
                {
                    "step_number": number,
                    "anchor_ids": ["anchor-hidden-integration"],
                }
                for number in range(1, 7)
            ],
        }
    )

    report = validate_gherkin_candidate(
        candidate=gherkin_generation.GherkinCandidate.from_draft(candidate),
        human_review=human_review,
        testability_assessment=assessment,
        context=context,
        comparison_bindings=_comparison_bindings(),
        evidence_envelope=envelope,
    )

    assert report.approved is False
    assert "unknown_step_evidence_anchor" in report.reason_codes


def test_edited_gherkin_classifies_removed_failure_oracle_as_hypothesis_changed() -> (
    None
):
    """Removing a required security outcome changes the approved risk idea."""
    human_review = _human_review()
    context = _context()
    candidate = _candidate(human_review)
    edited_text = candidate.gherkin_text.replace(
        human_review.reviewed_risk.possible_failure,
        "the request completes normally",
    )

    report = validate_edited_gherkin(
        candidate=candidate,
        text=edited_text,
        human_review=human_review,
        **_gherkin_boundary(human_review, context),
    )

    assert report.approved is False
    assert report.decision == "hypothesis_changed"
    assert "bound_risk_term_removed" in report.reason_codes


def test_edited_gherkin_classifies_new_unbound_identifier_as_needing_evidence() -> None:
    """A new route-like code identifier needs saved code evidence before approval."""
    human_review = _human_review()
    context = _context()
    candidate = _candidate(human_review)
    edited_text = candidate.gherkin_text.replace(
        "When the user requests deletePatient for a patient record",
        (
            "When the user requests deletePatient through "
            "authorizePatientDeletion for a patient record"
        ),
    )

    report = validate_edited_gherkin(
        candidate=candidate,
        text=edited_text,
        human_review=human_review,
        **_gherkin_boundary(human_review, context),
    )

    assert report.approved is False
    assert report.decision == "needs_more_frozen_evidence"
    assert "unbound_code_identifier" in report.reason_codes


def test_edited_gherkin_classifies_executable_content_as_invalid() -> None:
    """Executable-looking content cannot be smuggled into a scenario step."""
    human_review = _human_review()
    context = _context()
    candidate = _candidate(human_review)
    edited_text = candidate.gherkin_text.replace(
        human_review.reviewed_risk.expected_secure_behavior,
        human_review.reviewed_risk.expected_secure_behavior + " and import os",
    )

    report = validate_edited_gherkin(
        candidate=candidate,
        text=edited_text,
        human_review=human_review,
        **_gherkin_boundary(human_review, context),
    )

    assert report.approved is False
    assert report.decision == "invalid_gherkin"
    assert "gherkin_text_contains_implementation_code" in report.reason_codes
