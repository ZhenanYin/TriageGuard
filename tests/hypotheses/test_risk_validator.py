"""Tests for local validation of model-proposed risk hypotheses."""

import hashlib
from datetime import UTC, datetime

import pytest

from triageguard.domain.pr_analysis import (
    ClaimEvidenceBinding,
    ContextAnchor,
    ContextBundle,
    PullRequestSnapshot,
    RiskAssessmentDraft,
    RiskHypothesisDraft,
)
from triageguard.evidence import (
    EvidenceArtifactBinding,
    ModelEvidenceEnvelope,
    OmittedEvidenceAnchor,
    VisibleEvidenceAnchor,
)
from triageguard.hypotheses.generator import RISK_OUTPUT_SCHEMA
from triageguard.hypotheses.validator import (
    create_human_review,
)
from triageguard.hypotheses.validator import (
    validate_risk_assessment as _validate_risk_assessment,
)
from triageguard.provenance import canonical_sha256


def _snapshot() -> PullRequestSnapshot:
    """Return one frozen OpenMRS Core pull-request snapshot."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=7312,
        pull_url="https://github.com/openmrs/openmrs-core/pull/7312",
        state="open",
        default_branch="main",
        base_branch="main",
        merge_base_sha="a" * 40,
        base_sha="b" * 40,
        head_sha="c" * 40,
        candidate_sha="d" * 40,
        merge_base_tree_sha="e" * 40,
        base_tree_sha="f" * 40,
        head_tree_sha="1" * 40,
        candidate_tree_sha="2" * 40,
        acquired_at=datetime(2026, 8, 12, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="3" * 64,
    )


def _context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Return one immutable integration-change code excerpt."""
    text = "void purgePatient() {\n    dao.deletePatient();\n}\n"
    anchor = ContextAnchor(
        anchor_id="anchor-integration",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="4" * 40,
        path="api/PatientService.java",
        java_symbol="purgePatient",
        start_line=3,
        end_line=5,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        selection_reason="integration hunk",
        score_components=[],
        change_relation="integration_change",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(anchor,),
        selected_file_count=1,
        selected_anchor_count=1,
        selected_bytes=len(text.encode("utf-8")),
        max_files=40,
        max_anchors=80,
        max_bytes=160_000,
        max_anchor_lines=120,
        max_blob_bytes=1_000_000,
        max_search_identifiers=100,
        max_hits_per_identifier=20,
        primary_change_represented=True,
    )


def _envelope(
    context: ContextBundle,
    *,
    visible_anchor_ids: tuple[str, ...] | None = None,
) -> ModelEvidenceEnvelope:
    """Create the immutable model-visible partition used by validator tests."""
    visible_ids = set(
        visible_anchor_ids
        if visible_anchor_ids is not None
        else (anchor.anchor_id for anchor in context.anchors)
    )
    return ModelEvidenceEnvelope.from_content(
        stage="risk_hypothesis",
        snapshot_key=context.snapshot_key,
        context_sha256=context.context_sha256,
        comparison_bindings=tuple(
            EvidenceArtifactBinding(name=name, sha256=digest)
            for name, digest in (
                ("author_diff", "a" * 64),
                ("base_drift_diff", "b" * 64),
                ("integration_diff", "c" * 64),
            )
        ),
        input_bindings=(),
        visible_anchors=tuple(
            VisibleEvidenceAnchor.from_context_anchor(anchor).model_copy(
                update={"selection_reason": "required_by_stage"}
            )
            for anchor in context.anchors
            if anchor.anchor_id in visible_ids
        ),
        omitted_anchors=tuple(
            OmittedEvidenceAnchor(
                anchor_id=anchor.anchor_id,
                reason="request_budget",
            )
            for anchor in context.anchors
            if anchor.anchor_id not in visible_ids
        ),
        catalog_anchor_ids=tuple(anchor.anchor_id for anchor in context.anchors),
        max_request_body_bytes=7_000,
        selection_policy_version="risk-evidence-v1",
        output_schema_sha256=canonical_sha256(RISK_OUTPUT_SCHEMA),
    )


def validate_risk_assessment(
    *,
    draft: RiskAssessmentDraft,
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    evidence_envelope: ModelEvidenceEnvelope | None = None,
):
    """Keep fixtures concise while every validation uses an explicit envelope."""
    return _validate_risk_assessment(
        draft=draft,
        snapshot=snapshot,
        context=context,
        evidence_envelope=evidence_envelope or _envelope(context),
    )


def _risk_draft(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
) -> RiskAssessmentDraft:
    """Return one structurally valid but still unvalidated model proposal."""
    hypothesis = RiskHypothesisDraft(
        claim_status="unconfirmed_risk_hypothesis",
        title="Patient deletion may bypass an expected authorization check",
        explanation=(
            "The integration excerpt changes a deletion call, so the surrounding "
            "authorization behavior needs an executable check."
        ),
        actor="An authenticated OpenMRS user",
        preconditions=("The user can reach the patient-deletion service path.",),
        action="The user requests deletion of a patient record.",
        protected_asset="Patient records",
        security_property="Authorization",
        expected_secure_behavior=(
            "The deletion path enforces the required authorization before deleting "
            "a patient record."
        ),
        possible_failure=(
            "The changed path may delete a patient record without the expected "
            "authorization enforcement."
        ),
        observables=("The deletion attempt is rejected when authorization is absent.",),
        code_identifiers=("deletePatient",),
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
        ),
        limitations=(
            "The excerpt does not show the surrounding authorization checks.",
        ),
        missing_evidence=(
            "The relevant authorization implementation is not in the context.",
        ),
        priority_rationale=(
            "Patient deletion is security-relevant and needs human review."
        ),
    )
    return RiskAssessmentDraft(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        evidence_envelope_sha256=_envelope(context).envelope_sha256,
        outcome="risks_proposed",
        hypotheses=(hypothesis,),
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
    )


def test_validator_rejects_an_anchor_that_exists_but_was_hidden_from_the_model() -> (
    None
):
    """Frozen-but-omitted evidence cannot retroactively legitimize a model claim."""
    snapshot = _snapshot()
    visible = _context(snapshot).anchors[0]
    hidden_text = "void hiddenAuthorizationBypass() {}\n"
    hidden = ContextAnchor(
        anchor_id="anchor-hidden",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="5" * 40,
        path="api/HiddenService.java",
        java_symbol="hiddenAuthorizationBypass",
        start_line=1,
        end_line=1,
        text=hidden_text,
        text_sha256=hashlib.sha256(hidden_text.encode()).hexdigest(),
        selection_reason="repository context",
        score_components=(),
        change_relation="repository_context",
        truncated=False,
    )
    context = ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(visible, hidden),
        selected_file_count=2,
        selected_anchor_count=2,
        selected_bytes=len(visible.text.encode()) + len(hidden_text.encode()),
        max_files=40,
        max_anchors=80,
        max_bytes=160_000,
        max_anchor_lines=120,
        max_blob_bytes=1_000_000,
        max_search_identifiers=100,
        max_hits_per_identifier=20,
        primary_change_represented=True,
    )
    envelope = _envelope(context, visible_anchor_ids=(visible.anchor_id,))
    draft = _risk_draft(snapshot, context)
    hypothesis = draft.hypotheses[0]
    hidden_bindings = tuple(
        binding.model_copy(update={"anchor_ids": (hidden.anchor_id,)})
        for binding in hypothesis.evidence_bindings
    )
    hidden_hypothesis = hypothesis.model_copy(
        update={
            "code_identifiers": ("hiddenAuthorizationBypass",),
            "evidence_bindings": hidden_bindings,
        }
    )
    hidden_draft = draft.model_copy(
        update={
            "evidence_envelope_sha256": envelope.envelope_sha256,
            "hypotheses": (hidden_hypothesis,),
        }
    )

    assessment, report = validate_risk_assessment(
        draft=hidden_draft,
        snapshot=snapshot,
        context=context,
        evidence_envelope=envelope,
    )

    assert assessment is None
    assert "unknown_evidence_anchor" in report.reason_codes


def test_validator_rejects_a_tampered_evidence_envelope_before_grounding() -> None:
    """Changing envelope content without recomputing its identity fails closed."""
    snapshot = _snapshot()
    context = _context(snapshot)
    envelope = _envelope(context)
    tampered = envelope.model_copy(
        update={"max_request_body_bytes": envelope.max_request_body_bytes + 1}
    )

    assessment, report = validate_risk_assessment(
        draft=_risk_draft(snapshot, context),
        snapshot=snapshot,
        context=context,
        evidence_envelope=tampered,
    )

    assert assessment is None
    assert report.reason_codes == ("invalid_evidence_envelope",)


def test_hallucinated_anchor_is_rejected() -> None:
    """A risk cannot cite an anchor that is absent from frozen evidence."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = _risk_draft(snapshot, context)
    hypothesis = draft.hypotheses[0]

    changed_binding = hypothesis.evidence_bindings[0].model_copy(
        update={"anchor_ids": ("anchor-does-not-exist",)}
    )
    changed_hypothesis = hypothesis.model_copy(
        update={
            "evidence_bindings": (
                changed_binding,
                *hypothesis.evidence_bindings[1:],
            )
        }
    )
    changed_draft = draft.model_copy(update={"hypotheses": (changed_hypothesis,)})

    assessment, report = validate_risk_assessment(
        draft=changed_draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is None
    assert report.approved is False
    assert "unknown_evidence_anchor" in report.reason_codes


def test_valid_grounded_proposal_receives_a_local_risk_id() -> None:
    """A fully grounded model proposal becomes a locally validated assessment."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = _risk_draft(snapshot, context)

    assessment, report = validate_risk_assessment(
        draft=draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is not None
    assert report.approved is True
    assert report.reason_codes == ()
    assert len(report.validated_hypothesis_ids) == 1
    assert assessment.hypotheses[0].hypothesis_id.startswith("risk-")
    assert report.validated_hypothesis_ids == (assessment.hypotheses[0].hypothesis_id,)


def test_human_edit_preserves_model_proposal_and_records_delta() -> None:
    """A review creates an immutable successor without changing model output."""
    snapshot = _snapshot()
    context = _context(snapshot)
    assessment, report = validate_risk_assessment(
        draft=_risk_draft(snapshot, context),
        snapshot=snapshot,
        context=context,
    )

    assert assessment is not None
    assert report.approved is True

    original = assessment.hypotheses[0]
    reviewed = create_human_review(
        assessment=assessment,
        hypothesis_id=original.hypothesis_id,
        edits={
            "expected_secure_behavior": (
                "The API rejects deletion and the patient remains stored."
            )
        },
        selected_anchor_ids=original.citation_anchor_ids,
        reviewed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assert (
        original.expected_secure_behavior
        != reviewed.reviewed_risk.expected_secure_behavior
    )
    assert reviewed.selected_hypothesis_sha256 == canonical_sha256(
        original.model_dump(mode="json")
    )
    assert reviewed.reviewed_content_sha256 == canonical_sha256(
        reviewed.reviewed_risk.model_dump(mode="json")
    )
    assert [change.field_name for change in reviewed.field_changes] == [
        "expected_secure_behavior"
    ]


def test_human_review_rejects_noneditable_fields() -> None:
    """A reviewer may refine approved fields, not rewrite the model proposal."""
    snapshot = _snapshot()
    context = _context(snapshot)
    assessment, _ = validate_risk_assessment(
        draft=_risk_draft(snapshot, context),
        snapshot=snapshot,
        context=context,
    )

    assert assessment is not None
    original = assessment.hypotheses[0]

    with pytest.raises(ValueError, match="approved editable fields"):
        create_human_review(
            assessment=assessment,
            hypothesis_id=original.hypothesis_id,
            edits={"title": "A reviewer cannot replace the model title."},
            selected_anchor_ids=original.citation_anchor_ids,
            reviewed_at=datetime(2026, 8, 12, tzinfo=UTC),
        )


def test_human_review_rejects_anchors_outside_frozen_context() -> None:
    """A review cannot cite a code excerpt that was never frozen for this PR."""
    snapshot = _snapshot()
    context = _context(snapshot)
    assessment, _ = validate_risk_assessment(
        draft=_risk_draft(snapshot, context),
        snapshot=snapshot,
        context=context,
    )

    assert assessment is not None
    original = assessment.hypotheses[0]

    with pytest.raises(ValueError, match="exist in frozen context"):
        create_human_review(
            assessment=assessment,
            hypothesis_id=original.hypothesis_id,
            edits={},
            selected_anchor_ids=("anchor-not-in-context",),
            reviewed_at=datetime(2026, 8, 12, tzinfo=UTC),
        )


def test_validator_rejects_a_tampered_context_bundle() -> None:
    """A changed context bundle cannot be treated as the originally frozen evidence."""
    snapshot = _snapshot()
    context = _context(snapshot)
    tampered_context = context.model_copy(update={"max_bytes": context.max_bytes + 1})

    assessment, report = validate_risk_assessment(
        draft=_risk_draft(snapshot, context),
        snapshot=snapshot,
        context=tampered_context,
    )

    assert assessment is None
    assert report.approved is False
    assert "invalid_context_bundle" in report.reason_codes


def test_validator_rejects_a_tampered_snapshot() -> None:
    """A changed frozen revision cannot retain the old snapshot identity."""
    snapshot = _snapshot()
    context = _context(snapshot)
    tampered_snapshot = snapshot.model_copy(update={"base_sha": "f" * 40})

    assessment, report = validate_risk_assessment(
        draft=_risk_draft(snapshot, context),
        snapshot=tampered_snapshot,
        context=context,
    )

    assert assessment is None
    assert report.approved is False
    assert "invalid_snapshot" in report.reason_codes


def test_validator_rejects_an_identifier_missing_from_bound_evidence() -> None:
    """A proposed code identifier must occur in its cited frozen excerpt."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = _risk_draft(snapshot, context)
    changed_hypothesis = draft.hypotheses[0].model_copy(
        update={"code_identifiers": ("identifierNotInTheExcerpt",)}
    )
    changed_draft = draft.model_copy(update={"hypotheses": (changed_hypothesis,)})

    assessment, report = validate_risk_assessment(
        draft=changed_draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is None
    assert report.approved is False
    assert "identifier_not_in_bound_excerpt" in report.reason_codes


def test_no_risk_outcome_rejects_an_unknown_supporting_anchor() -> None:
    """An abstention is evidence-bound and is never a proof of safety."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = RiskAssessmentDraft(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        evidence_envelope_sha256=_envelope(context).envelope_sha256,
        outcome="no_meaningful_security_risk_found",
        rationale=(
            "The bounded evidence does not show a specific testable security-risk "
            "hypothesis."
        ),
        security_relevant_areas=("Patient deletion service behavior.",),
        supporting_anchor_ids=("anchor-not-in-context",),
        coverage_limitations=(
            "This is not proof of safety because the evidence is bounded.",
        ),
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assessment, report = validate_risk_assessment(
        draft=draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is None
    assert report.approved is False
    assert "unknown_evidence_anchor" in report.reason_codes


@pytest.mark.parametrize(
    "invalid_kind",
    [
        "missing_binding",
        "duplicate_citation",
        "decisive_safety_claim",
        "cvss_claim",
        "invalid_no_risk_limitation",
        "invalid_insufficient_context_reason",
    ],
)
def test_invalid_model_contract_drafts_are_never_approved(
    invalid_kind: str,
) -> None:
    """Bypassed model-contract validation still cannot produce an assessment."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = _risk_draft(snapshot, context)
    hypothesis = draft.hypotheses[0]

    if invalid_kind == "missing_binding":
        changed_hypothesis = hypothesis.model_copy(
            update={"evidence_bindings": hypothesis.evidence_bindings[1:]}
        )
        changed_draft = draft.model_copy(update={"hypotheses": (changed_hypothesis,)})
    elif invalid_kind == "duplicate_citation":
        changed_binding = hypothesis.evidence_bindings[0].model_copy(
            update={
                "anchor_ids": (
                    "anchor-integration",
                    "anchor-integration",
                )
            }
        )
        changed_hypothesis = hypothesis.model_copy(
            update={
                "evidence_bindings": (
                    changed_binding,
                    *hypothesis.evidence_bindings[1:],
                )
            }
        )
        changed_draft = draft.model_copy(update={"hypotheses": (changed_hypothesis,)})
    elif invalid_kind == "decisive_safety_claim":
        changed_hypothesis = hypothesis.model_copy(
            update={"explanation": "This change is confirmed safe."}
        )
        changed_draft = draft.model_copy(update={"hypotheses": (changed_hypothesis,)})
    elif invalid_kind == "cvss_claim":
        changed_hypothesis = hypothesis.model_copy(update={"explanation": "CVSS: 9.0"})
        changed_draft = draft.model_copy(update={"hypotheses": (changed_hypothesis,)})
    elif invalid_kind == "invalid_no_risk_limitation":
        changed_draft = draft.model_copy(
            update={
                "outcome": "no_meaningful_security_risk_found",
                "hypotheses": (),
                "rationale": "The bounded evidence has no specific testable risk.",
                "security_relevant_areas": ("Patient deletion service behavior.",),
                "coverage_limitations": ("Only bounded evidence was reviewed.",),
            }
        )
    elif invalid_kind == "invalid_insufficient_context_reason":
        changed_draft = draft.model_copy(
            update={
                "outcome": "insufficient_context_to_assess",
                "hypotheses": (),
                "reason_code": "invented_reason_code",
                "missing_evidence": ("The relevant authorization implementation.",),
                "needed_evidence": (
                    "The relevant authorization implementation and tests.",
                ),
            }
        )
    else:
        raise ValueError("test received an unsupported invalid draft kind")

    assessment, report = validate_risk_assessment(
        draft=changed_draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is None
    assert report.approved is False
    assert report.reason_codes == ("invalid_risk_assessment_draft",)


def test_valid_no_risk_outcome_remains_bound_to_integration_evidence() -> None:
    """A no-risk conclusion is valid only when it cites frozen merge evidence."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = RiskAssessmentDraft(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        evidence_envelope_sha256=_envelope(context).envelope_sha256,
        outcome="no_meaningful_security_risk_found",
        rationale=(
            "The bounded evidence does not show a specific testable security-risk "
            "hypothesis."
        ),
        security_relevant_areas=("Patient deletion service behavior.",),
        supporting_anchor_ids=("anchor-integration",),
        coverage_limitations=(
            "This is not proof of safety because the evidence is bounded.",
        ),
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assessment, report = validate_risk_assessment(
        draft=draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is not None
    assert assessment.outcome == "no_meaningful_security_risk_found"
    assert report.approved is True
    assert report.validated_hypothesis_ids == ()


def test_valid_insufficient_context_outcome_is_retained() -> None:
    """Insufficient evidence is recorded as an explicit valid outcome."""
    snapshot = _snapshot()
    context = _context(snapshot)
    draft = RiskAssessmentDraft(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        evidence_envelope_sha256=_envelope(context).envelope_sha256,
        outcome="insufficient_context_to_assess",
        reason_code="analysis_limit_exceeded",
        missing_evidence=(
            "The relevant authorization implementation exceeds the approved context limit.",
        ),
        needed_evidence=(
            "A bounded excerpt containing the authorization implementation.",
        ),
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assessment, report = validate_risk_assessment(
        draft=draft,
        snapshot=snapshot,
        context=context,
    )

    assert assessment is not None
    assert assessment.outcome == "insufficient_context_to_assess"
    assert report.approved is True
    assert report.validated_hypothesis_ids == ()
