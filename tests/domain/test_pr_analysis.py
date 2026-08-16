"""Contracts for immutable Milestone 2 PR-analysis artifacts."""

import hashlib
import warnings
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from triageguard.domain.pr_analysis import (
    ClaimEvidenceBinding,
    ContextAnchor,
    ContextBundle,
    ContextRefinement,
    FrozenEvidenceNeed,
    GherkinApproval,
    GherkinCandidate,
    GherkinCandidateDraft,
    GherkinStep,
    GherkinStepBinding,
    GherkinStepEvidenceBinding,
    GroundingReport,
    HumanReviewedRisk,
    IdentifierEvidence,
    MilestoneTwoRunRecord,
    PullRequestSnapshot,
    RiskAssessment,
    RiskAssessmentDraft,
    RiskHypothesis,
    RiskHypothesisDraft,
    SnapshotFreshness,
)
from triageguard.domain.pr_analysis import (
    TestabilityAssessment as Assessment,
)
from triageguard.provenance import canonical_sha256


def test_pr_analysis_contracts_are_available_from_the_domain_package() -> None:
    """Later workflow components should depend on the stable domain boundary."""
    import triageguard.domain as public_domain

    assert public_domain.PullRequestSnapshot is PullRequestSnapshot
    assert all(
        contract is not None
        for contract in (
            public_domain.ContextBundle,
            public_domain.DiffArtifact,
            public_domain.GherkinApproval,
            public_domain.GherkinCandidate,
            public_domain.HumanReviewedRisk,
            public_domain.MilestoneTwoRunRecord,
            public_domain.RiskAssessment,
            public_domain.RiskAssessmentDraft,
            public_domain.SnapshotFreshness,
        )
    )


def snapshot_payload(**changes: object) -> dict[str, object]:
    """Return independent, canonical values for a frozen PR snapshot."""
    payload: dict[str, object] = {
        "snapshot_key": "0" * 64,
        "repository": "openmrs/openmrs-core",
        "pull_number": 123,
        "pull_url": "https://github.com/openmrs/openmrs-core/pull/123",
        "state": "open",
        "default_branch": "main",
        "base_branch": "main",
        "merge_base_sha": "1" * 40,
        "base_sha": "2" * 40,
        "head_sha": "3" * 40,
        "candidate_sha": "4" * 40,
        "merge_base_tree_sha": "5" * 40,
        "base_tree_sha": "6" * 40,
        "head_tree_sha": "7" * 40,
        "candidate_tree_sha": "8" * 40,
        "acquired_at": datetime(2026, 8, 11, tzinfo=UTC),
        "github_api_version": "2026-03-10",
        "git_version": "2.47.1",
        "acquisition_tool_version": "triageguard/2.0.0",
        "analysis_config_sha256": "9" * 64,
    }
    payload.update(changes)
    payload["snapshot_key"] = canonical_sha256(
        {
            key: payload[key]
            for key in (
                "repository",
                "pull_number",
                "merge_base_sha",
                "base_sha",
                "head_sha",
                "candidate_sha",
                "analysis_config_sha256",
            )
        }
    )
    return payload


def test_snapshot_requires_four_distinct_full_commit_shas() -> None:
    """Short or reused frozen revisions would make a PR analysis non-reproducible."""
    with pytest.raises(ValidationError, match="full 40-character"):
        PullRequestSnapshot.model_validate(snapshot_payload(base_sha="abc1234"))

    with pytest.raises(ValidationError, match="distinct"):
        PullRequestSnapshot.model_validate(snapshot_payload(candidate_sha="2" * 40))


def test_snapshot_rejects_non_utc_acquisition_time() -> None:
    """A local timestamp would make the frozen acquisition order ambiguous."""
    with pytest.raises(ValidationError, match="UTC"):
        PullRequestSnapshot.model_validate(
            snapshot_payload(
                acquired_at=datetime(2026, 8, 11, tzinfo=timezone(timedelta(hours=1)))
            )
        )


def test_risk_hypothesis_derives_stable_unique_citation_ids() -> None:
    """A duplicate citation list must not be persisted separately from bindings."""
    hypothesis = RiskHypothesis.from_draft(_risk_draft())

    assert hypothesis.citation_anchor_ids == ["anchor-b", "anchor-a", "anchor-c"]
    assert (
        hypothesis.hypothesis_id
        == RiskHypothesis.from_draft(_risk_draft()).hypothesis_id
    )


def _risk_draft(**changes: object) -> RiskHypothesisDraft:
    payload: dict[str, object] = {
        "claim_status": "unconfirmed_risk_hypothesis",
        "title": "Authorization check may be bypassed",
        "explanation": "The integration change changes a privilege boundary.",
        "actor": "authenticated clerk",
        "preconditions": ["A protected record exists."],
        "action": "Submit the changed endpoint request.",
        "protected_asset": "Protected patient record",
        "security_property": "Authorization is enforced.",
        "expected_secure_behavior": "The request is denied.",
        "possible_failure": "The request succeeds without the privilege.",
        "observables": ["HTTP response", "Persistent record state"],
        "code_identifiers": ["requirePrivilege"],
        "evidence_bindings": [
            ClaimEvidenceBinding(
                claim_field="explanation",
                observable_index=None,
                anchor_ids=["anchor-b"],
            ),
            ClaimEvidenceBinding(
                claim_field="actor", observable_index=None, anchor_ids=["anchor-b"]
            ),
            ClaimEvidenceBinding(
                claim_field="action", observable_index=None, anchor_ids=["anchor-a"]
            ),
            ClaimEvidenceBinding(
                claim_field="expected_secure_behavior",
                observable_index=None,
                anchor_ids=["anchor-b", "anchor-c"],
            ),
            ClaimEvidenceBinding(
                claim_field="possible_failure",
                observable_index=None,
                anchor_ids=["anchor-c"],
            ),
            ClaimEvidenceBinding(
                claim_field="observable", observable_index=0, anchor_ids=["anchor-a"]
            ),
            ClaimEvidenceBinding(
                claim_field="observable", observable_index=1, anchor_ids=["anchor-c"]
            ),
        ],
        "limitations": ["Only bounded context was reviewed."],
        "missing_evidence": [],
        "priority_rationale": "The changed check protects patient data.",
    }
    payload.update(changes)
    return RiskHypothesisDraft.model_validate(payload)


def test_model_cannot_supply_hypothesis_id_or_prohibited_claims() -> None:
    """Provider output must remain an unconfirmed hypothesis with a local identity."""
    with pytest.raises(ValidationError, match="locally derived"):
        RiskHypothesis.model_validate(
            {**_risk_draft().model_dump(), "hypothesis_id": "provider-id"}
        )

    for forbidden in (
        "CVSS:4.0/AV:N",
        "This is a confirmed vulnerability.",
        "This change is confirmed safe.",
    ):
        with pytest.raises(ValidationError, match="prohibited"):
            _risk_draft(explanation=forbidden)

    derived = RiskHypothesis.from_draft(_risk_draft())
    with pytest.raises(ValidationError, match="locally derived"):
        RiskHypothesis.model_validate(derived.model_dump(mode="json"))


def test_artifact_collections_are_deeply_immutable() -> None:
    """Recorded evidence cannot be changed through a mutable nested collection."""
    binding = ClaimEvidenceBinding(
        claim_field="actor", observable_index=None, anchor_ids=["anchor-a"]
    )

    assert binding.anchor_ids == ("anchor-a",)
    with pytest.raises(AttributeError):
        binding.anchor_ids.append("anchor-b")  # type: ignore[attr-defined]


def test_context_bundle_requires_all_reproducibility_limits() -> None:
    """Context artifacts must retain every configured selection limit."""
    anchor = ContextAnchor(
        anchor_id="anchor-a",
        revision_role="candidate",
        commit_sha="4" * 40,
        blob_sha="5" * 40,
        path="api/Patient.java",
        java_symbol="Patient.delete",
        start_line=10,
        end_line=11,
        text="requirePrivilege();\nreturn denied;",
        text_sha256=hashlib.sha256(b"requirePrivilege();\nreturn denied;").hexdigest(),
        selection_reason="integration change",
        score_components=[],
        change_relation="integration_change",
        truncated=False,
    )
    payload = {
        "snapshot_key": "0" * 64,
        "anchors": [anchor],
        "selected_file_count": 1,
        "selected_anchor_count": 1,
        "selected_bytes": 20,
        "max_files": 40,
        "max_anchors": 80,
        "max_bytes": 160_000,
        "primary_change_represented": True,
        "context_sha256": "b" * 64,
    }

    with pytest.raises(ValidationError, match="max_anchor_lines"):
        ContextBundle.model_validate(payload)


def test_context_bundle_enforces_anchor_bound_and_exact_truncation_inventory() -> None:
    """A context record cannot contradict line or truncation limits it claims to use."""
    anchor = ContextAnchor(
        anchor_id="anchor-a",
        revision_role="candidate",
        commit_sha="4" * 40,
        blob_sha="5" * 40,
        path="api/Patient.java",
        java_symbol=None,
        start_line=1,
        end_line=3,
        text="abc",
        text_sha256=hashlib.sha256(b"abc").hexdigest(),
        selection_reason="integration change",
        score_components=[],
        change_relation="integration_change",
        truncated=True,
    )
    payload = {
        "snapshot_key": "0" * 64,
        "anchors": [anchor],
        "selected_file_count": 1,
        "selected_anchor_count": 1,
        "selected_bytes": 3,
        "max_files": 1,
        "max_anchors": 1,
        "max_bytes": 3,
        "max_anchor_lines": 2,
        "max_blob_bytes": 10,
        "max_search_identifiers": 1,
        "max_hits_per_identifier": 1,
        "primary_change_represented": True,
        "context_sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="context SHA-256"):
        ContextBundle.model_validate(payload)

    payload["max_anchor_lines"] = 3
    with pytest.raises(ValidationError, match="context SHA-256"):
        ContextBundle.model_validate(payload)


def test_candidate_rejects_non_gherkin_or_incomplete_traceability() -> None:
    """An approval candidate needs parsed Gherkin and every approved-risk binding."""
    approved_risk = RiskHypothesis.from_draft(_risk_draft())
    with pytest.raises(ValidationError, match="Feature"):
        GherkinCandidateDraft(
            snapshot_key="0" * 64,
            context_sha256=_context_bundle().context_sha256,
            reviewed_risk_sha256=canonical_sha256(
                approved_risk.model_dump(mode="json")
            ),
            approved_risk=approved_risk,
            feature_title="Privilege enforcement",
            scenario_title="Unauthorized request is denied",
            steps=[GherkinStep(number=1, keyword="Given", text="a clerk")],
            gherkin_text="not gherkin",
            bindings=[],
            step_evidence_bindings=(
                GherkinStepEvidenceBinding(
                    step_number=1,
                    anchor_ids=("anchor-a",),
                ),
            ),
            testability_notes=[],
            setup_gaps=[],
            generated_at=datetime(2026, 8, 11, tzinfo=UTC),
        )


def test_raw_gherkin_cannot_supply_a_candidate_id_or_code_like_step() -> None:
    """Provider output cannot select a durable ID or hide executable text in a step."""
    draft = _candidate().model_dump(mode="json", exclude={"candidate_id"})
    with pytest.raises(ValidationError, match="locally derived"):
        GherkinCandidate.model_validate(draft | {"candidate_id": "provider-id"})

    malicious = draft | {
        "steps": [
            *draft["steps"][:2],
            {"number": 3, "keyword": "When", "text": "os.system('untrusted')"},
            *draft["steps"][3:],
        ],
        "gherkin_text": draft["gherkin_text"].replace(
            "When the changed request is sent", "When os.system('untrusted')"
        ),
    }
    with pytest.raises(ValidationError, match="implementation code"):
        GherkinCandidateDraft.model_validate(malicious)


def _candidate(snapshot_key: str = "0" * 64) -> GherkinCandidate:
    risk = _risk_draft()
    steps = (
        GherkinStep(number=1, keyword="Given", text="an authenticated clerk"),
        GherkinStep(number=2, keyword="And", text="a protected record exists"),
        GherkinStep(number=3, keyword="When", text="the changed request is sent"),
        GherkinStep(number=4, keyword="Then", text="the request is denied"),
        GherkinStep(number=5, keyword="And", text="a failed request is observable"),
        GherkinStep(number=6, keyword="And", text="the HTTP response is recorded"),
        GherkinStep(
            number=7,
            keyword="And",
            text="the requirePrivilege record state is recorded",
        ),
    )
    return GherkinCandidate.from_draft(
        GherkinCandidateDraft(
            snapshot_key=snapshot_key,
            context_sha256=_context_bundle(snapshot_key).context_sha256,
            reviewed_risk_sha256=canonical_sha256(risk.model_dump(mode="json")),
            approved_risk=risk,
            feature_title="Privilege enforcement",
            scenario_title="Unauthorized request is denied",
            steps=steps,
            gherkin_text="\n".join(
                (
                    "Feature: Privilege enforcement",
                    "Scenario: Unauthorized request is denied",
                    *(f"{step.keyword} {step.text}" for step in steps),
                )
            ),
            bindings=(
                GherkinStepBinding(
                    claim_field="actor", source_index=None, step_numbers=[1]
                ),
                GherkinStepBinding(
                    claim_field="precondition", source_index=0, step_numbers=[2]
                ),
                GherkinStepBinding(
                    claim_field="action", source_index=None, step_numbers=[3]
                ),
                GherkinStepBinding(
                    claim_field="expected_secure_behavior",
                    source_index=None,
                    step_numbers=[4],
                ),
                GherkinStepBinding(
                    claim_field="possible_failure", source_index=None, step_numbers=[5]
                ),
                GherkinStepBinding(
                    claim_field="observable", source_index=0, step_numbers=[6]
                ),
                GherkinStepBinding(
                    claim_field="observable", source_index=1, step_numbers=[7]
                ),
            ),
            step_evidence_bindings=_step_evidence_bindings(),
            testability_notes=[],
            setup_gaps=[],
            generated_at=datetime(2026, 8, 11, tzinfo=UTC),
        )
    )


def _step_evidence_bindings() -> tuple[GherkinStepEvidenceBinding, ...]:
    """Bind every fixture step to saved integration-change evidence."""
    return tuple(
        GherkinStepEvidenceBinding(
            step_number=step_number,
            anchor_ids=("anchor-a",),
        )
        for step_number in range(1, 8)
    )


def _context_bundle(snapshot_key: str = "0" * 64) -> ContextBundle:
    anchors = tuple(
        ContextAnchor(
            anchor_id=anchor_id,
            revision_role="candidate",
            commit_sha="4" * 40,
            blob_sha="5" * 40,
            path="api/Patient.java",
            java_symbol="Patient.delete",
            start_line=index,
            end_line=index,
            text="requirePrivilege" if anchor_id == "anchor-a" else "related evidence",
            text_sha256=hashlib.sha256(
                (
                    "requirePrivilege"
                    if anchor_id == "anchor-a"
                    else "related evidence"
                ).encode()
            ).hexdigest(),
            selection_reason="integration change",
            score_components=[],
            change_relation="integration_change"
            if anchor_id == "anchor-a"
            else "repository_context",
            truncated=False,
        )
        for index, anchor_id in enumerate(("anchor-a", "anchor-b", "anchor-c"), start=1)
    )
    payload = {
        "snapshot_key": snapshot_key,
        "anchors": [anchor.model_dump(mode="json") for anchor in anchors],
        "selected_file_count": 1,
        "selected_anchor_count": 3,
        "selected_bytes": sum(len(anchor.text.encode("utf-8")) for anchor in anchors),
        "max_files": 40,
        "max_anchors": 80,
        "max_bytes": 160_000,
        "max_anchor_lines": 120,
        "max_blob_bytes": 1_000_000,
        "max_search_identifiers": 100,
        "max_hits_per_identifier": 20,
        "primary_change_represented": True,
    }
    return ContextBundle.from_content(
        **payload,
    )


def _assessment(
    risk: RiskHypothesis,
    snapshot_key: str = "0" * 64,
) -> RiskAssessment:
    context = _context_bundle(snapshot_key)
    report = GroundingReport(
        producer="local_grounding_validator",
        snapshot_key=snapshot_key,
        context_sha256=context.context_sha256,
        hypothesis_id=risk.hypothesis_id,
        hypothesis_sha256=canonical_sha256(risk.model_dump(mode="json")),
        cited_anchor_ids=risk.citation_anchor_ids,
        identifier_evidence=[
            IdentifierEvidence(identifier="requirePrivilege", anchor_ids=["anchor-a"])
        ],
    )
    return RiskAssessment.from_content(
        snapshot_key=snapshot_key,
        context_sha256=context.context_sha256,
        outcome="risks_proposed",
        hypotheses=[risk],
        generated_at=datetime(2026, 8, 11, tzinfo=UTC),
        validated_at=datetime(2026, 8, 11, tzinfo=UTC),
        context_bundle=context,
        grounding_reports=[report],
    )


def _reviewed_grounding(
    reviewed: RiskHypothesisDraft,
    snapshot_key: str = "0" * 64,
) -> GroundingReport:
    return GroundingReport(
        producer="local_grounding_validator",
        snapshot_key=snapshot_key,
        context_sha256=_context_bundle(snapshot_key).context_sha256,
        hypothesis_id="reviewed-risk",
        hypothesis_sha256=canonical_sha256(reviewed.model_dump(mode="json")),
        cited_anchor_ids=reviewed.citation_anchor_ids,
        identifier_evidence=[
            IdentifierEvidence(identifier="requirePrivilege", anchor_ids=["anchor-a"])
        ],
    )


def test_candidate_requires_complete_bindings_and_phase_order() -> None:
    """A scenario may reach approval only when every approved claim has a legal phase."""
    candidate = _candidate()
    with pytest.raises(ValidationError, match="cover every"):
        GherkinCandidateDraft.model_validate(
            candidate.model_dump(mode="json", exclude={"bindings", "candidate_id"})
            | {"bindings": candidate.bindings[:-1]}
        )


def test_candidate_steps_are_immutable_after_validation() -> None:
    """A validated scenario cannot be altered after its candidate hash is computed."""
    candidate = _candidate()

    assert isinstance(candidate.steps, tuple)
    with pytest.raises(AttributeError):
        candidate.steps.append(candidate.steps[0])  # type: ignore[attr-defined]

    bad_steps = list(candidate.steps)
    bad_steps[2] = GherkinStep(
        number=3, keyword="Then", text="the changed request is sent"
    )
    with pytest.raises(ValidationError, match="phase"):
        GherkinCandidateDraft.model_validate(
            candidate.model_dump(mode="json", exclude={"candidate_id"})
            | {
                "steps": bad_steps,
                "gherkin_text": candidate.gherkin_text.replace(
                    "When the changed request is sent",
                    "Then the changed request is sent",
                ),
            }
        )


def test_terminal_record_rejects_mixed_snapshot_artifacts() -> None:
    """An approved record cannot join a candidate from another frozen PR snapshot."""
    risk = RiskHypothesis.from_draft(_risk_draft())
    assessment = _assessment(risk)
    reviewed = _risk_draft()
    review = HumanReviewedRisk(
        snapshot_key="0" * 64,
        assessment_sha256=assessment.assessment_sha256,
        selected_hypothesis_id=risk.hypothesis_id,
        selected_hypothesis_sha256=canonical_sha256(risk.model_dump(mode="json")),
        reviewed_risk=reviewed,
        reviewed_content_sha256=canonical_sha256(reviewed.model_dump(mode="json")),
        reviewed_grounding=_reviewed_grounding(reviewed),
        approved_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    candidate = _candidate(snapshot_key="e" * 64)
    approval = GherkinApproval(
        snapshot_key="e" * 64,
        candidate_id=candidate.candidate_id,
        candidate_sha256=canonical_sha256(candidate.model_dump(mode="json")),
        reviewed_risk_sha256=candidate.reviewed_risk_sha256,
        approved_at=datetime(2026, 8, 11, tzinfo=UTC),
    )

    with pytest.raises(ValidationError, match="snapshot key"):
        MilestoneTwoRunRecord(
            run_id="run-1",
            snapshot=PullRequestSnapshot.model_validate(snapshot_payload()),
            status="approved_gherkin",
            reason_code="gherkin_approved",
            explanation="A human approved the scenario.",
            started_at=datetime(2026, 8, 11, tzinfo=UTC),
            finished_at=datetime(2026, 8, 11, 0, 1, tzinfo=UTC),
            risk_assessment=assessment,
            human_reviewed_risk=review,
            gherkin_candidate=candidate,
            gherkin_approval=approval,
        )


def test_approved_terminal_requires_current_matching_final_freshness() -> None:
    """Gherkin approval must be preceded by a current check of this exact snapshot."""
    risk = RiskHypothesis.from_draft(_risk_draft())
    assessment = _assessment(risk)
    reviewed = _risk_draft()
    review = HumanReviewedRisk(
        snapshot_key="0" * 64,
        assessment_sha256=assessment.assessment_sha256,
        selected_hypothesis_id=risk.hypothesis_id,
        selected_hypothesis_sha256=canonical_sha256(risk.model_dump(mode="json")),
        reviewed_risk=reviewed,
        reviewed_content_sha256=canonical_sha256(reviewed.model_dump(mode="json")),
        reviewed_grounding=_reviewed_grounding(reviewed),
        approved_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    candidate = _candidate()
    approval = GherkinApproval(
        snapshot_key="0" * 64,
        candidate_id=candidate.candidate_id,
        candidate_sha256=canonical_sha256(candidate.model_dump(mode="json")),
        reviewed_risk_sha256=candidate.reviewed_risk_sha256,
        approved_at=datetime(2026, 8, 11, 0, 1, tzinfo=UTC),
    )
    with pytest.raises(ValidationError, match="snapshot key"):
        MilestoneTwoRunRecord(
            run_id="run-currentness",
            snapshot=PullRequestSnapshot.model_validate(snapshot_payload()),
            status="approved_gherkin",
            reason_code="gherkin_approved",
            explanation="A human approved the scenario.",
            started_at=datetime(2026, 8, 11, tzinfo=UTC),
            finished_at=datetime(2026, 8, 11, 0, 2, tzinfo=UTC),
            risk_assessment=assessment,
            human_reviewed_risk=review,
            gherkin_candidate=candidate,
            gherkin_approval=approval,
        )


def test_insufficient_context_requires_an_allowlisted_reason_code() -> None:
    """An abstention reason must be from the documented operational vocabulary."""
    with pytest.raises(ValidationError, match="reason_code"):
        RiskAssessmentDraft(
            snapshot_key="0" * 64,
            context_sha256="a" * 64,
            outcome="insufficient_context_to_assess",
            reason_code="made_up_reason",
            missing_evidence=["integration diff"],
            needed_evidence=["current merge commit"],
            generated_at=datetime(2026, 8, 11, tzinfo=UTC),
        )


def test_failed_terminal_record_requires_an_allowlisted_reason_code() -> None:
    """Terminal failures may not invent a reason outside the approved vocabulary."""
    with pytest.raises(ValidationError, match="supported reason"):
        MilestoneTwoRunRecord(
            run_id="run-failure",
            snapshot=PullRequestSnapshot.model_validate(snapshot_payload()),
            status="failed",
            reason_code="made_up_reason",
            explanation="An unsupported failure occurred.",
            started_at=datetime(2026, 8, 11, tzinfo=UTC),
            finished_at=datetime(2026, 8, 11, 0, 1, tzinfo=UTC),
        )


def test_exhausted_frozen_evidence_terminal_preserves_its_testability_history() -> None:
    """An untestable risk is not safe when every allowed code search is exhausted."""
    approved = _approved_terminal_record()
    assert approved.risk_assessment is not None
    assert approved.human_reviewed_risk is not None
    snapshot = approved.snapshot
    assessment = approved.risk_assessment
    review = approved.human_reviewed_risk
    need = FrozenEvidenceNeed(
        need_id="need-observable",
        category="observable",
        search_terms=("response",),
        explanation="Find a frozen observable outcome for the reviewed action.",
        supporting_anchor_ids=("anchor-a",),
    )
    testability = Assessment.from_content(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=assessment.context_sha256,
        reviewed_risk_sha256=review.reviewed_content_sha256,
        decision="needs_more_frozen_evidence",
        bindings=(),
        evidence_needs=(need,),
        explanation="The frozen context does not establish an executable observable.",
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
        validated_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    refinement = ContextRefinement.from_content(
        snapshot_key=snapshot.snapshot_key,
        parent_context_sha256=assessment.context_sha256,
        refined_context_sha256=assessment.context_sha256,
        evidence_need_ids=(need.need_id,),
        added_anchor_ids=(),
        exhausted=True,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    record = MilestoneTwoRunRecord(
        run_id="run-exhausted-frozen-evidence",
        snapshot=snapshot,
        status="insufficient_frozen_evidence_for_scenario",
        reason_code="insufficient_frozen_evidence_for_scenario",
        explanation="Insufficient frozen code evidence to design an executable scenario.",
        started_at=datetime(2026, 8, 11, tzinfo=UTC),
        finished_at=datetime(2026, 8, 12, tzinfo=UTC),
        freshness=SnapshotFreshness(
            snapshot_key=snapshot.snapshot_key,
            status="current",
            reason_code="snapshot_current",
            checked_at=datetime(2026, 8, 12, tzinfo=UTC),
            observed_base_sha=snapshot.base_sha,
            observed_head_sha=snapshot.head_sha,
            observed_candidate_sha=snapshot.candidate_sha,
        ),
        risk_assessment=assessment,
        human_reviewed_risk=review,
        testability_assessment=testability,
        context_refinements=(refinement,),
        gherkin_candidate=None,
        gherkin_approval=None,
    )

    assert record.status == "insufficient_frozen_evidence_for_scenario"
    assert record.testability_assessment == testability
    assert record.context_refinements == (refinement,)


def test_persisted_assessment_round_trip_revalidates_derived_hypotheses() -> None:
    """Saved assessments must reload only when nested local IDs remain genuine."""
    assessment = _assessment(RiskHypothesis.from_draft(_risk_draft()))
    persisted = assessment.model_dump(mode="json")

    restored = RiskAssessment.from_persisted(persisted)

    assert restored == assessment

    tampered = assessment.model_dump(mode="json")
    tampered["hypotheses"][0]["hypothesis_id"] = "risk-" + ("f" * 64)

    with pytest.raises(ValueError, match="persisted hypothesis ID"):
        RiskAssessment.from_persisted(tampered)


def test_persisted_failed_terminal_record_round_trips() -> None:
    """A saved terminal record must be safely readable in a later session."""
    record = MilestoneTwoRunRecord(
        run_id="run-persisted-failure",
        snapshot=PullRequestSnapshot.model_validate(snapshot_payload()),
        status="failed",
        reason_code="model_output_invalid",
        explanation="The recorded model output did not pass local validation.",
        started_at=datetime(2026, 8, 11, tzinfo=UTC),
        finished_at=datetime(2026, 8, 11, 0, 1, tzinfo=UTC),
    )

    restored = MilestoneTwoRunRecord.from_persisted(record.model_dump(mode="json"))

    assert restored == record


def test_context_factory_normalizes_anchors_before_hashing() -> None:
    """The context factory must not hash a temporary mutable anchor collection."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        context = _context_bundle()

    assert isinstance(context.anchors, tuple)


def test_assessment_factory_normalizes_artifact_collections_before_hashing() -> None:
    """The assessment factory must hash immutable collections without warnings."""
    risk = RiskHypothesis.from_draft(_risk_draft())

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assessment = _assessment(risk)

    assert isinstance(assessment.hypotheses, tuple)
    assert isinstance(assessment.grounding_reports, tuple)


def _approved_terminal_record() -> MilestoneTwoRunRecord:
    snapshot = PullRequestSnapshot.model_validate(snapshot_payload())
    risk = RiskHypothesis.from_draft(_risk_draft())
    assessment = _assessment(risk, snapshot.snapshot_key)
    reviewed = _risk_draft()

    review = HumanReviewedRisk(
        snapshot_key=snapshot.snapshot_key,
        assessment_sha256=assessment.assessment_sha256,
        selected_hypothesis_id=risk.hypothesis_id,
        selected_hypothesis_sha256=canonical_sha256(risk.model_dump(mode="json")),
        reviewed_risk=reviewed,
        reviewed_content_sha256=canonical_sha256(reviewed.model_dump(mode="json")),
        reviewed_grounding=_reviewed_grounding(
            reviewed,
            snapshot.snapshot_key,
        ),
        approved_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    candidate = _candidate(snapshot.snapshot_key)
    approval = GherkinApproval(
        snapshot_key=snapshot.snapshot_key,
        candidate_id=candidate.candidate_id,
        candidate_sha256=canonical_sha256(candidate.model_dump(mode="json")),
        reviewed_risk_sha256=candidate.reviewed_risk_sha256,
        approved_at=datetime(2026, 8, 11, 0, 1, tzinfo=UTC),
    )
    freshness = SnapshotFreshness(
        snapshot_key=snapshot.snapshot_key,
        status="current",
        reason_code="snapshot_current",
        checked_at=datetime(2026, 8, 11, tzinfo=UTC),
        observed_base_sha=snapshot.base_sha,
        observed_head_sha=snapshot.head_sha,
        observed_candidate_sha=snapshot.candidate_sha,
    )

    return MilestoneTwoRunRecord(
        run_id="run-approved-persisted",
        snapshot=snapshot,
        status="approved_gherkin",
        reason_code="gherkin_approved",
        explanation="A reviewer approved the generated scenario.",
        started_at=datetime(2026, 8, 11, tzinfo=UTC),
        finished_at=datetime(2026, 8, 11, 0, 2, tzinfo=UTC),
        freshness=freshness,
        risk_assessment=assessment,
        human_reviewed_risk=review,
        gherkin_candidate=candidate,
        gherkin_approval=approval,
    )


def test_persisted_approved_terminal_record_revalidates_nested_ids() -> None:
    """Saved approval evidence must reload through trusted nested ID checks."""
    record = _approved_terminal_record()
    persisted = record.model_dump(mode="json")

    restored = MilestoneTwoRunRecord.from_persisted(persisted)

    assert restored == record

    tampered = record.model_dump(mode="json")
    tampered["risk_assessment"]["hypotheses"][0]["hypothesis_id"] = "risk-" + ("f" * 64)

    with pytest.raises(ValueError, match="persisted hypothesis ID"):
        MilestoneTwoRunRecord.from_persisted(tampered)


def test_readable_hypothesis_explanation_must_have_frozen_evidence() -> None:
    """The reviewer-facing risk paragraph must cite the saved code catalog."""
    hypothesis = _risk_draft()

    assert any(
        binding.claim_field == "explanation" for binding in hypothesis.evidence_bindings
    )

    payload = hypothesis.model_dump(mode="json")
    payload["evidence_bindings"] = [
        binding
        for binding in payload["evidence_bindings"]
        if binding["claim_field"] != "explanation"
    ]

    with pytest.raises(ValidationError, match="cover every required claim"):
        RiskHypothesisDraft.model_validate(payload)
