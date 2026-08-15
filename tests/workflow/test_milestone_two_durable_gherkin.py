"""Tests for durable storage of a generated Milestone 2 Gherkin candidate."""

import hashlib
import json
from datetime import UTC, datetime

import pytest

from triageguard.config import Settings
from triageguard.domain import (
    ClaimEvidenceBinding,
    ContextAnchor,
    ContextBundle,
    DiffArtifact,
    EnvironmentKind,
    GherkinCandidate,
    GherkinCandidateDraft,
    GherkinStep,
    GherkinStepBinding,
    HumanReviewedRisk,
    PullRequestSnapshot,
    RiskAssessmentDraft,
    RiskHypothesisDraft,
    SnapshotFreshness,
)
from triageguard.hypotheses import (
    create_human_review,
    validate_risk_assessment,
)
from triageguard.llm import ModelAttempt, ModelResponse, ReplayGateway
from triageguard.provenance import canonical_sha256
from triageguard.research import ArtifactRecorder
from triageguard.workflow import milestone_two
from triageguard.workflow.milestone_two import (
    MilestoneTwoDependencies,
    MilestoneTwoWorkflow,
    PreparedPullRequest,
    resume_milestone_two_workflow,
)


class _SnapshotAcquirer:
    """Return a current freshness result for one frozen snapshot."""

    def recheck(self, snapshot: PullRequestSnapshot) -> SnapshotFreshness:
        return SnapshotFreshness(
            snapshot_key=snapshot.snapshot_key,
            status="current",
            reason_code="snapshot_current",
            checked_at=datetime(2026, 8, 15, tzinfo=UTC),
            observed_base_sha=snapshot.base_sha,
            observed_head_sha=snapshot.head_sha,
            observed_candidate_sha=snapshot.candidate_sha,
        )


class _UnusedDiffBuilder:
    """This focused test starts after the three diffs already exist."""

    def build_all(self, snapshot: PullRequestSnapshot) -> tuple[object, object, object]:
        raise AssertionError("the test directly supplies the frozen evidence")


class _UnusedContextBuilder:
    """This focused test starts after the context catalog already exists."""

    def build(self, **kwargs: object) -> ContextBundle:
        raise AssertionError("the test directly supplies the frozen evidence")


def _snapshot() -> PullRequestSnapshot:
    """Build one valid synthetic frozen OpenMRS Core pull-request identity."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=900000003,
        pull_url="https://github.com/openmrs/openmrs-core/pull/900000003",
        state="open",
        default_branch="main",
        base_branch="main",
        merge_base_sha="1" * 40,
        base_sha="2" * 40,
        head_sha="3" * 40,
        candidate_sha="4" * 40,
        merge_base_tree_sha="5" * 40,
        base_tree_sha="6" * 40,
        head_tree_sha="7" * 40,
        candidate_tree_sha="8" * 40,
        acquired_at=datetime(2026, 8, 15, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="9" * 64,
    )


def _context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Build one valid frozen integration-change context anchor."""
    text = "void purgePatient() { dao.deletePatient(); }"
    anchor = ContextAnchor(
        anchor_id="anchor-integration",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="a" * 40,
        path="api/src/main/java/org/openmrs/PatientService.java",
        java_symbol="purgePatient",
        start_line=10,
        end_line=10,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        selection_reason="primary integration change",
        score_components=(),
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


def _diffs(
    snapshot: PullRequestSnapshot,
) -> tuple[DiffArtifact, DiffArtifact, DiffArtifact]:
    """Build the required three reproducible comparisons."""
    return (
        DiffArtifact(
            kind="author_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.head_sha,
            git_arguments=("diff", "--no-ext-diff"),
            git_version="2.47.1",
            files=(),
            patch_sha256="a" * 64,
            artifact_sha256="a" * 64,
        ),
        DiffArtifact(
            kind="integration_diff",
            old_revision=snapshot.base_sha,
            new_revision=snapshot.candidate_sha,
            git_arguments=("diff", "--no-ext-diff"),
            git_version="2.47.1",
            files=(),
            patch_sha256="b" * 64,
            artifact_sha256="b" * 64,
        ),
        DiffArtifact(
            kind="base_drift_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.base_sha,
            git_arguments=("diff", "--no-ext-diff"),
            git_version="2.47.1",
            files=(),
            patch_sha256="c" * 64,
            artifact_sha256="c" * 64,
        ),
    )


def _human_review(snapshot: PullRequestSnapshot) -> HumanReviewedRisk:
    """Build one valid already-approved human risk selection."""
    risk = RiskHypothesisDraft(
        claim_status="unconfirmed_risk_hypothesis",
        title="Patient deletion may bypass authorization",
        explanation="The changed deletion path needs an executable authorization check.",
        actor="An authenticated OpenMRS user",
        preconditions=("The user can reach the patient deletion service path.",),
        action="The user requests deletion of a patient record.",
        protected_asset="Patient records",
        security_property="Authorization",
        expected_secure_behavior=(
            "The API rejects deletion and the patient remains stored."
        ),
        possible_failure=(
            "The API deletes a patient record without the expected authorization."
        ),
        observables=("The deletion request is rejected.",),
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
        ),
        limitations=("Authorization configuration is outside this bounded context.",),
        missing_evidence=(),
        priority_rationale="Patient deletion needs a human-approved executable test.",
    )
    return HumanReviewedRisk(
        snapshot_key=snapshot.snapshot_key,
        assessment_sha256="b" * 64,
        selected_hypothesis_id="risk-original",
        selected_hypothesis_sha256="c" * 64,
        reviewed_risk=risk,
        reviewed_content_sha256=canonical_sha256(risk.model_dump(mode="json")),
        approved_at=datetime(2026, 8, 15, tzinfo=UTC),
    )


def _candidate(review: HumanReviewedRisk) -> GherkinCandidate:
    """Build one locally valid candidate bound to the human-reviewed risk."""
    risk = review.reviewed_risk
    steps = (
        GherkinStep(
            number=1,
            keyword="Given",
            text=(
                "an authenticated OpenMRS user can reach the patient deletion "
                "service path"
            ),
        ),
        GherkinStep(
            number=2,
            keyword="When",
            text=("the user requests deletion using purgePatient and deletePatient"),
        ),
        GherkinStep(
            number=3,
            keyword="Then",
            text=risk.expected_secure_behavior,
        ),
        GherkinStep(
            number=4,
            keyword="And",
            text=risk.possible_failure,
        ),
        GherkinStep(
            number=5,
            keyword="And",
            text=risk.observables[0],
        ),
    )
    feature_title = "Patient deletion authorization"
    scenario_title = "Unauthorized patient deletion is rejected"
    text = "\n".join(
        (
            f"Feature: {feature_title}",
            "",
            f"Scenario: {scenario_title}",
            "",
            *(f"{step.keyword} {step.text}" for step in steps),
        )
    )
    draft = GherkinCandidateDraft(
        snapshot_key=review.snapshot_key,
        reviewed_risk_sha256=review.reviewed_content_sha256,
        approved_risk=risk,
        feature_title=feature_title,
        scenario_title=scenario_title,
        steps=steps,
        gherkin_text=text,
        bindings=(
            GherkinStepBinding(
                claim_field="actor",
                source_index=None,
                step_numbers=(1,),
            ),
            GherkinStepBinding(
                claim_field="precondition",
                source_index=0,
                step_numbers=(1,),
            ),
            GherkinStepBinding(
                claim_field="action",
                source_index=None,
                step_numbers=(2,),
            ),
            GherkinStepBinding(
                claim_field="expected_secure_behavior",
                source_index=None,
                step_numbers=(3,),
            ),
            GherkinStepBinding(
                claim_field="possible_failure",
                source_index=None,
                step_numbers=(4,),
            ),
            GherkinStepBinding(
                claim_field="observable",
                source_index=0,
                step_numbers=(5,),
            ),
        ),
        testability_notes=("Run the scenario in an OpenMRS test environment.",),
        setup_gaps=(),
        generated_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    return GherkinCandidate.from_draft(draft)


def _response() -> ModelResponse:
    """Build recorder-ready provenance for one completed replayed model call."""
    moment = datetime(2026, 8, 15, tzinfo=UTC)
    return ModelResponse(
        data={},
        provider="replay",
        model="replay/openai-gpt-oss-120b",
        latency_ms=0,
        prompt_sha256="d" * 64,
        response_sha256="e" * 64,
        input_tokens=0,
        output_tokens=0,
        attempts=(
            ModelAttempt(
                number=1,
                started_at=moment,
                finished_at=moment,
                latency_ms=0,
                outcome="succeeded",
            ),
        ),
    )


def test_generated_gherkin_and_its_model_response_are_saved_before_review(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart after generation must have the exact candidate response to reload."""
    snapshot = _snapshot()
    context = _context(snapshot)
    review = _human_review(snapshot)
    candidate = _candidate(review)
    response = _response()
    recorder = ArtifactRecorder(tmp_path)
    workflow = MilestoneTwoWorkflow(
        run_id="m2-durable-gherkin-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        context=context,
    )
    workflow._human_reviewed_risk = review
    workflow._state = milestone_two._State.RISK_APPROVED

    monkeypatch.setattr(
        milestone_two,
        "request_gherkin_candidate",
        lambda **_kwargs: (candidate, response),
    )

    assert workflow.generate_gherkin() == candidate

    stored = json.loads(
        recorder.read_artifact(
            workflow.run_handle,
            "artifacts/workflow/gherkin_generation.json",
        )
    )
    assert stored["candidate"] == candidate.model_dump(mode="json")
    assert stored["response"] == response.model_dump(mode="json")
    assert stored["freshness"]["status"] == "current"


def _assessment(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
):
    """Locally validate one reviewable risk against its frozen context."""
    draft = RiskAssessmentDraft(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        outcome="risks_proposed",
        hypotheses=(_human_review(snapshot).reviewed_risk,),
        generated_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    assessment, report = validate_risk_assessment(
        draft=draft,
        snapshot=snapshot,
        context=context,
    )
    assert assessment is not None
    assert report.approved is True
    return assessment


def test_human_review_is_saved_before_gherkin_generation(tmp_path) -> None:
    """A restart after review must retain the reviewer-selected exact risk."""
    snapshot = _snapshot()
    context = _context(snapshot)
    assessment = _assessment(snapshot, context)
    recorder = ArtifactRecorder(tmp_path)
    workflow = MilestoneTwoWorkflow(
        run_id="m2-durable-review-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        context=context,
    )
    workflow._risk_assessment = assessment
    workflow._state = milestone_two._State.RISKS_READY

    selected = assessment.hypotheses[0]
    review = workflow.approve_risk(
        selected.hypothesis_id,
        {},
        selected.citation_anchor_ids,
    )

    stored = json.loads(
        recorder.read_artifact(
            workflow.run_handle,
            "artifacts/workflow/human_review.json",
        )
    )
    assert stored["review"] == review.model_dump(mode="json")
    assert stored["freshness"]["status"] == "current"


def test_resume_restores_review_and_gherkin_without_another_model_call(
    tmp_path,
) -> None:
    """A restart at Gherkin review must reuse both completed model outputs."""
    snapshot = _snapshot()
    context = _context(snapshot)
    prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        context=context,
    )
    risk_draft = RiskAssessmentDraft(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        outcome="risks_proposed",
        hypotheses=(_human_review(snapshot).reviewed_risk,),
        generated_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    assessment, report = validate_risk_assessment(
        draft=risk_draft,
        snapshot=snapshot,
        context=context,
    )
    assert assessment is not None
    assert report.approved is True

    selected = assessment.hypotheses[0]
    review = create_human_review(
        assessment=assessment,
        hypothesis_id=selected.hypothesis_id,
        edits={},
        selected_anchor_ids=selected.citation_anchor_ids,
        reviewed_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    candidate = _candidate(review)
    freshness = _SnapshotAcquirer().recheck(snapshot)
    risk_response = _response().model_copy(
        update={"data": risk_draft.model_dump(mode="json")}
    )
    gherkin_response = _response().model_copy(
        update={"data": candidate.model_dump(mode="json")}
    )

    recorder = ArtifactRecorder(tmp_path)
    dependencies = MilestoneTwoDependencies(
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )
    first = MilestoneTwoWorkflow(
        run_id="m2-gherkin-recovery-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )
    first._persist_prepared(prepared)
    first._persist_risk_generation(
        prepared=prepared,
        draft=risk_draft,
        response=risk_response,
    )
    first._persist_human_review(
        prepared=prepared,
        assessment=assessment,
        review=review,
        freshness=freshness,
    )
    first._persist_gherkin_generation(
        prepared=prepared,
        human_review=review,
        candidate=candidate,
        response=gherkin_response,
        freshness=freshness,
    )

    resumed = resume_milestone_two_workflow(
        run_handle=first.run_handle,
        dependencies=dependencies,
    )

    assert resumed.risk_assessment == assessment
    assert resumed.human_reviewed_risk == review
    assert resumed.gherkin_candidate == candidate

    terminal_record = resumed.approve_gherkin(candidate.gherkin_text)

    assert terminal_record.gherkin_candidate == candidate
    assert resumed.terminal_record == terminal_record
