"""End-to-end offline replay of one synthetic OpenMRS-shaped pull request."""

from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.workflow.milestone_two_replay import (
    build_milestone_two_replay_workflow,
)

SUPPORTED_PR_URL = "https://github.com/openmrs/openmrs-core/pull/900000001"


def test_offline_replay_reaches_approved_gherkin(tmp_path) -> None:
    """The synthetic fixture exercises the complete human-gated workflow offline."""
    settings = Settings(
        llm_mode="replay",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        artifacts_dir=tmp_path / "artifacts",
        analysis_cache_dir=tmp_path / "analysis-cache",
    )
    workflow = build_milestone_two_replay_workflow(settings)

    prepared = workflow.prepare_pr(SUPPORTED_PR_URL)
    assessment = workflow.propose_risks()
    hypothesis = assessment.hypotheses[0]
    review = workflow.approve_risk(
        hypothesis.hypothesis_id,
        {},
        hypothesis.citation_anchor_ids,
    )
    testability = workflow.assess_testability()
    candidate = workflow.generate_gherkin()
    record = workflow.approve_gherkin(candidate.gherkin_text)

    assert prepared.snapshot.repository == "openmrs/openmrs-core"
    assert prepared.snapshot.pull_number == 900000001
    assert prepared.snapshot.merge_base_sha != prepared.snapshot.base_sha
    assert prepared.snapshot.head_sha != prepared.snapshot.candidate_sha
    assert assessment.outcome == "risks_proposed"
    assert review.selected_hypothesis_id == hypothesis.hypothesis_id
    assert testability.decision == "testable_from_frozen_evidence"
    assert candidate.snapshot_key == prepared.snapshot.snapshot_key
    assert record.status.value == "approved_gherkin"
    assert record.gherkin_candidate == candidate
    for stage, envelope in (
        ("risk_hypothesis", workflow.risk_evidence_envelope),
        ("testability_assessment", workflow.testability_evidence_envelope),
        ("gherkin_generation", workflow.gherkin_evidence_envelope),
    ):
        assert envelope is not None
        assert envelope.stage == stage
        assert envelope.visible_anchors
        assert len(envelope.visible_anchors) + len(envelope.omitted_anchors) == len(
            envelope.catalog_anchor_ids
        )


def test_offline_no_drift_replay_uses_an_explicit_unchanged_comparison(
    tmp_path,
) -> None:
    """M == B must still traverse the normal replay and evidence-boundary path."""
    settings = Settings(
        llm_mode="replay",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        artifacts_dir=tmp_path / "artifacts",
        analysis_cache_dir=tmp_path / "analysis-cache",
    )
    workflow = build_milestone_two_replay_workflow(
        settings,
        snapshot_variant="no_drift",
    )

    prepared = workflow.prepare_pr(SUPPORTED_PR_URL)
    assessment = workflow.propose_risks()

    base_drift = next(
        artifact for artifact in prepared.diffs if artifact.kind == "base_drift_diff"
    )
    assert prepared.snapshot.merge_base_sha == prepared.snapshot.base_sha
    assert base_drift.old_revision == base_drift.new_revision
    assert base_drift.comparison_status == "unchanged"
    assert base_drift.files == ()
    assert assessment.evidence_envelope_sha256 == (
        workflow.risk_evidence_envelope.envelope_sha256
    )
