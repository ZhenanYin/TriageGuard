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
    candidate = workflow.generate_gherkin()
    record = workflow.approve_gherkin(candidate.gherkin_text)

    assert prepared.snapshot.repository == "openmrs/openmrs-core"
    assert prepared.snapshot.pull_number == 900000001
    assert prepared.snapshot.merge_base_sha != prepared.snapshot.base_sha
    assert prepared.snapshot.head_sha != prepared.snapshot.candidate_sha
    assert assessment.outcome == "risks_proposed"
    assert review.selected_hypothesis_id == hypothesis.hypothesis_id
    assert candidate.snapshot_key == prepared.snapshot.snapshot_key
    assert record.status.value == "approved_gherkin"
    assert record.gherkin_candidate == candidate
