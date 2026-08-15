"""Offline terminal outcomes for the synthetic Milestone 2 fixture."""

import pytest

from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.workflow.milestone_two_replay import (
    build_milestone_two_replay_workflow,
)

SUPPORTED_PR_URL = "https://github.com/openmrs/openmrs-core/pull/900000001"


@pytest.mark.parametrize(
    ("outcome", "terminal_status"),
    (
        (
            "no_meaningful_security_risk_found",
            "no_meaningful_security_risk_found",
        ),
        (
            "insufficient_context_to_assess",
            "insufficient_context_to_assess",
        ),
    ),
)
def test_nonrisk_outcomes_finish_without_gherkin(
    tmp_path,
    outcome: str,
    terminal_status: str,
) -> None:
    """Both supported non-risk outcomes stop before human risk review."""
    settings = Settings(
        llm_mode="replay",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        artifacts_dir=tmp_path / "artifacts",
        analysis_cache_dir=tmp_path / "analysis-cache",
    )
    workflow = build_milestone_two_replay_workflow(
        settings,
        outcome=outcome,
    )

    workflow.prepare_pr(SUPPORTED_PR_URL)
    assessment = workflow.propose_risks()
    record = workflow.finish_without_risk()

    assert assessment.outcome == outcome
    assert record.status.value == terminal_status
    assert record.human_reviewed_risk is None
    assert record.gherkin_candidate is None
    assert record.gherkin_approval is None
