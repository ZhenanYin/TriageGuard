"""Tests for the one-way Milestone 2 pull-request workflow."""

import pytest

from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.workflow.milestone_two import (
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
)


def _workflow(tmp_path) -> MilestoneTwoWorkflow:
    """Create a workflow whose dependencies are unused before preparation."""
    return MilestoneTwoWorkflow(
        run_id="m2-run-1",
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=object(),
        diff_builder=object(),
        context_builder=object(),
        store=object(),
        gateway=ReplayGateway({}),
    )


def test_workflow_rejects_risk_proposal_before_pr_is_prepared(tmp_path) -> None:
    """Risk generation cannot run until one exact PR snapshot exists."""
    workflow = _workflow(tmp_path)

    with pytest.raises(MilestoneTwoTransitionError, match="prepare"):
        workflow.propose_risks()


def test_workflow_rejects_gherkin_generation_before_risk_approval(tmp_path) -> None:
    """Gherkin cannot be generated without a human-approved risk first."""
    workflow = _workflow(tmp_path)

    with pytest.raises(MilestoneTwoTransitionError, match="prepare"):
        workflow.generate_gherkin()
