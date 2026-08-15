"""Tests for the public Milestone 2 workflow package boundary."""

from triageguard.workflow import (
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
    PreparedPullRequest,
)
from triageguard.workflow.milestone_two import (
    MilestoneTwoTransitionError as InternalTransitionError,
)
from triageguard.workflow.milestone_two import (
    MilestoneTwoWorkflow as InternalWorkflow,
)
from triageguard.workflow.milestone_two import (
    PreparedPullRequest as InternalPreparedPullRequest,
)


def test_milestone_two_workflow_is_available_from_the_public_package() -> None:
    """Application code should not need to import the internal module path."""
    assert MilestoneTwoWorkflow is InternalWorkflow
    assert PreparedPullRequest is InternalPreparedPullRequest
    assert MilestoneTwoTransitionError is InternalTransitionError
