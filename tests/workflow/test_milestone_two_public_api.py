"""Tests for the public Milestone 2 workflow package boundary."""

from triageguard.workflow import (
    MilestoneTwoDependencies,
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
    PreparedPullRequest,
    resume_milestone_two_workflow,
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


from triageguard.workflow.milestone_two import (
    MilestoneTwoDependencies as InternalDependencies,
)
from triageguard.workflow.milestone_two import (
    resume_milestone_two_workflow as internal_resume_milestone_two_workflow,
)


def test_milestone_two_recovery_api_is_available_from_the_public_package() -> None:
    """The UI can resume a durable run without importing an internal path."""
    assert MilestoneTwoDependencies is InternalDependencies
    assert resume_milestone_two_workflow is internal_resume_milestone_two_workflow
