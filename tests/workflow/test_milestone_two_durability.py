"""Tests for safe Milestone 2 workflow recovery."""

import pytest

from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder, RunHandle, RunOwnership
from triageguard.workflow.milestone_two import (
    MilestoneTwoDependencies,
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
    resume_milestone_two_workflow,
)


class _SnapshotAcquirer:
    """Placeholder dependency; recovery does not acquire or recheck a PR."""

    def acquire(self, pr_url: str) -> object:
        raise AssertionError("resume must not acquire a new snapshot")

    def recheck(self, snapshot: object) -> object:
        raise AssertionError("resume must not recheck a snapshot")


class _DiffBuilder:
    """Placeholder dependency; recovery must not rebuild differences."""

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        raise AssertionError("resume must not rebuild diffs")


class _ContextBuilder:
    """Placeholder dependency; recovery must not rebuild context."""

    def build(self, **kwargs: object) -> object:
        raise AssertionError("resume must not rebuild context")


def _dependencies(tmp_path) -> MilestoneTwoDependencies:
    """Return the exact dependencies that a resumed workflow may use later."""
    return MilestoneTwoDependencies(
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_DiffBuilder(),
        context_builder=_ContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )


def _new_workflow(
    dependencies: MilestoneTwoDependencies,
) -> MilestoneTwoWorkflow:
    """Start one ordinary run whose handle can later be resumed."""
    return MilestoneTwoWorkflow(
        run_id="m2-durability-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )


def test_resume_reuses_the_authenticated_existing_run(tmp_path) -> None:
    """Recovery attaches to the original run instead of trying to create it again."""
    dependencies = _dependencies(tmp_path)
    first = _new_workflow(dependencies)

    resumed = resume_milestone_two_workflow(
        run_handle=first.run_handle,
        dependencies=dependencies,
    )

    assert resumed is not first
    assert resumed.run_handle == first.run_handle
    assert resumed.prepared_pull_request is None


def test_resume_rejects_a_forged_handle_before_any_workflow_operation(
    tmp_path,
) -> None:
    """A different ownership token cannot attach to an existing evidence run."""
    dependencies = _dependencies(tmp_path)
    first = _new_workflow(dependencies)
    forged = RunHandle(
        run_id=first.run_handle.run_id,
        ownership=RunOwnership.issue(first.run_handle.run_id),
    )

    with pytest.raises(MilestoneTwoTransitionError, match="authenticated"):
        resume_milestone_two_workflow(
            run_handle=forged,
            dependencies=dependencies,
        )


def test_resume_requires_the_typed_handle_and_dependency_bundle(tmp_path) -> None:
    """Recovery rejects untyped input before it can touch the artifact recorder."""
    dependencies = _dependencies(tmp_path)

    with pytest.raises(TypeError, match="RunHandle"):
        resume_milestone_two_workflow(
            run_handle=object(),  # type: ignore[arg-type]
            dependencies=dependencies,
        )

    ownership = RunOwnership.issue("m2-type-check-run")
    handle = RunHandle(run_id="m2-type-check-run", ownership=ownership)

    with pytest.raises(TypeError, match="MilestoneTwoDependencies"):
        resume_milestone_two_workflow(
            run_handle=handle,
            dependencies=object(),  # type: ignore[arg-type]
        )
