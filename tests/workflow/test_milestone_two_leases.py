"""Tests for cross-process serialization of Milestone 2 workflow actions."""

from contextlib import contextmanager

import pytest

from triageguard.analysis.context import ContextLimits
from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.workflow.milestone_two import MilestoneTwoWorkflow


class _SnapshotAcquirer:
    """Return one placeholder snapshot without external I/O."""

    def acquire(self, pr_url: str) -> object:
        return object()

    def recheck(self, snapshot: object) -> object:
        raise AssertionError("this test only prepares a pull request")


class _DiffBuilder:
    """Return three placeholder comparison artifacts."""

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        return (object(), object(), object())


class _ContextBuilder:
    """Return one placeholder bounded context."""

    def build(
        self,
        *,
        snapshot: object,
        diffs: object,
        store: object,
        limits: ContextLimits,
    ) -> object:
        return object()


def test_prepare_pr_holds_the_recorder_workflow_lease(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public workflow transition must be serialized across local processes."""
    recorder = ArtifactRecorder(tmp_path)
    workflow = MilestoneTwoWorkflow(
        run_id="m2-lease-run",
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_DiffBuilder(),
        context_builder=_ContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )
    entered: list[object] = []

    @contextmanager
    def fake_workflow_lease(_self, handle: object):
        entered.append(handle)
        yield

    monkeypatch.setattr(
        ArtifactRecorder,
        "workflow_lease",
        fake_workflow_lease,
    )

    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/7312")

    assert entered == [workflow.run_handle]
