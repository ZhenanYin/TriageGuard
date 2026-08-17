"""Tests for non-risk terminal outcomes and freshness in Milestone 2."""

from types import SimpleNamespace

import pytest

from triageguard.analysis.context import ContextLimits
from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.workflow import milestone_two
from triageguard.workflow.milestone_two import (
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
)


class _SnapshotAcquirer:
    """Return one fixed snapshot and controlled recheck statuses."""

    def __init__(self, snapshot: object, statuses: tuple[str, ...]) -> None:
        self.snapshot = snapshot
        self._statuses = list(statuses)
        self.rechecked: list[object] = []

    def acquire(self, pr_url: str) -> object:
        return self.snapshot

    def recheck(self, snapshot: object) -> object:
        self.rechecked.append(snapshot)
        return SimpleNamespace(status=self._statuses.pop(0))


class _DiffBuilder:
    """Return the fixed three-diff inventory required by preparation."""

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        return (object(), object(), object())


class _ContextBuilder:
    """Return one fixed bounded context."""

    def build(
        self,
        *,
        snapshot: object,
        diffs: object,
        store: object,
        limits: ContextLimits,
    ) -> object:
        return object()


def _workflow(
    tmp_path,
    *,
    statuses: tuple[str, ...] = (),
) -> tuple[MilestoneTwoWorkflow, _SnapshotAcquirer]:
    """Create one empty workflow without external I/O."""
    snapshot = object()
    acquirer = _SnapshotAcquirer(snapshot, statuses)
    workflow = MilestoneTwoWorkflow(
        run_id="m2-terminal-run",
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=acquirer,
        diff_builder=_DiffBuilder(),
        context_builder=_ContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )
    return workflow, acquirer


def _risks_ready_workflow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcome: str,
    statuses: tuple[str, ...],
) -> tuple[MilestoneTwoWorkflow, object, _SnapshotAcquirer]:
    """Prepare one workflow with one locally validated non-risk assessment."""
    workflow, acquirer = _workflow(tmp_path, statuses=statuses)
    assessment = SimpleNamespace(
        outcome=outcome,
        rationale="The bounded evidence did not identify a meaningful risk.",
        reason_code="analysis_limit_exceeded",
    )

    monkeypatch.setattr(
        milestone_two,
        "build_risk_evidence",
        lambda **_kwargs: SimpleNamespace(envelope=object()),
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "generate_risk_assessment",
        lambda **_kwargs: (object(), object()),
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "validate_risk_assessment",
        lambda **_kwargs: (assessment, object()),
        raising=False,
    )

    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/7312")
    assert workflow.propose_risks() is assessment
    return workflow, assessment, acquirer


def test_finish_without_risk_seals_only_supported_nonrisk_outcomes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human may explicitly finish only a valid bounded no-risk result."""
    workflow, assessment, acquirer = _risks_ready_workflow(
        tmp_path,
        monkeypatch,
        outcome="no_meaningful_security_risk_found",
        statuses=("current",),
    )
    terminal_record = object()
    finalized: list[object] = []

    def fake_terminal_record(**kwargs: object) -> object:
        assert kwargs["run_id"] == "m2-terminal-run"
        assert kwargs["snapshot"] is acquirer.snapshot
        assert kwargs["status"].value == "no_meaningful_security_risk_found"
        assert kwargs["reason_code"] == "no_meaningful_security_risk_found"
        assert kwargs["risk_assessment"] is assessment
        assert kwargs["human_reviewed_risk"] is None
        assert kwargs["gherkin_candidate"] is None
        assert kwargs["gherkin_approval"] is None
        return terminal_record

    def fake_finalize(
        _self: ArtifactRecorder,
        handle: object,
        record: object,
    ) -> object:
        assert handle is workflow.run_handle
        assert record is terminal_record
        finalized.append(record)
        return object()

    monkeypatch.setattr(
        milestone_two,
        "MilestoneTwoRunRecord",
        fake_terminal_record,
        raising=False,
    )
    monkeypatch.setattr(ArtifactRecorder, "finalize_run", fake_finalize)

    assert workflow.finish_without_risk() is terminal_record
    assert workflow.terminal_record is terminal_record
    assert acquirer.rechecked == [acquirer.snapshot]
    assert finalized == [terminal_record]


def test_finish_without_risk_rejects_a_risk_proposal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposed risk must go through human review, not the non-risk exit."""
    workflow, _assessment, _acquirer = _risks_ready_workflow(
        tmp_path,
        monkeypatch,
        outcome="risks_proposed",
        statuses=(),
    )

    with pytest.raises(MilestoneTwoTransitionError, match="non-risk"):
        workflow.finish_without_risk()


def test_finish_without_risk_rejects_a_stale_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed PR cannot be finalized as a no-risk result."""
    workflow, _assessment, acquirer = _risks_ready_workflow(
        tmp_path,
        monkeypatch,
        outcome="no_meaningful_security_risk_found",
        statuses=("stale",),
    )

    with pytest.raises(MilestoneTwoTransitionError, match="snapshot_stale"):
        workflow.finish_without_risk()

    assert acquirer.rechecked == [acquirer.snapshot]


def test_freshness_requires_preparation_and_returns_a_recheck(tmp_path) -> None:
    """The UI can request the currentness result only after a PR is frozen."""
    workflow, _acquirer = _workflow(tmp_path)

    with pytest.raises(MilestoneTwoTransitionError, match="prepare"):
        workflow.freshness()

    prepared_workflow, prepared_acquirer = _workflow(
        tmp_path / "prepared",
        statuses=("current",),
    )
    prepared_workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/7312")

    result = prepared_workflow.freshness()

    assert result.status == "current"
    assert prepared_acquirer.rechecked == [prepared_acquirer.snapshot]
