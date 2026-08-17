"""Tests for human risk approval in the Milestone 2 workflow."""

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
    """Return a fixed snapshot and a controlled freshness result."""

    def __init__(self, snapshot: object, freshness_status: str) -> None:
        self.snapshot = snapshot
        self.freshness_status = freshness_status
        self.rechecked: list[object] = []

    def acquire(self, pr_url: str) -> object:
        return self.snapshot

    def recheck(self, snapshot: object) -> object:
        self.rechecked.append(snapshot)
        return SimpleNamespace(status=self.freshness_status)


class _DiffBuilder:
    """Return the fixed three-diff inventory required by preparation."""

    def __init__(self, diffs: tuple[object, object, object]) -> None:
        self.diffs = diffs

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        return self.diffs


class _ContextBuilder:
    """Return the fixed bounded context required by preparation."""

    def __init__(self, context: object) -> None:
        self.context = context

    def build(
        self,
        *,
        snapshot: object,
        diffs: object,
        store: object,
        limits: ContextLimits,
    ) -> object:
        return self.context


def _risk_ready_workflow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    freshness_status: str,
) -> tuple[MilestoneTwoWorkflow, object, _SnapshotAcquirer]:
    """Prepare a workflow and make one locally grounded assessment available."""
    snapshot = object()
    diffs = (object(), object(), object())
    context = object()
    evidence_envelope = object()
    assessment = object()
    acquirer = _SnapshotAcquirer(snapshot, freshness_status)

    workflow = MilestoneTwoWorkflow(
        run_id="m2-run-1",
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=acquirer,
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/7312")

    def fake_generate_risk_assessment(
        *,
        snapshot: object,
        diffs: object,
        context: object,
        evidence_envelope: object,
        gateway: object,
    ) -> tuple[object, object]:
        return object(), object()

    def fake_validate_risk_assessment(
        *,
        draft: object,
        snapshot: object,
        context: object,
        evidence_envelope: object,
    ) -> tuple[object, object]:
        return assessment, object()

    monkeypatch.setattr(
        milestone_two,
        "build_risk_evidence",
        lambda **_kwargs: SimpleNamespace(envelope=evidence_envelope),
    )

    monkeypatch.setattr(
        milestone_two,
        "generate_risk_assessment",
        fake_generate_risk_assessment,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "validate_risk_assessment",
        fake_validate_risk_assessment,
        raising=False,
    )

    assert workflow.propose_risks() is assessment
    return workflow, assessment, acquirer


def test_approve_risk_rechecks_freshness_then_creates_human_review(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current snapshot is required before an approved risk may be recorded."""
    workflow, assessment, acquirer = _risk_ready_workflow(
        tmp_path,
        monkeypatch,
        freshness_status="current",
    )
    review = object()

    def fake_create_human_review(
        *,
        assessment: object,
        hypothesis_id: str,
        edits: dict[str, object],
        selected_anchor_ids: tuple[str, ...],
        reviewed_at: object,
    ) -> object:
        assert assessment is expected_assessment
        assert hypothesis_id == "risk-1"
        assert edits == {"actor": "A reviewed actor"}
        assert selected_anchor_ids == ("anchor-1",)
        return review

    expected_assessment = assessment
    monkeypatch.setattr(
        milestone_two,
        "create_human_review",
        fake_create_human_review,
        raising=False,
    )

    assert (
        workflow.approve_risk(
            "risk-1",
            {"actor": "A reviewed actor"},
            ("anchor-1",),
        )
        is review
    )
    assert workflow.human_reviewed_risk is review
    assert acquirer.rechecked == [acquirer.snapshot]


def test_approve_risk_rejects_a_stale_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed PR cannot receive approval based on old frozen evidence."""
    workflow, _assessment, acquirer = _risk_ready_workflow(
        tmp_path,
        monkeypatch,
        freshness_status="stale",
    )

    with pytest.raises(MilestoneTwoTransitionError, match="snapshot_stale"):
        workflow.approve_risk("risk-1", {}, ("anchor-1",))

    assert acquirer.rechecked == [acquirer.snapshot]
