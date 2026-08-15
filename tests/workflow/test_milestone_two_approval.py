"""Tests for terminal Gherkin approval in the Milestone 2 workflow."""

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
    """Return a fixed snapshot and controlled results for each freshness check."""

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
    """Return the three fixed evidence differences required for preparation."""

    def __init__(self, diffs: tuple[object, object, object]) -> None:
        self.diffs = diffs

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        return self.diffs


class _ContextBuilder:
    """Return the fixed bounded context required for preparation."""

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


def _gherkin_ready_workflow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    freshness_statuses: tuple[str, ...],
) -> tuple[MilestoneTwoWorkflow, object, object, _SnapshotAcquirer]:
    """Reach the reviewable-Gherkin stage without external network activity."""
    snapshot = object()
    diffs = (object(), object(), object())
    context = object()
    assessment = object()
    human_review = object()
    candidate = SimpleNamespace(gherkin_text="Feature: proposed scenario")
    acquirer = _SnapshotAcquirer(snapshot, freshness_statuses)

    workflow = MilestoneTwoWorkflow(
        run_id="m2-approval-run",
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
    monkeypatch.setattr(
        milestone_two,
        "create_human_review",
        lambda **_kwargs: human_review,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "request_gherkin_candidate",
        lambda **_kwargs: (candidate, object()),
        raising=False,
    )

    assert workflow.propose_risks() is assessment
    assert workflow.approve_risk("risk-1", {}, ("anchor-1",)) is human_review
    assert workflow.generate_gherkin() is candidate
    return workflow, human_review, candidate, acquirer


def test_approve_gherkin_applies_edit_then_seals_terminal_record(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human-approved edited scenario becomes the sealed terminal evidence."""
    workflow, human_review, candidate, acquirer = _gherkin_ready_workflow(
        tmp_path,
        monkeypatch,
        freshness_statuses=("current", "current", "current"),
    )
    edited_text = "Feature: reviewer-approved scenario"
    edited_candidate = SimpleNamespace(gherkin_text=edited_text)
    approval = SimpleNamespace(
        candidate_id="gherkin-approved",
        candidate_sha256="a" * 64,
    )
    terminal_record = object()
    events: list[tuple[object, object, object]] = []
    finalized: list[object] = []

    def fake_apply_edit(
        *,
        candidate: object,
        text: str,
        human_review: object,
    ) -> object:
        assert candidate is expected_candidate
        assert text == edited_text
        assert human_review is expected_human_review
        return edited_candidate

    def fake_approve(
        *,
        candidate: object,
        human_review: object,
        approved_at: object,
    ) -> object:
        assert candidate is edited_candidate
        assert human_review is expected_human_review
        assert approved_at is not None
        return approval

    def fake_terminal_record(**kwargs: object) -> object:
        assert kwargs["run_id"] == "m2-approval-run"
        assert kwargs["snapshot"] is acquirer.snapshot
        assert kwargs["risk_assessment"] is workflow.risk_assessment
        assert kwargs["human_reviewed_risk"] is expected_human_review
        assert kwargs["gherkin_candidate"] is edited_candidate
        assert kwargs["gherkin_approval"] is approval
        return terminal_record

    def fake_record_event(
        _self: ArtifactRecorder,
        handle: object,
        event_type: object,
        payload: object,
    ) -> None:
        events.append((handle, event_type, payload))

    def fake_finalize(
        _self: ArtifactRecorder,
        handle: object,
        record: object,
    ) -> object:
        assert handle is workflow.run_handle
        assert record is terminal_record
        finalized.append(record)
        return object()

    expected_candidate = candidate
    expected_human_review = human_review
    monkeypatch.setattr(
        milestone_two,
        "apply_gherkin_text_edit",
        fake_apply_edit,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "approve_gherkin_candidate",
        fake_approve,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "MilestoneTwoRunRecord",
        fake_terminal_record,
        raising=False,
    )
    monkeypatch.setattr(ArtifactRecorder, "record_event", fake_record_event)
    monkeypatch.setattr(ArtifactRecorder, "finalize_run", fake_finalize)

    assert workflow.approve_gherkin(edited_text) is terminal_record
    assert workflow.gherkin_candidate is edited_candidate
    assert workflow.gherkin_approval is approval
    assert workflow.terminal_record is terminal_record
    assert acquirer.rechecked == [acquirer.snapshot] * 3
    assert finalized == [terminal_record]
    assert len(events) == 1


def test_approve_gherkin_rejects_a_stale_snapshot_before_editing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed PR cannot receive final Gherkin approval from stale evidence."""
    workflow, _human_review, _candidate, acquirer = _gherkin_ready_workflow(
        tmp_path,
        monkeypatch,
        freshness_statuses=("current", "current", "stale"),
    )

    with pytest.raises(MilestoneTwoTransitionError, match="snapshot_stale"):
        workflow.approve_gherkin("Feature: reviewer-approved scenario")

    assert acquirer.rechecked == [acquirer.snapshot] * 3


def test_finalized_workflow_rejects_every_later_transition(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed record prevents preparation, generation, review, or rechecking."""
    workflow, human_review, candidate, acquirer = _gherkin_ready_workflow(
        tmp_path,
        monkeypatch,
        freshness_statuses=("current", "current", "current"),
    )
    approval = SimpleNamespace(
        candidate_id="gherkin-final",
        candidate_sha256="b" * 64,
    )
    terminal_record = object()

    monkeypatch.setattr(
        milestone_two,
        "apply_gherkin_text_edit",
        lambda **_kwargs: candidate,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "approve_gherkin_candidate",
        lambda **_kwargs: approval,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "MilestoneTwoRunRecord",
        lambda **_kwargs: terminal_record,
        raising=False,
    )
    monkeypatch.setattr(
        ArtifactRecorder,
        "record_event",
        lambda _self, _handle, _event_type, _payload: None,
    )
    monkeypatch.setattr(
        ArtifactRecorder,
        "finalize_run",
        lambda _self, _handle, _record: object(),
    )

    assert workflow.approve_gherkin(candidate.gherkin_text) is terminal_record

    actions = (
        lambda: workflow.prepare_pr(
            "https://github.com/openmrs/openmrs-core/pull/7312"
        ),
        workflow.propose_risks,
        lambda: workflow.approve_risk("risk-1", {}, ("anchor-1",)),
        workflow.generate_gherkin,
        lambda: workflow.approve_gherkin(candidate.gherkin_text),
        workflow.finish_without_risk,
        workflow.freshness,
    )

    for action in actions:
        with pytest.raises(MilestoneTwoTransitionError):
            action()

    assert human_review is workflow.human_reviewed_risk
    assert acquirer.rechecked == [acquirer.snapshot] * 3
