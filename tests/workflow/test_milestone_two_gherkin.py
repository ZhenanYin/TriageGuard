"""Tests for Gherkin generation in the Milestone 2 workflow."""

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
    """Return a fixed snapshot and one controlled freshness result per recheck."""

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


def _risk_approved_workflow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    freshness_statuses: tuple[str, ...],
) -> tuple[MilestoneTwoWorkflow, object, _SnapshotAcquirer]:
    """Prepare, ground, and human-approve one risk without external I/O."""
    snapshot = object()
    diffs = (object(), object(), object())
    context = object()
    assessment = object()
    human_review = object()
    acquirer = _SnapshotAcquirer(snapshot, freshness_statuses)

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
    monkeypatch.setattr(
        milestone_two,
        "create_human_review",
        lambda **_kwargs: human_review,
        raising=False,
    )

    assert workflow.propose_risks() is assessment
    assert workflow.approve_risk("risk-1", {}, ("anchor-1",)) is human_review
    return workflow, human_review, acquirer


def _assess_testability(
    workflow: MilestoneTwoWorkflow,
    human_review: object,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Move one approved risk through a locally accepted testability decision."""
    assessment = SimpleNamespace(decision="testable_from_frozen_evidence")
    draft = object()
    response = object()

    def fake_generate_testability_assessment(
        *,
        human_review: object,
        context: object,
        gateway: object,
    ) -> tuple[object, object]:
        assert human_review is expected_human_review
        assert context is workflow._prepared.context
        assert gateway is workflow._gateway
        return draft, response

    def fake_validate_testability_assessment(
        *,
        draft: object,
        human_review: object,
        context: object,
    ) -> tuple[object, object]:
        assert draft is expected_draft
        assert human_review is expected_human_review
        assert context is workflow._prepared.context
        return assessment, object()

    expected_human_review = human_review
    expected_draft = draft
    monkeypatch.setattr(
        milestone_two,
        "generate_testability_assessment",
        fake_generate_testability_assessment,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "validate_testability_assessment",
        fake_validate_testability_assessment,
        raising=False,
    )

    assert workflow.assess_testability() is assessment
    return assessment


def test_generate_gherkin_rechecks_freshness_and_uses_approved_risk(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gherkin request receives the exact risk approved by the human."""
    workflow, human_review, acquirer = _risk_approved_workflow(
        tmp_path,
        monkeypatch,
        freshness_statuses=("current", "current", "current"),
    )
    testability_assessment = _assess_testability(
        workflow,
        human_review,
        monkeypatch,
    )
    candidate = SimpleNamespace(gherkin_text="Feature: generated scenario")
    response = object()

    def fake_generate_gherkin(
        *,
        human_review: object,
        testability_assessment: object,
        context: object,
        gateway: object,
    ) -> tuple[object, object]:
        assert human_review is expected_human_review
        assert testability_assessment is expected_testability_assessment
        assert context is workflow._prepared.context
        assert gateway is workflow._gateway
        return candidate, response

    expected_human_review = human_review
    expected_testability_assessment = testability_assessment
    monkeypatch.setattr(
        milestone_two,
        "request_gherkin_candidate",
        fake_generate_gherkin,
        raising=False,
    )
    monkeypatch.setattr(
        milestone_two,
        "validate_gherkin_candidate",
        lambda **_kwargs: SimpleNamespace(approved=True),
        raising=False,
    )

    assert workflow.generate_gherkin() is candidate
    assert workflow.gherkin_candidate is candidate
    assert acquirer.rechecked == [
        acquirer.snapshot,
        acquirer.snapshot,
        acquirer.snapshot,
    ]


def test_generate_gherkin_rejects_a_stale_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed PR cannot produce a Gherkin candidate from stale evidence."""
    workflow, human_review, acquirer = _risk_approved_workflow(
        tmp_path,
        monkeypatch,
        freshness_statuses=("current", "current", "stale"),
    )
    _assess_testability(workflow, human_review, monkeypatch)

    with pytest.raises(MilestoneTwoTransitionError, match="snapshot_stale"):
        workflow.generate_gherkin()

    assert acquirer.rechecked == [
        acquirer.snapshot,
        acquirer.snapshot,
        acquirer.snapshot,
    ]


def test_generate_gherkin_requires_a_testability_assessment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Risk approval alone must not bypass the frozen-evidence testability gate."""
    workflow, _human_review, acquirer = _risk_approved_workflow(
        tmp_path,
        monkeypatch,
        freshness_statuses=("current", "current"),
    )

    with pytest.raises(
        MilestoneTwoTransitionError,
        match="assess testability",
    ):
        workflow.generate_gherkin()

    assert acquirer.rechecked == [acquirer.snapshot]
