"""Tests for risk-proposal orchestration in the Milestone 2 workflow."""

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
    """Return one opaque frozen snapshot for workflow-wiring tests."""

    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot

    def acquire(self, pr_url: str) -> object:
        return self.snapshot


class _DiffBuilder:
    """Return one opaque three-diff inventory for workflow-wiring tests."""

    def __init__(self, diffs: tuple[object, object, object]) -> None:
        self.diffs = diffs

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        return self.diffs


class _ContextBuilder:
    """Return one opaque bounded context for workflow-wiring tests."""

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


def _prepared_workflow(
    tmp_path,
) -> tuple[
    MilestoneTwoWorkflow,
    object,
    tuple[object, object, object],
    object,
]:
    """Create a workflow that has completed preparation without external I/O."""
    snapshot = object()
    diffs = (object(), object(), object())
    context = object()

    workflow = MilestoneTwoWorkflow(
        run_id="m2-run-1",
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/7312")
    return workflow, snapshot, diffs, context


def test_propose_risks_runs_generation_then_local_grounding(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workflow may expose risks only after the local grounding gate accepts them."""
    workflow, snapshot, diffs, context = _prepared_workflow(tmp_path)
    draft = object()
    response = object()
    assessment = object()
    evidence_envelope = object()
    calls: list[str] = []

    def fake_generate_risk_assessment(
        *,
        snapshot: object,
        diffs: object,
        context: object,
        evidence_envelope: object,
        gateway: object,
    ) -> tuple[object, object]:
        assert snapshot is expected_snapshot
        assert diffs is expected_diffs
        assert context is expected_context
        assert evidence_envelope is expected_evidence_envelope
        assert gateway is workflow._gateway
        calls.append("generate")
        return draft, response

    def fake_validate_risk_assessment(
        *,
        draft: object,
        snapshot: object,
        context: object,
        evidence_envelope: object,
    ) -> tuple[object, object]:
        assert draft is expected_draft
        assert snapshot is expected_snapshot
        assert context is expected_context
        assert evidence_envelope is expected_evidence_envelope
        calls.append("validate")
        return assessment, object()

    expected_snapshot = snapshot
    expected_diffs = diffs
    expected_context = context
    expected_draft = draft
    expected_evidence_envelope = evidence_envelope

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
    assert calls == ["generate", "validate"]


def test_propose_risks_rejects_an_ungrounded_model_draft(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid model proposal must not become a reviewable risk assessment."""
    workflow, _snapshot, _diffs, _context = _prepared_workflow(tmp_path)
    evidence_envelope = object()

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
    ) -> tuple[None, object]:
        return None, object()

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

    with pytest.raises(MilestoneTwoTransitionError, match="risk grounding"):
        workflow.propose_risks()
