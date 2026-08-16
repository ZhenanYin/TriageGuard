"""Tests for frozen-evidence refinement in the Milestone 2 workflow."""

from types import SimpleNamespace

from triageguard.analysis.context import ContextLimits
from triageguard.config import Settings
from triageguard.domain import EnvironmentKind, MilestoneTwoStatus
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.workflow import milestone_two
from triageguard.workflow.milestone_two import (
    MilestoneTwoWorkflow,
    PreparedPullRequest,
)


class _SnapshotAcquirer:
    """Return one current result when refinement rechecks the frozen PR."""

    def __init__(self) -> None:
        self.rechecked: list[object] = []

    def recheck(self, snapshot: object) -> object:
        self.rechecked.append(snapshot)
        return SimpleNamespace(status="current")


class _UnusedDiffBuilder:
    """Preparation is supplied directly in this focused transition test."""

    def build_all(self, snapshot: object) -> tuple[object, object, object]:
        raise AssertionError("the test directly supplies frozen diffs")


class _UnusedContextBuilder:
    """Preparation is supplied directly in this focused transition test."""

    def build(self, **kwargs: object) -> object:
        raise AssertionError("the test directly supplies frozen context")


class _RecordingRefiner:
    """Return one successor frozen context and record its exact inputs."""

    def __init__(
        self,
        refined_context: object,
        refinement: object,
    ) -> None:
        self.refined_context = refined_context
        self.refinement = refinement
        self.calls: list[dict[str, object]] = []

    def refine(self, **kwargs: object) -> tuple[object, object]:
        self.calls.append(kwargs)
        return self.refined_context, self.refinement


def test_refinement_replaces_context_and_clears_downstream_analysis(
    tmp_path,
) -> None:
    """More frozen code evidence starts a new risk proposal from its successor."""
    snapshot = object()
    diffs = (object(), object(), object())
    original_context = object()
    refined_context = object()
    assessment = SimpleNamespace(
        decision="needs_more_frozen_evidence",
        evidence_needs=(object(),),
    )
    refinement = SimpleNamespace(exhausted=False)
    acquirer = _SnapshotAcquirer()
    refiner = _RecordingRefiner(refined_context, refinement)

    workflow = MilestoneTwoWorkflow(
        run_id="m2-refinement-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=acquirer,
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
        evidence_refiner=refiner,
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=diffs,
        context=original_context,
    )
    workflow._risk_draft = object()
    workflow._risk_response = object()
    workflow._risk_assessment = object()
    workflow._human_reviewed_risk = object()
    workflow._testability_draft = object()
    workflow._testability_response = object()
    workflow._testability_assessment = assessment
    workflow._gherkin_candidate = object()
    workflow._gherkin_response = object()
    workflow._gherkin_validation_report = object()
    workflow._state = milestone_two._State.EVIDENCE_REFINEMENT_REQUIRED

    assert workflow.refine_frozen_evidence() is refinement
    assert workflow.prepared_pull_request is not None
    assert workflow.prepared_pull_request.snapshot is snapshot
    assert workflow.prepared_pull_request.diffs == diffs
    assert workflow.prepared_pull_request.context is refined_context
    assert workflow.risk_assessment is None
    assert workflow.human_reviewed_risk is None
    assert workflow.testability_assessment is None
    assert workflow.gherkin_candidate is None
    assert workflow.gherkin_validation_report is None
    assert workflow._state is milestone_two._State.PREPARED
    assert acquirer.rechecked == [snapshot]
    assert refiner.calls == [
        {
            "snapshot": snapshot,
            "context": original_context,
            "assessment": assessment,
            "store": workflow._store,
            "limits": ContextLimits.from_settings(workflow._settings),
            "created_at": refiner.calls[0]["created_at"],
        }
    ]


def test_exhausted_refinement_finishes_with_insufficient_frozen_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    """No additional saved code must end as evidence-insufficient, never safe."""
    snapshot = object()
    original_context = object()
    assessment = SimpleNamespace(
        decision="needs_more_frozen_evidence",
        evidence_needs=(object(),),
    )
    refinement = SimpleNamespace(exhausted=True)
    acquirer = _SnapshotAcquirer()
    refiner = _RecordingRefiner(original_context, refinement)
    finalized: list[tuple[object, object]] = []
    terminal_marker = object()

    def fake_terminal_record(**values: object) -> object:
        assert values["status"] is (
            MilestoneTwoStatus.INSUFFICIENT_FROZEN_EVIDENCE_FOR_SCENARIO
        )
        assert values["reason_code"] == "insufficient_frozen_evidence_for_scenario"
        assert values["explanation"] == (
            "Insufficient frozen code evidence to design an executable scenario."
        )
        assert values["risk_assessment"] is workflow._risk_assessment
        assert values["human_reviewed_risk"] is workflow._human_reviewed_risk
        assert values["gherkin_candidate"] is None
        assert values["gherkin_approval"] is None
        return terminal_marker

    def fake_finalize(_recorder: object, handle: object, record: object) -> None:
        finalized.append((handle, record))

    monkeypatch.setattr(milestone_two, "MilestoneTwoRunRecord", fake_terminal_record)
    monkeypatch.setattr(ArtifactRecorder, "finalize_run", fake_finalize)

    workflow = MilestoneTwoWorkflow(
        run_id="m2-exhausted-refinement-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=acquirer,
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
        evidence_refiner=refiner,
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=(object(), object(), object()),
        context=original_context,
    )
    workflow._risk_assessment = object()
    workflow._human_reviewed_risk = object()
    workflow._testability_assessment = assessment
    workflow._state = milestone_two._State.EVIDENCE_REFINEMENT_REQUIRED

    assert workflow.refine_frozen_evidence() is refinement
    assert workflow.finish_with_insufficient_frozen_evidence() is terminal_marker
    assert workflow.terminal_record is terminal_marker
    assert workflow._state is milestone_two._State.FINALIZED
    assert finalized == [(workflow.run_handle, terminal_marker)]
    assert acquirer.rechecked == [snapshot, snapshot]
