"""Tests for frozen-evidence refinement in the Milestone 2 workflow."""

from types import SimpleNamespace

import pytest

from triageguard.analysis.context import ContextLimits
from triageguard.config import Settings
from triageguard.domain import (
    EnvironmentKind,
    EvidenceRefinementResult,
    FrozenEvidenceNeed,
    MilestoneTwoStatus,
)
from triageguard.evidence import FrozenEvidenceResolution
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

    def resolve(self, **kwargs: object) -> FrozenEvidenceResolution:
        self.calls.append(kwargs)
        return FrozenEvidenceResolution(
            context=self.refined_context,
            refinement=self.refinement,
        )


class _RecordingResolver:
    """Return one bounded resolution and retain the workflow's exact request."""

    def __init__(self, resolution: FrozenEvidenceResolution) -> None:
        self.resolution = resolution
        self.calls: list[dict[str, object]] = []

    def resolve(self, **kwargs: object) -> FrozenEvidenceResolution:
        self.calls.append(kwargs)
        return self.resolution


def _need() -> FrozenEvidenceNeed:
    return FrozenEvidenceNeed(
        need_id="need-has-privilege",
        category="authorization",
        search_terms=("hasPrivilege",),
        explanation="Find the exact frozen authorization decision.",
        supporting_anchor_ids=("anchor-visible",),
    )


def _result(*, exhausted: bool = False) -> EvidenceRefinementResult:
    return EvidenceRefinementResult.from_content(
        parent_context_sha256="a" * 64,
        successor_context_sha256="a" * 64,
        requested_need_sha256="b" * 64,
        priority_anchor_ids=() if exhausted else ("anchor-hidden",),
        added_anchor_ids=(),
        round_number=1,
        exhausted=exhausted,
        reason_code=(
            "frozen_evidence_exhausted" if exhausted else "catalog_evidence_prioritized"
        ),
    )


def test_refinement_cannot_continue_after_an_exhausted_result(tmp_path) -> None:
    """A second exhaustion record would make the durable chain unrecoverable."""
    snapshot = object()
    context = object()
    exhausted = _result(exhausted=True)
    resolver = _RecordingResolver(
        FrozenEvidenceResolution(context=context, refinement=exhausted)
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-already-exhausted-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
        evidence_refiner=resolver,
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=(object(), object(), object()),
        context=context,
    )
    workflow._risk_assessment = SimpleNamespace(
        outcome="insufficient_context_to_assess",
        evidence_needs=(_need(),),
    )
    workflow._context_refinements = [exhausted]
    workflow._state = milestone_two._State.EVIDENCE_REFINEMENT_REQUIRED

    with pytest.raises(milestone_two.MilestoneTwoTransitionError, match="exhausted"):
        workflow.refine_frozen_evidence()

    assert resolver.calls == []


def test_risk_level_refinement_is_persisted_before_downstream_state_is_cleared(
    tmp_path,
    monkeypatch,
) -> None:
    """Moving persistence after mutation must expose cleared causal inputs."""
    snapshot = object()
    original_context = object()
    refined_context = object()
    need = _need()
    risk_assessment = SimpleNamespace(
        outcome="insufficient_context_to_assess",
        evidence_needs=(need,),
    )
    refinement = _result()
    resolver = _RecordingResolver(
        FrozenEvidenceResolution(
            context=refined_context,
            refinement=refinement,
        )
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-risk-refinement-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
        evidence_refiner=resolver,
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=(object(), object(), object()),
        context=original_context,
    )
    workflow._risk_assessment = risk_assessment
    workflow._risk_evidence_envelope = object()
    workflow._risk_draft = object()
    workflow._risk_response = object()
    workflow._state = milestone_two._State.EVIDENCE_REFINEMENT_REQUIRED
    persisted: list[object] = []

    monkeypatch.setattr(workflow, "_is_typed_prepared", lambda _prepared: True)

    def fake_persist(**values: object) -> None:
        assert workflow._risk_assessment is risk_assessment
        assert workflow.prepared_pull_request is not None
        assert workflow.prepared_pull_request.context is original_context
        assert values["needs"] == (need,)
        assert values["refinement"] == refinement
        persisted.append(values)

    monkeypatch.setattr(
        workflow,
        "_persist_evidence_refinement",
        fake_persist,
        raising=False,
    )

    assert workflow.refine_frozen_evidence() == refinement
    assert len(persisted) == 1
    assert workflow.prepared_pull_request is not None
    assert workflow.prepared_pull_request.context is refined_context
    assert workflow.risk_assessment is None
    assert workflow.human_reviewed_risk is None
    assert workflow.testability_assessment is None
    assert workflow.gherkin_candidate is None
    assert workflow._refinement_priority_anchor_ids == ("anchor-hidden",)
    assert workflow._state is milestone_two._State.PREPARED
    assert resolver.calls[0]["needs"] == (need,)
    assert resolver.calls[0]["completed_rounds"] == 0
    assert resolver.calls[0]["max_rounds"] == 2


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
    refinement = SimpleNamespace(
        exhausted=False,
        priority_anchor_ids=(),
        added_anchor_ids=("anchor-added",),
    )
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
            "needs": assessment.evidence_needs,
            "store": workflow._store,
            "limits": ContextLimits.from_settings(workflow._settings),
            "completed_rounds": 0,
            "max_rounds": workflow._settings.max_model_evidence_rounds,
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
    refinement = SimpleNamespace(
        exhausted=True,
        priority_anchor_ids=(),
        added_anchor_ids=(),
    )
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


def test_exhausted_risk_refinement_seals_only_insufficient_context(
    tmp_path,
    monkeypatch,
) -> None:
    """A risk-stage exhausted search must never be recorded as a no-risk result."""
    snapshot = object()
    context = object()
    risk_assessment = SimpleNamespace(
        outcome="insufficient_context_to_assess",
        reason_code="analysis_limit_exceeded",
        evidence_needs=(_need(),),
    )
    refinement = _result(exhausted=True)
    terminal_marker = object()
    finalized: list[object] = []

    def fake_terminal_record(**values: object) -> object:
        assert values["status"] is MilestoneTwoStatus.INSUFFICIENT_CONTEXT_TO_ASSESS
        assert values["reason_code"] == "analysis_limit_exceeded"
        assert values["risk_assessment"] is risk_assessment
        assert values["human_reviewed_risk"] is None
        assert values["testability_assessment"] is None
        assert values["context_refinements"] == (refinement,)
        return terminal_marker

    monkeypatch.setattr(milestone_two, "MilestoneTwoRunRecord", fake_terminal_record)
    monkeypatch.setattr(
        ArtifactRecorder,
        "finalize_run",
        lambda _recorder, _handle, record: finalized.append(record),
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-risk-exhaustion-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=_SnapshotAcquirer(),
        diff_builder=_UnusedDiffBuilder(),
        context_builder=_UnusedContextBuilder(),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow._prepared = PreparedPullRequest(
        snapshot=snapshot,
        diffs=(object(), object(), object()),
        context=context,
    )
    workflow._risk_assessment = risk_assessment
    workflow._context_refinements = [refinement]
    workflow._state = milestone_two._State.EVIDENCE_REFINEMENT_REQUIRED

    assert workflow.finish_with_insufficient_frozen_evidence() is terminal_marker
    assert finalized == [terminal_marker]
