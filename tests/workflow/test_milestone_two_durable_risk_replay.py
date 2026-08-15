"""Tests that saved Milestone 2 risk responses survive interruption."""

import json
from datetime import UTC, datetime

import pytest

from triageguard.config import Settings
from triageguard.domain import (
    ContextAnchor,
    ContextBundle,
    DiffArtifact,
    EnvironmentKind,
    PullRequestSnapshot,
    SnapshotFreshness,
)
from triageguard.llm import ModelRequest, ModelResponse, ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.workflow.milestone_two import (
    MilestoneTwoDependencies,
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
    resume_milestone_two_workflow,
)


class _CountingGateway:
    """Count each model request while using a fixed replay response."""

    def __init__(self, response: dict[str, object]) -> None:
        self._replay = ReplayGateway({"risk_hypothesis": response})
        self.call_count = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        return self._replay.generate(request)


class _InterruptAfterRiskArtifactRecorder(ArtifactRecorder):
    """Raise only after the completed model-response bytes were durably written."""

    def __init__(self, root_directory) -> None:
        super().__init__(root_directory)
        self.interrupt_after_risk_response = True

    def write_artifact(self, handle, name, content, provenance):
        result = super().write_artifact(handle, name, content, provenance)
        if (
            self.interrupt_after_risk_response
            and name == "artifacts/workflow/risk_generation.json"
        ):
            self.interrupt_after_risk_response = False
            raise OSError("simulated interruption after durable model response")
        return result


class _SnapshotAcquirer:
    """Return one frozen snapshot and currentness evidence for it."""

    def __init__(self, snapshot: PullRequestSnapshot) -> None:
        self.snapshot = snapshot

    def acquire(self, pr_url: str) -> PullRequestSnapshot:
        return self.snapshot

    def recheck(self, snapshot: PullRequestSnapshot) -> SnapshotFreshness:
        assert snapshot is self.snapshot
        return SnapshotFreshness(
            snapshot_key=snapshot.snapshot_key,
            status="current",
            reason_code="snapshot_current",
            checked_at=datetime(2026, 8, 15, tzinfo=UTC),
            observed_base_sha=snapshot.base_sha,
            observed_head_sha=snapshot.head_sha,
            observed_candidate_sha=snapshot.candidate_sha,
        )


class _DiffBuilder:
    """Return the exact frozen three-way comparison inventory."""

    def __init__(
        self,
        diffs: tuple[DiffArtifact, DiffArtifact, DiffArtifact],
    ) -> None:
        self.diffs = diffs

    def build_all(
        self,
        snapshot: PullRequestSnapshot,
    ) -> tuple[DiffArtifact, DiffArtifact, DiffArtifact]:
        return self.diffs


class _ContextBuilder:
    """Return one exact bounded evidence catalog."""

    def __init__(self, context: ContextBundle) -> None:
        self.context = context

    def build(self, **kwargs: object) -> ContextBundle:
        return self.context


def _snapshot() -> PullRequestSnapshot:
    """Build one valid synthetic frozen identity for a storage-only test."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=900000001,
        pull_url="https://github.com/openmrs/openmrs-core/pull/900000001",
        state="open",
        default_branch="main",
        base_branch="main",
        merge_base_sha="1" * 40,
        base_sha="2" * 40,
        head_sha="3" * 40,
        candidate_sha="4" * 40,
        merge_base_tree_sha="5" * 40,
        base_tree_sha="6" * 40,
        head_tree_sha="7" * 40,
        candidate_tree_sha="8" * 40,
        acquired_at=datetime(2026, 8, 15, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="9" * 64,
    )


def _diff(
    *,
    kind: str,
    old_revision: str,
    new_revision: str,
    digest: str,
) -> DiffArtifact:
    """Build one valid empty diff whose identity is still reproducible."""
    return DiffArtifact(
        kind=kind,
        old_revision=old_revision,
        new_revision=new_revision,
        git_arguments=("diff", "--no-ext-diff"),
        git_version="2.47.1",
        files=(),
        patch_sha256=digest,
        artifact_sha256=digest,
    )


def _context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Build one valid integration anchor for the model request."""
    anchor = ContextAnchor(
        anchor_id="anchor-integration",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="a" * 40,
        path="api/src/main/java/org/openmrs/PatientService.java",
        java_symbol="purgePatient",
        start_line=10,
        end_line=10,
        text="void purgePatient() { requirePrivilege(); }",
        text_sha256="e23acd10a13fbe8105c9d4cd95820585882c3d2a7bc67cf80f0451e77e1e7066",
        selection_reason="primary integration change",
        score_components=(),
        change_relation="integration_change",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(anchor,),
        selected_file_count=1,
        selected_anchor_count=1,
        selected_bytes=len(anchor.text.encode("utf-8")),
        max_files=40,
        max_anchors=80,
        max_bytes=160_000,
        max_anchor_lines=120,
        max_blob_bytes=1_000_000,
        max_search_identifiers=100,
        max_hits_per_identifier=20,
        primary_change_represented=True,
    )


def _risk_response(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
) -> dict[str, object]:
    """Return a schema-valid insufficient-context response for replay."""
    return {
        "snapshot_key": snapshot.snapshot_key,
        "context_sha256": context.context_sha256,
        "outcome": "insufficient_context_to_assess",
        "hypotheses": [],
        "rationale": None,
        "security_relevant_areas": [],
        "supporting_anchor_ids": [],
        "coverage_limitations": [],
        "reason_code": "analysis_limit_exceeded",
        "missing_evidence": ["The authorization configuration is not present."],
        "needed_evidence": ["The applicable authorization configuration."],
        "generated_at": "2026-08-15T00:00:00Z",
    }


def test_resume_reuses_a_durable_risk_response_without_another_model_call(
    tmp_path,
) -> None:
    """An interruption after response storage must not cause a second LLM call."""
    snapshot = _snapshot()
    context = _context(snapshot)
    diffs = (
        _diff(
            kind="author_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.head_sha,
            digest="a" * 64,
        ),
        _diff(
            kind="integration_diff",
            old_revision=snapshot.base_sha,
            new_revision=snapshot.candidate_sha,
            digest="b" * 64,
        ),
        _diff(
            kind="base_drift_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.base_sha,
            digest="c" * 64,
        ),
    )
    recorder = _InterruptAfterRiskArtifactRecorder(tmp_path)
    gateway = _CountingGateway(_risk_response(snapshot, context))
    dependencies = MilestoneTwoDependencies(
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=gateway,
    )
    first = MilestoneTwoWorkflow(
        run_id="m2-risk-replay-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )

    first.prepare_pr("https://github.com/openmrs/openmrs-core/pull/900000001")

    with pytest.raises(OSError, match="simulated interruption"):
        first.propose_risks()

    assert gateway.call_count == 1

    resumed = resume_milestone_two_workflow(
        run_handle=first.run_handle,
        dependencies=dependencies,
    )
    assessment = resumed.risk_assessment

    assert assessment is not None
    assert assessment.outcome == "insufficient_context_to_assess"
    assert gateway.call_count == 1


def test_nonrisk_terminal_measurements_bind_every_planned_stage(tmp_path) -> None:
    """Final research measurements contain facts for all Milestone 2 stages."""
    snapshot = _snapshot()
    context = _context(snapshot)
    diffs = (
        _diff(
            kind="author_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.head_sha,
            digest="a" * 64,
        ),
        _diff(
            kind="integration_diff",
            old_revision=snapshot.base_sha,
            new_revision=snapshot.candidate_sha,
            digest="b" * 64,
        ),
        _diff(
            kind="base_drift_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.base_sha,
            digest="c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    gateway = _CountingGateway(_risk_response(snapshot, context))
    dependencies = MilestoneTwoDependencies(
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=gateway,
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-measurements-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )

    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/900000001")
    assessment = workflow.propose_risks()
    record = workflow.finish_without_risk()

    measurements = json.loads(
        recorder.read_artifact(
            workflow.run_handle,
            "artifacts/measurements/final.json",
        )
    )

    assert record.status.value == "insufficient_context_to_assess"
    assert assessment.outcome == "insufficient_context_to_assess"
    assert set(measurements) == {
        "acquisition",
        "diffs",
        "context",
        "risk_generation",
        "human_review",
        "gherkin_generation",
        "staleness",
        "end_to_end",
    }
    assert measurements["acquisition"]["snapshot_key"] == snapshot.snapshot_key
    assert measurements["diffs"]["comparison_count"] == 3
    assert measurements["context"]["selected_anchor_count"] == 1
    assert measurements["risk_generation"]["model_call_count"] == 1
    assert measurements["human_review"]["status"] == "not_applicable"
    assert measurements["gherkin_generation"]["status"] == "not_applicable"
    assert measurements["staleness"]["status"] == "current"
    assert (
        measurements["end_to_end"]["terminal_status"]
        == "insufficient_context_to_assess"
    )


def test_resume_reloads_a_finalized_nonrisk_record(tmp_path) -> None:
    """A sealed non-risk outcome remains final after a process restart."""
    snapshot = _snapshot()
    context = _context(snapshot)
    diffs = (
        _diff(
            kind="author_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.head_sha,
            digest="a" * 64,
        ),
        _diff(
            kind="integration_diff",
            old_revision=snapshot.base_sha,
            new_revision=snapshot.candidate_sha,
            digest="b" * 64,
        ),
        _diff(
            kind="base_drift_diff",
            old_revision=snapshot.merge_base_sha,
            new_revision=snapshot.base_sha,
            digest="c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    gateway = _CountingGateway(_risk_response(snapshot, context))
    dependencies = MilestoneTwoDependencies(
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
        ),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=gateway,
    )
    first = MilestoneTwoWorkflow(
        run_id="m2-terminal-recovery-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )

    first.prepare_pr("https://github.com/openmrs/openmrs-core/pull/900000001")
    first.propose_risks()
    sealed = first.finish_without_risk()

    resumed = resume_milestone_two_workflow(
        run_handle=first.run_handle,
        dependencies=dependencies,
    )

    assert resumed.terminal_record == sealed
    assert gateway.call_count == 1

    with pytest.raises(MilestoneTwoTransitionError, match="already finalized"):
        resumed.freshness()
