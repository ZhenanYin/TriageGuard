"""Tests that Milestone 2 recovery rejects tampered saved evidence."""

import hashlib
from datetime import UTC, datetime

import pytest

from triageguard.config import Settings
from triageguard.domain import (
    ContextAnchor,
    ContextBundle,
    DiffArtifact,
    EnvironmentKind,
    EvidenceRefinementResult,
    FrozenEvidenceNeed,
    PullRequestSnapshot,
    SnapshotFreshness,
)
from triageguard.llm import (
    ModelGatewayError,
    ModelRequest,
    ModelResponse,
    ReplayGateway,
)
from triageguard.provenance import canonical_sha256
from triageguard.research import ArtifactRecorder
from triageguard.workflow.milestone_two import (
    MilestoneTwoDependencies,
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
    resume_milestone_two_workflow,
)


class _SnapshotAcquirer:
    """Return one fixed frozen snapshot."""

    def __init__(self, snapshot: PullRequestSnapshot) -> None:
        self._snapshot = snapshot

    def acquire(self, pr_url: str) -> PullRequestSnapshot:
        return self._snapshot

    def recheck(self, snapshot: PullRequestSnapshot) -> SnapshotFreshness:
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
    """Return the three fixed comparisons for the frozen snapshot."""

    def __init__(self, diffs: tuple[DiffArtifact, DiffArtifact, DiffArtifact]) -> None:
        self._diffs = diffs

    def build_all(
        self,
        snapshot: PullRequestSnapshot,
    ) -> tuple[DiffArtifact, DiffArtifact, DiffArtifact]:
        return self._diffs


class _ContextBuilder:
    """Return one fixed bounded context catalog."""

    def __init__(self, context: ContextBundle) -> None:
        self._context = context

    def build(self, **kwargs: object) -> ContextBundle:
        return self._context


class _NoRiskGateway:
    """Return one bounded no-risk result bound to the request envelope."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        snapshot_key = request.payload["snapshot_key"]
        context_sha256 = request.payload["context_sha256"]
        envelope = request.payload["evidence_envelope"]
        assert isinstance(snapshot_key, str)
        assert isinstance(context_sha256, str)
        assert isinstance(envelope, dict)
        return ReplayGateway(
            {
                "risk_hypothesis": {
                    "snapshot_key": snapshot_key,
                    "context_sha256": context_sha256,
                    "evidence_envelope_sha256": envelope["envelope_sha256"],
                    "outcome": "no_meaningful_security_risk_found",
                    "hypotheses": [],
                    "rationale": "No specific bounded risk remained after refinement.",
                    "security_relevant_areas": ["Authorization behavior."],
                    "supporting_anchor_ids": ["anchor-integration"],
                    "coverage_limitations": [
                        "This bounded result is not proof of safety."
                    ],
                    "reason_code": None,
                    "missing_evidence": [],
                    "evidence_needs": [],
                    "generated_at": "2026-08-15T00:00:00Z",
                }
            }
        ).generate(request)


def _snapshot() -> PullRequestSnapshot:
    """Build one valid synthetic frozen OpenMRS Core PR identity."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=900000002,
        pull_url="https://github.com/openmrs/openmrs-core/pull/900000002",
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


def _diff(kind: str, old_revision: str, new_revision: str, digest: str) -> DiffArtifact:
    """Build one valid empty reproducible diff."""
    return DiffArtifact(
        kind=kind,
        comparison_status="unchanged",
        old_revision=old_revision,
        new_revision=new_revision,
        git_arguments=("diff", "--no-ext-diff"),
        git_version="2.47.1",
        files=(),
        patch_sha256=hashlib.sha256(b"").hexdigest(),
        artifact_sha256=digest,
    )


def _context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Build one valid integration anchor."""
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


def _successor_context(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
) -> ContextBundle:
    """Add one deterministic repository-context anchor to the frozen catalog."""
    text = "boolean hasPrivilege(String privilege) { return true; }"
    anchor = ContextAnchor(
        anchor_id="anchor-hidden-authorization",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="b" * 40,
        path="api/src/main/java/org/openmrs/AuthorizationContext.java",
        java_symbol="hasPrivilege",
        start_line=20,
        end_line=20,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        selection_reason="bounded frozen-evidence refinement",
        score_components=(),
        change_relation="repository_context",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(*context.anchors, anchor),
        selected_file_count=2,
        selected_anchor_count=2,
        selected_bytes=context.selected_bytes + len(text.encode()),
        max_files=context.max_files,
        max_anchors=context.max_anchors,
        max_bytes=context.max_bytes,
        max_anchor_lines=context.max_anchor_lines,
        max_blob_bytes=context.max_blob_bytes,
        max_search_identifiers=context.max_search_identifiers,
        max_hits_per_identifier=context.max_hits_per_identifier,
        primary_change_represented=True,
    )


def test_resume_replays_a_saved_refinement_before_loading_the_next_model_round(
    tmp_path,
) -> None:
    """Ignoring the refinement chain on restart must restore the stale context."""
    snapshot = _snapshot()
    context = _context(snapshot)
    successor = _successor_context(snapshot, context)
    diffs = (
        _diff("author_diff", snapshot.merge_base_sha, snapshot.head_sha, "a" * 64),
        _diff(
            "integration_diff",
            snapshot.base_sha,
            snapshot.candidate_sha,
            "b" * 64,
        ),
        _diff(
            "base_drift_diff",
            snapshot.merge_base_sha,
            snapshot.base_sha,
            "c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    dependencies = MilestoneTwoDependencies(
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-refinement-recovery-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )
    prepared = workflow.prepare_pr(snapshot.pull_url)
    need = FrozenEvidenceNeed(
        need_id="need-has-privilege",
        category="authorization",
        search_terms=("hasPrivilege",),
        explanation="Find the exact frozen authorization decision.",
        supporting_anchor_ids=("anchor-integration",),
    )
    refinement = EvidenceRefinementResult.from_content(
        parent_context_sha256=context.context_sha256,
        successor_context_sha256=successor.context_sha256,
        requested_need_sha256=canonical_sha256([need.model_dump(mode="json")]),
        priority_anchor_ids=(),
        added_anchor_ids=("anchor-hidden-authorization",),
        round_number=1,
        exhausted=False,
        reason_code="frozen_context_extended",
    )
    workflow._persist_evidence_refinement(
        prepared=prepared,
        needs=(need,),
        successor_context=successor,
        refinement=refinement,
        freshness=dependencies.snapshot_acquirer.recheck(snapshot),
    )

    resumed = resume_milestone_two_workflow(
        run_handle=workflow.run_handle,
        dependencies=dependencies,
    )

    assert resumed.prepared_pull_request is not None
    assert resumed.prepared_pull_request.context == successor
    assert resumed.context_refinements == (refinement,)
    assert resumed._refinement_priority_anchor_ids == ("anchor-hidden-authorization",)


def test_persistence_rejects_a_false_added_anchor_inventory(tmp_path) -> None:
    """A resolver cannot claim anchors that are absent from its successor context."""
    snapshot = _snapshot()
    context = _context(snapshot)
    successor = _successor_context(snapshot, context)
    diffs = (
        _diff("author_diff", snapshot.merge_base_sha, snapshot.head_sha, "a" * 64),
        _diff(
            "integration_diff",
            snapshot.base_sha,
            snapshot.candidate_sha,
            "b" * 64,
        ),
        _diff(
            "base_drift_diff",
            snapshot.merge_base_sha,
            snapshot.base_sha,
            "c" * 64,
        ),
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-false-anchor-inventory-run",
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    prepared = workflow.prepare_pr(snapshot.pull_url)
    need = FrozenEvidenceNeed(
        need_id="need-has-privilege",
        category="authorization",
        search_terms=("hasPrivilege",),
        explanation="Find the exact frozen authorization decision.",
        supporting_anchor_ids=("anchor-integration",),
    )
    refinement = EvidenceRefinementResult.from_content(
        parent_context_sha256=context.context_sha256,
        successor_context_sha256=successor.context_sha256,
        requested_need_sha256=canonical_sha256([need.model_dump(mode="json")]),
        priority_anchor_ids=(),
        added_anchor_ids=("anchor-fabricated",),
        round_number=1,
        exhausted=False,
        reason_code="frozen_context_extended",
    )

    with pytest.raises(MilestoneTwoTransitionError, match="not bound"):
        workflow._persist_evidence_refinement(
            prepared=prepared,
            needs=(need,),
            successor_context=successor,
            refinement=refinement,
            freshness=workflow._snapshot_acquirer.recheck(snapshot),
        )


def test_resume_reloads_a_terminal_record_from_the_refined_round(tmp_path) -> None:
    """Terminal recovery must locate envelopes saved after a successful refinement."""
    snapshot = _snapshot()
    context = _context(snapshot)
    successor = _successor_context(snapshot, context)
    diffs = (
        _diff("author_diff", snapshot.merge_base_sha, snapshot.head_sha, "a" * 64),
        _diff(
            "integration_diff",
            snapshot.base_sha,
            snapshot.candidate_sha,
            "b" * 64,
        ),
        _diff(
            "base_drift_diff",
            snapshot.merge_base_sha,
            snapshot.base_sha,
            "c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    dependencies = MilestoneTwoDependencies(
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=_NoRiskGateway(),
    )
    first = MilestoneTwoWorkflow(
        run_id="m2-refined-terminal-recovery-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )
    prepared = first.prepare_pr(snapshot.pull_url)
    need = FrozenEvidenceNeed(
        need_id="need-has-privilege",
        category="authorization",
        search_terms=("hasPrivilege",),
        explanation="Find the exact frozen authorization decision.",
        supporting_anchor_ids=("anchor-integration",),
    )
    refinement = EvidenceRefinementResult.from_content(
        parent_context_sha256=context.context_sha256,
        successor_context_sha256=successor.context_sha256,
        requested_need_sha256=canonical_sha256([need.model_dump(mode="json")]),
        priority_anchor_ids=(),
        added_anchor_ids=("anchor-hidden-authorization",),
        round_number=1,
        exhausted=False,
        reason_code="frozen_context_extended",
    )
    first._persist_evidence_refinement(
        prepared=prepared,
        needs=(need,),
        successor_context=successor,
        refinement=refinement,
        freshness=dependencies.snapshot_acquirer.recheck(snapshot),
    )
    refined = resume_milestone_two_workflow(
        run_handle=first.run_handle,
        dependencies=dependencies,
    )
    refined.propose_risks()
    sealed = refined.finish_without_risk()

    resumed = resume_milestone_two_workflow(
        run_handle=first.run_handle,
        dependencies=dependencies,
    )

    assert resumed.terminal_record == sealed
    assert resumed.context_refinements == (refinement,)


@pytest.mark.parametrize(
    ("artifact_round", "message"),
    [
        (0, "invalid round path"),
        (2, "missing parent round"),
        (4, "exceeds the configured round limit"),
    ],
)
def test_resume_rejects_a_non_contiguous_or_unbounded_refinement_chain(
    tmp_path,
    artifact_round: int,
    message: str,
) -> None:
    """Accepting a skipped or unbounded round would make replay non-causal."""
    snapshot = _snapshot()
    context = _context(snapshot)
    diffs = (
        _diff("author_diff", snapshot.merge_base_sha, snapshot.head_sha, "a" * 64),
        _diff(
            "integration_diff",
            snapshot.base_sha,
            snapshot.candidate_sha,
            "b" * 64,
        ),
        _diff(
            "base_drift_diff",
            snapshot.merge_base_sha,
            snapshot.base_sha,
            "c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    dependencies = MilestoneTwoDependencies(
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow = MilestoneTwoWorkflow(
        run_id=f"m2-invalid-refinement-round-{artifact_round}",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )
    workflow.prepare_pr(snapshot.pull_url)
    need = FrozenEvidenceNeed(
        need_id="need-has-privilege",
        category="authorization",
        search_terms=("hasPrivilege",),
        explanation="Find the exact frozen authorization decision.",
        supporting_anchor_ids=("anchor-integration",),
    )
    refinement = EvidenceRefinementResult.from_content(
        parent_context_sha256=context.context_sha256,
        successor_context_sha256=context.context_sha256,
        requested_need_sha256=canonical_sha256([need.model_dump(mode="json")]),
        priority_anchor_ids=("anchor-integration",),
        added_anchor_ids=(),
        round_number=max(1, artifact_round),
        exhausted=False,
        reason_code="catalog_evidence_prioritized",
    )
    workflow._persist_transition(
        artifact_name=(
            f"artifacts/workflow/evidence_refinements/{artifact_round}.json"
        ),
        event_type=f"workflow_frozen_evidence_refinement_{artifact_round}",
        payload={
            "refinement": refinement.model_dump(mode="json"),
            "needs": [need.model_dump(mode="json")],
            "successor_context": context.model_dump(mode="json"),
            "freshness": dependencies.snapshot_acquirer.recheck(snapshot).model_dump(
                mode="json"
            ),
        },
        input_hashes={
            "snapshot": snapshot.snapshot_key,
            "parent_context": context.context_sha256,
            "successor_context": context.context_sha256,
            "requested_need": refinement.requested_need_sha256,
            "refinement": refinement.refinement_sha256,
        },
        reason_code=refinement.reason_code,
    )

    with pytest.raises(MilestoneTwoTransitionError, match=message):
        resume_milestone_two_workflow(
            run_handle=workflow.run_handle,
            dependencies=dependencies,
        )


def test_resume_rejects_tampered_prepared_evidence(tmp_path) -> None:
    """Recovery must stop if saved evidence no longer matches its journal hash."""
    snapshot = _snapshot()
    context = _context(snapshot)
    diffs = (
        _diff("author_diff", snapshot.merge_base_sha, snapshot.head_sha, "a" * 64),
        _diff(
            "integration_diff",
            snapshot.base_sha,
            snapshot.candidate_sha,
            "b" * 64,
        ),
        _diff(
            "base_drift_diff",
            snapshot.merge_base_sha,
            snapshot.base_sha,
            "c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    dependencies = MilestoneTwoDependencies(
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-tampered-prepared-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )

    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/900000002")

    prepared_file = (
        recorder.locate_run(workflow.run_handle.run_id)
        / "artifacts"
        / "workflow"
        / "prepared.json"
    )
    prepared_file.write_bytes(b'{"tampered": true}\n')

    with pytest.raises(MilestoneTwoTransitionError, match="does not match its journal"):
        resume_milestone_two_workflow(
            run_handle=workflow.run_handle,
            dependencies=dependencies,
        )


def test_resume_rejects_a_tampered_risk_evidence_envelope(tmp_path) -> None:
    """Recovery cannot ground a response after its visibility boundary changes."""
    snapshot = _snapshot()
    context = _context(snapshot)
    diffs = (
        _diff("author_diff", snapshot.merge_base_sha, snapshot.head_sha, "a" * 64),
        _diff(
            "integration_diff",
            snapshot.base_sha,
            snapshot.candidate_sha,
            "b" * 64,
        ),
        _diff(
            "base_drift_diff",
            snapshot.merge_base_sha,
            snapshot.base_sha,
            "c" * 64,
        ),
    )
    recorder = ArtifactRecorder(tmp_path)
    dependencies = MilestoneTwoDependencies(
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        recorder=recorder,
        snapshot_acquirer=_SnapshotAcquirer(snapshot),
        diff_builder=_DiffBuilder(diffs),
        context_builder=_ContextBuilder(context),
        store=object(),
        gateway=ReplayGateway({}),
    )
    workflow = MilestoneTwoWorkflow(
        run_id="m2-tampered-risk-envelope-run",
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
    )
    workflow.prepare_pr("https://github.com/openmrs/openmrs-core/pull/900000002")

    with pytest.raises(ModelGatewayError):
        workflow.propose_risks()

    envelope_file = (
        recorder.locate_run(workflow.run_handle.run_id)
        / "artifacts"
        / "model_evidence"
        / "risk_hypothesis.json"
    )
    envelope_file.write_bytes(b'{"tampered": true}\n')

    with pytest.raises(MilestoneTwoTransitionError, match="does not match its journal"):
        resume_milestone_two_workflow(
            run_handle=workflow.run_handle,
            dependencies=dependencies,
        )
