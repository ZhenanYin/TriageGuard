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
    PullRequestSnapshot,
    SnapshotFreshness,
)
from triageguard.llm import ReplayGateway
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
