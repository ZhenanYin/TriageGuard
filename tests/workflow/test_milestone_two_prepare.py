"""Tests for the preparation stage of Milestone 2 PR analysis."""

from datetime import UTC, datetime

import pytest

from triageguard.analysis.context import ContextLimits
from triageguard.analysis.diffs import parse_patch
from triageguard.config import Settings
from triageguard.domain import (
    ContextBundle,
    EnvironmentKind,
    PullRequestSnapshot,
)
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.workflow.milestone_two import (
    MilestoneTwoTransitionError,
    MilestoneTwoWorkflow,
    PreparedPullRequest,
)

SUPPORTED_PR_URL = "https://github.com/openmrs/openmrs-core/pull/7312"


def _snapshot() -> PullRequestSnapshot:
    """Return one frozen four-revision OpenMRS Core snapshot."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=7312,
        pull_url=SUPPORTED_PR_URL,
        state="open",
        default_branch="main",
        base_branch="main",
        merge_base_sha="a" * 40,
        base_sha="b" * 40,
        head_sha="c" * 40,
        candidate_sha="d" * 40,
        merge_base_tree_sha="e" * 40,
        base_tree_sha="f" * 40,
        head_tree_sha="1" * 40,
        candidate_tree_sha="2" * 40,
        acquired_at=datetime(2026, 8, 15, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="git version 2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="3" * 64,
    )


def _diffs(snapshot: PullRequestSnapshot):
    """Return the exact three empty comparisons for this focused workflow test."""
    return (
        parse_patch(
            kind="author_diff",
            old_sha=snapshot.merge_base_sha,
            new_sha=snapshot.head_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version="git version 2.47.1",
        ),
        parse_patch(
            kind="integration_diff",
            old_sha=snapshot.base_sha,
            new_sha=snapshot.candidate_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version="git version 2.47.1",
        ),
        parse_patch(
            kind="base_drift_diff",
            old_sha=snapshot.merge_base_sha,
            new_sha=snapshot.base_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version="git version 2.47.1",
        ),
    )


def _context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Return a bounded context inventory supplied by the context builder fake."""
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(),
        selected_file_count=0,
        selected_anchor_count=0,
        selected_bytes=0,
        max_files=40,
        max_anchors=80,
        max_bytes=160_000,
        max_anchor_lines=120,
        max_blob_bytes=1_000_000,
        max_search_identifiers=100,
        max_hits_per_identifier=20,
        primary_change_represented=True,
    )


class _SnapshotAcquirer:
    """Return one known frozen snapshot and record the requested URL."""

    def __init__(self, snapshot: PullRequestSnapshot) -> None:
        self.snapshot = snapshot
        self.urls: list[str] = []

    def acquire(self, pr_url: str) -> PullRequestSnapshot:
        self.urls.append(pr_url)
        return self.snapshot


class _DiffBuilder:
    """Return one known diff inventory and record its snapshot input."""

    def __init__(self, diffs: tuple[object, object, object]) -> None:
        self.diffs = diffs
        self.snapshots: list[PullRequestSnapshot] = []

    def build_all(
        self,
        snapshot: PullRequestSnapshot,
    ) -> tuple[object, object, object]:
        self.snapshots.append(snapshot)
        return self.diffs


class _ContextBuilder:
    """Return one known context and record all workflow inputs."""

    def __init__(self, context: ContextBundle) -> None:
        self.context = context
        self.calls: list[tuple[object, object, object, ContextLimits]] = []

    def build(
        self,
        *,
        snapshot: object,
        diffs: object,
        store: object,
        limits: ContextLimits,
    ) -> ContextBundle:
        self.calls.append((snapshot, diffs, store, limits))
        return self.context


def test_prepare_pr_freezes_snapshot_diffs_and_context(tmp_path) -> None:
    """Preparation connects the three existing evidence stages in one order."""
    settings = Settings(
        environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
    )
    snapshot = _snapshot()
    diffs = _diffs(snapshot)
    context = _context(snapshot)
    acquirer = _SnapshotAcquirer(snapshot)
    diff_builder = _DiffBuilder(diffs)
    context_builder = _ContextBuilder(context)
    store = object()

    workflow = MilestoneTwoWorkflow(
        run_id="m2-run-1",
        settings=settings,
        recorder=ArtifactRecorder(tmp_path),
        snapshot_acquirer=acquirer,
        diff_builder=diff_builder,
        context_builder=context_builder,
        store=store,
        gateway=ReplayGateway({}),
    )

    prepared = workflow.prepare_pr(SUPPORTED_PR_URL)

    assert isinstance(prepared, PreparedPullRequest)
    assert prepared.snapshot == snapshot
    assert prepared.diffs == diffs
    assert prepared.context == context
    assert acquirer.urls == [SUPPORTED_PR_URL]
    assert diff_builder.snapshots == [snapshot]
    assert context_builder.calls == [
        (snapshot, diffs, store, ContextLimits.from_settings(settings))
    ]

    with pytest.raises(MilestoneTwoTransitionError, match="already"):
        workflow.prepare_pr(SUPPORTED_PR_URL)
