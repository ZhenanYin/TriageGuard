"""Tests for freezing one exact OpenMRS Core pull-request snapshot."""

from datetime import UTC, datetime

import pytest

from triageguard.analysis.snapshot import (
    SnapshotAcquirer,
    SnapshotAcquisitionError,
)
from triageguard.config import Settings
from triageguard.domain.pr_analysis import PullRequestSnapshot
from triageguard.domain.statuses import EnvironmentKind
from triageguard.sources.github import (
    GitHubPullMetadata,
    GitHubRepositoryMetadata,
    GitHubResponseProvenance,
)


def test_acquirer_freezes_one_exact_base_head_and_merge_candidate() -> None:
    """GitHub metadata and local Git objects become one reproducible PR snapshot."""
    merge_base_sha = "a" * 40
    base_sha = "b" * 40
    head_sha = "c" * 40
    candidate_sha = "d" * 40
    acquired_at = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    events: list[str] = []

    provenance = GitHubResponseProvenance(api_version="2026-03-10")
    repository = GitHubRepositoryMetadata(
        default_branch="main",
        response_provenance=provenance,
    )
    pull = GitHubPullMetadata(
        number=7312,
        html_url="https://github.com/openmrs/openmrs-core/pull/7312",
        state="open",
        base_branch="main",
        base_sha=merge_base_sha,
        head_sha=head_sha,
        mergeable=True,
        merge_commit_sha=candidate_sha,
        observed_at=acquired_at,
        response_provenance=provenance,
    )

    class FakeGitHub:
        def get_repository(self) -> GitHubRepositoryMetadata:
            events.append("repository")
            return repository

        def get_pull(self, number: int) -> GitHubPullMetadata:
            assert number == 7312
            events.append("pull:7312")
            return pull

    class FakeStore:
        def fetch_snapshot(self, base_branch: str, pull_number: int) -> None:
            events.append(f"fetch:{base_branch}:{pull_number}")

        def resolve_commit(self, ref: str) -> str:
            events.append(f"resolve:{ref}")
            return {
                "refs/triageguard/base": base_sha,
                "refs/triageguard/head": head_sha,
                "refs/triageguard/candidate": candidate_sha,
            }[ref]

        def commit_parents(self, commit_sha: str) -> tuple[str, ...]:
            assert commit_sha == candidate_sha
            events.append("candidate-parents")
            return (base_sha, head_sha)

        def merge_base(self, observed_base_sha: str, observed_head_sha: str) -> str:
            assert (observed_base_sha, observed_head_sha) == (base_sha, head_sha)
            events.append("merge-base")
            return merge_base_sha

        def tree_sha(self, commit_sha: str) -> str:
            events.append(f"tree:{commit_sha}")
            return {
                merge_base_sha: "e" * 40,
                base_sha: "f" * 40,
                head_sha: "1" * 40,
                candidate_sha: "2" * 40,
            }[commit_sha]

        def git_version(self) -> str:
            events.append("git-version")
            return "2.47.1"

    snapshot = SnapshotAcquirer(
        github=FakeGitHub(),
        store=FakeStore(),
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        clock=lambda: acquired_at,
    ).acquire("https://github.com/openmrs/openmrs-core/pull/7312")

    assert (
        snapshot.merge_base_sha,
        snapshot.base_sha,
        snapshot.head_sha,
        snapshot.candidate_sha,
    ) == (
        merge_base_sha,
        base_sha,
        head_sha,
        candidate_sha,
    )
    assert snapshot.default_branch == "main"
    assert snapshot.base_branch == "main"
    assert snapshot.git_version == "2.47.1"
    assert snapshot.acquired_at == acquired_at
    assert snapshot.acquisition_tool_version == "triageguard/2.0.0"
    assert len(snapshot.analysis_config_sha256) == 64
    assert events == [
        "repository",
        "pull:7312",
        "fetch:main:7312",
        "resolve:refs/triageguard/base",
        "resolve:refs/triageguard/head",
        "resolve:refs/triageguard/candidate",
        "candidate-parents",
        "merge-base",
        f"tree:{merge_base_sha}",
        f"tree:{base_sha}",
        f"tree:{head_sha}",
        f"tree:{candidate_sha}",
        "git-version",
        "pull:7312",
    ]


def test_acquirer_rejects_a_candidate_with_reversed_parents() -> None:
    """A candidate must be GitHub's base-parent followed by its head-parent."""
    base_sha = "b" * 40
    head_sha = "c" * 40
    candidate_sha = "d" * 40
    observed_at = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    provenance = GitHubResponseProvenance(api_version="2026-03-10")

    class FakeGitHub:
        def get_repository(self) -> GitHubRepositoryMetadata:
            return GitHubRepositoryMetadata(
                default_branch="main",
                response_provenance=provenance,
            )

        def get_pull(self, number: int) -> GitHubPullMetadata:
            return GitHubPullMetadata(
                number=number,
                html_url=f"https://github.com/openmrs/openmrs-core/pull/{number}",
                state="open",
                base_branch="main",
                base_sha=base_sha,
                head_sha=head_sha,
                mergeable=True,
                merge_commit_sha=candidate_sha,
                observed_at=observed_at,
                response_provenance=provenance,
            )

    class FakeStore:
        def fetch_snapshot(self, base_branch: str, pull_number: int) -> None:
            assert (base_branch, pull_number) == ("main", 7312)

        def resolve_commit(self, ref: str) -> str:
            return {
                "refs/triageguard/base": base_sha,
                "refs/triageguard/head": head_sha,
                "refs/triageguard/candidate": candidate_sha,
            }[ref]

        def commit_parents(self, commit_sha: str) -> tuple[str, ...]:
            assert commit_sha == candidate_sha
            return (head_sha, base_sha)

    acquirer = SnapshotAcquirer(
        github=FakeGitHub(),
        store=FakeStore(),
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        clock=lambda: observed_at,
    )

    with pytest.raises(SnapshotAcquisitionError, match="candidate_parent_mismatch"):
        acquirer.acquire("https://github.com/openmrs/openmrs-core/pull/7312")


def test_acquirer_uses_the_fetched_candidate_when_github_omits_its_sha() -> None:
    """A mergeable PR remains analyzable when GitHub omits a transient SHA."""
    merge_base_sha = "a" * 40
    base_sha = "b" * 40
    head_sha = "c" * 40
    candidate_sha = "d" * 40
    observed_at = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    provenance = GitHubResponseProvenance(api_version="2026-03-10")

    class FakeGitHub:
        def get_repository(self) -> GitHubRepositoryMetadata:
            return GitHubRepositoryMetadata(
                default_branch="main",
                response_provenance=provenance,
            )

        def get_pull(self, number: int) -> GitHubPullMetadata:
            return GitHubPullMetadata(
                number=number,
                html_url=f"https://github.com/openmrs/openmrs-core/pull/{number}",
                state="open",
                base_branch="main",
                base_sha=base_sha,
                head_sha=head_sha,
                mergeable=True,
                merge_commit_sha=None,
                observed_at=observed_at,
                response_provenance=provenance,
            )

    class FakeStore:
        def fetch_snapshot(self, base_branch: str, pull_number: int) -> None:
            assert (base_branch, pull_number) == ("main", 7312)

        def resolve_commit(self, ref: str) -> str:
            return {
                "refs/triageguard/base": base_sha,
                "refs/triageguard/head": head_sha,
                "refs/triageguard/candidate": candidate_sha,
            }[ref]

        def commit_parents(self, commit_sha: str) -> tuple[str, ...]:
            assert commit_sha == candidate_sha
            return (base_sha, head_sha)

        def merge_base(self, observed_base_sha: str, observed_head_sha: str) -> str:
            assert (observed_base_sha, observed_head_sha) == (base_sha, head_sha)
            return merge_base_sha

        def tree_sha(self, commit_sha: str) -> str:
            return {
                merge_base_sha: "e" * 40,
                base_sha: "f" * 40,
                head_sha: "1" * 40,
                candidate_sha: "2" * 40,
            }[commit_sha]

        def git_version(self) -> str:
            return "2.47.1"

    snapshot = SnapshotAcquirer(
        github=FakeGitHub(),
        store=FakeStore(),
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        clock=lambda: observed_at,
    ).acquire("https://github.com/openmrs/openmrs-core/pull/7312")

    assert snapshot.candidate_sha == candidate_sha


def _frozen_snapshot() -> PullRequestSnapshot:
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=7312,
        pull_url="https://github.com/openmrs/openmrs-core/pull/7312",
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
        acquired_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="3" * 64,
    )


def test_recheck_marks_an_old_snapshot_stale_when_candidate_changes() -> None:
    """Approval must stop when Git's merge preview no longer matches."""
    snapshot = _frozen_snapshot()
    checked_at = datetime(2026, 8, 11, 13, 0, tzinfo=UTC)
    provenance = GitHubResponseProvenance(api_version="2026-03-10")

    class FakeGitHub:
        def get_repository(self) -> GitHubRepositoryMetadata:
            return GitHubRepositoryMetadata(
                default_branch="main",
                response_provenance=provenance,
            )

        def get_pull(self, number: int) -> GitHubPullMetadata:
            assert number == snapshot.pull_number
            return GitHubPullMetadata(
                number=number,
                html_url=snapshot.pull_url,
                state="open",
                base_branch="main",
                base_sha=snapshot.base_sha,
                head_sha=snapshot.head_sha,
                mergeable=True,
                merge_commit_sha=None,
                observed_at=checked_at,
                response_provenance=provenance,
            )

    class FakeStore:
        def remote_snapshot_refs(
            self,
            base_branch: str,
            pull_number: int,
        ) -> tuple[str, str]:
            assert (base_branch, pull_number) == (
                snapshot.base_branch,
                snapshot.pull_number,
            )
            return (snapshot.base_sha, "0" * 40)

    freshness = SnapshotAcquirer(
        github=FakeGitHub(),
        store=FakeStore(),
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        clock=lambda: checked_at,
    ).recheck(snapshot)

    assert freshness.status == "stale"
    assert freshness.reason_code == "snapshot_changed"
    assert freshness.checked_at == checked_at
    assert freshness.observed_base_sha == snapshot.base_sha
    assert freshness.observed_head_sha == snapshot.head_sha
    assert freshness.observed_candidate_sha == "0" * 40


def test_recheck_uses_remote_refs_when_github_base_metadata_is_stale() -> None:
    """Final approval trusts the current base and preview refs Git exposes."""
    snapshot = _frozen_snapshot()
    checked_at = datetime(2026, 8, 11, 13, 0, tzinfo=UTC)
    provenance = GitHubResponseProvenance(api_version="2026-03-10")

    class FakeGitHub:
        def get_repository(self) -> GitHubRepositoryMetadata:
            return GitHubRepositoryMetadata(
                default_branch="main",
                response_provenance=provenance,
            )

        def get_pull(self, number: int) -> GitHubPullMetadata:
            assert number == snapshot.pull_number
            return GitHubPullMetadata(
                number=number,
                html_url=snapshot.pull_url,
                state="open",
                base_branch="main",
                base_sha="0" * 40,
                head_sha=snapshot.head_sha,
                mergeable=True,
                merge_commit_sha=None,
                observed_at=checked_at,
                response_provenance=provenance,
            )

    class FakeStore:
        def remote_snapshot_refs(
            self,
            base_branch: str,
            pull_number: int,
        ) -> tuple[str, str]:
            assert (base_branch, pull_number) == (
                snapshot.base_branch,
                snapshot.pull_number,
            )
            return (snapshot.base_sha, snapshot.candidate_sha)

    freshness = SnapshotAcquirer(
        github=FakeGitHub(),
        store=FakeStore(),
        settings=Settings(environment_kind=EnvironmentKind.REAL_PR_ANALYSIS),
        clock=lambda: checked_at,
    ).recheck(snapshot)

    assert freshness.status == "current"
    assert freshness.reason_code == "snapshot_current"
    assert freshness.observed_base_sha == snapshot.base_sha
    assert freshness.observed_candidate_sha == snapshot.candidate_sha


def test_analysis_configuration_hash_changes_with_diff_limits() -> None:
    """Different raw-diff bounds must create different research configurations."""
    common_time = datetime(2026, 8, 12, tzinfo=UTC)

    first = SnapshotAcquirer(
        github=object(),
        store=object(),
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
            max_diff_files=1_000,
            max_diff_bytes=25_000_000,
        ),
        clock=lambda: common_time,
    )
    second = SnapshotAcquirer(
        github=object(),
        store=object(),
        settings=Settings(
            environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
            max_diff_files=1_001,
            max_diff_bytes=25_000_001,
        ),
        clock=lambda: common_time,
    )

    assert first._analysis_config_sha256() != second._analysis_config_sha256()
