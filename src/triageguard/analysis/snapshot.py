"""Freeze and recheck exact OpenMRS Core pull-request identities."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from triageguard.config import Settings
from triageguard.domain.pr_analysis import PullRequestSnapshot, SnapshotFreshness
from triageguard.provenance import canonical_sha256
from triageguard.sources.git import GitCommandError
from triageguard.sources.github import (
    GitHubPullMetadata,
    GitHubReadError,
    GitHubRepositoryMetadata,
    parse_openmrs_pr_url,
)

_ACQUISITION_TOOL_VERSION = "triageguard/2.0.0"


class SnapshotAcquisitionError(RuntimeError):
    """A safe, typed failure while freezing one pull-request snapshot."""

    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(f"{reason_code}: {safe_message}")
        self.reason_code = reason_code
        self.safe_message = safe_message


class SnapshotAcquirer:
    """Combine read-only GitHub and local-Git facts into a frozen PR snapshot."""

    def __init__(
        self,
        *,
        github: object,
        store: object,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._github = github
        self._store = store
        self._settings = settings
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._sleep = sleep

    def acquire(self, pr_url: str) -> PullRequestSnapshot:
        """Freeze exactly one currently open, mergeable OpenMRS Core pull request."""
        try:
            pull_number = parse_openmrs_pr_url(pr_url)
        except (TypeError, ValueError) as error:
            raise SnapshotAcquisitionError(
                "unsupported_pr_url",
                "Only a canonical OpenMRS Core pull-request URL can be analyzed.",
            ) from error

        try:
            repository = self._github.get_repository()
            first = self._bounded_mergeability_read(pull_number)
            self._validate_supported_pull(repository, first)

            self._store.fetch_snapshot(first.base_branch, first.number)
            snapshot = self._validate_git_relationships(repository, first)

            second = self._github.get_pull(first.number)
            self._require_same_remote_identity(
                first,
                second,
                repository.default_branch,
            )
        except GitHubReadError as error:
            raise SnapshotAcquisitionError(
                error.reason_code,
                error.safe_message,
            ) from error
        except GitCommandError as error:
            raise SnapshotAcquisitionError(
                error.reason_code,
                error.safe_message,
            ) from error

        return snapshot

    def recheck(self, snapshot: PullRequestSnapshot) -> SnapshotFreshness:
        """Report whether a previously frozen snapshot still matches GitHub."""
        checked_at = self._clock()

        try:
            repository = self._github.get_repository()
            pull = self._bounded_mergeability_read(snapshot.pull_number)
        except (GitHubReadError, SnapshotAcquisitionError):
            return SnapshotFreshness(
                snapshot_key=snapshot.snapshot_key,
                status="unknown",
                reason_code="github_recheck_unavailable",
                checked_at=checked_at,
                observed_base_sha=None,
                observed_head_sha=None,
                observed_candidate_sha=None,
            )

        candidate_sha = pull.merge_commit_sha
        if candidate_sha is None:
            return SnapshotFreshness(
                snapshot_key=snapshot.snapshot_key,
                status="unknown",
                reason_code="github_recheck_unavailable",
                checked_at=checked_at,
                observed_base_sha=None,
                observed_head_sha=None,
                observed_candidate_sha=None,
            )

        is_current = (
            repository.default_branch == snapshot.default_branch
            and pull.state == "open"
            and pull.base_branch == snapshot.base_branch
            and pull.base_sha == snapshot.base_sha
            and pull.head_sha == snapshot.head_sha
            and pull.mergeable is True
            and candidate_sha == snapshot.candidate_sha
        )

        return SnapshotFreshness(
            snapshot_key=snapshot.snapshot_key,
            status="current" if is_current else "stale",
            reason_code="snapshot_current" if is_current else "snapshot_changed",
            checked_at=checked_at,
            observed_base_sha=pull.base_sha,
            observed_head_sha=pull.head_sha,
            observed_candidate_sha=candidate_sha,
        )

    def _bounded_mergeability_read(self, pull_number: int) -> GitHubPullMetadata:
        """Read GitHub mergeability at most three times with fixed delays."""
        pull = self._github.get_pull(pull_number)
        if pull.mergeable is not None:
            return pull

        self._sleep(0.25)
        pull = self._github.get_pull(pull_number)
        if pull.mergeable is not None:
            return pull

        self._sleep(0.50)
        pull = self._github.get_pull(pull_number)
        if pull.mergeable is None:
            raise SnapshotAcquisitionError(
                "mergeability_unknown",
                "GitHub did not determine pull-request mergeability in time.",
            )

        return pull

    @staticmethod
    def _validate_supported_pull(
        repository: GitHubRepositoryMetadata,
        pull: GitHubPullMetadata,
    ) -> None:
        """Reject PR states that this research prototype cannot safely analyze."""
        try:
            returned_number = parse_openmrs_pr_url(pull.html_url)
        except (TypeError, ValueError) as error:
            raise SnapshotAcquisitionError(
                "unsupported_pr_url",
                "GitHub returned an unsupported pull-request URL.",
            ) from error

        if returned_number != pull.number:
            raise SnapshotAcquisitionError(
                "snapshot_changed_during_acquisition",
                "GitHub pull-request identity was inconsistent.",
            )
        if pull.state != "open":
            raise SnapshotAcquisitionError(
                "pr_not_open",
                "Only open pull requests can be analyzed.",
            )
        if pull.base_branch != repository.default_branch:
            raise SnapshotAcquisitionError(
                "non_default_base_branch",
                "The pull request does not target the repository default branch.",
            )
        if pull.mergeable is None:
            raise SnapshotAcquisitionError(
                "mergeability_unknown",
                "GitHub has not determined pull-request mergeability.",
            )
        if pull.mergeable is False:
            raise SnapshotAcquisitionError(
                "merge_conflict",
                "GitHub reported that the pull request cannot be merged.",
            )
        if pull.merge_commit_sha is None:
            raise SnapshotAcquisitionError(
                "candidate_ref_missing",
                "GitHub did not provide a merge candidate for the pull request.",
            )

    def _validate_git_relationships(
        self,
        repository: GitHubRepositoryMetadata,
        pull: GitHubPullMetadata,
    ) -> PullRequestSnapshot:
        """Verify GitHub metadata against the exact locally fetched object graph."""
        base_sha = self._store.resolve_commit("refs/triageguard/base")
        head_sha = self._store.resolve_commit("refs/triageguard/head")
        candidate_sha = self._store.resolve_commit("refs/triageguard/candidate")

        if (
            base_sha != pull.base_sha
            or head_sha != pull.head_sha
            or candidate_sha != pull.merge_commit_sha
        ):
            raise SnapshotAcquisitionError(
                "snapshot_changed_during_acquisition",
                "Fetched Git objects did not match the first GitHub observation.",
            )

        if self._store.commit_parents(candidate_sha) != (base_sha, head_sha):
            raise SnapshotAcquisitionError(
                "candidate_parent_mismatch",
                "The merge candidate did not have base then head as parents.",
            )

        merge_base_sha = self._store.merge_base(base_sha, head_sha)
        return PullRequestSnapshot.from_identity(
            repository="openmrs/openmrs-core",
            pull_number=pull.number,
            pull_url=pull.html_url,
            state="open",
            default_branch=repository.default_branch,
            base_branch=pull.base_branch,
            merge_base_sha=merge_base_sha,
            base_sha=base_sha,
            head_sha=head_sha,
            candidate_sha=candidate_sha,
            merge_base_tree_sha=self._store.tree_sha(merge_base_sha),
            base_tree_sha=self._store.tree_sha(base_sha),
            head_tree_sha=self._store.tree_sha(head_sha),
            candidate_tree_sha=self._store.tree_sha(candidate_sha),
            acquired_at=self._clock(),
            github_api_version=self._settings.github_api_version,
            git_version=self._store.git_version(),
            acquisition_tool_version=_ACQUISITION_TOOL_VERSION,
            analysis_config_sha256=self._analysis_config_sha256(),
        )

    def _analysis_config_sha256(self) -> str:
        """Hash only public, analysis-relevant configuration."""
        return canonical_sha256(
            {
                "github_api_version": self._settings.github_api_version,
                "max_context_files": self._settings.max_context_files,
                "max_context_anchors": self._settings.max_context_anchors,
                "max_context_bytes": self._settings.max_context_bytes,
                "max_context_anchor_lines": self._settings.max_context_anchor_lines,
                "max_context_blob_bytes": self._settings.max_context_blob_bytes,
                "max_context_search_identifiers": (
                    self._settings.max_context_search_identifiers
                ),
                "max_context_hits_per_identifier": (
                    self._settings.max_context_hits_per_identifier
                ),
            }
        )

    @staticmethod
    def _require_same_remote_identity(
        first: GitHubPullMetadata,
        second: GitHubPullMetadata,
        default_branch: str,
    ) -> None:
        """Reject an acquisition if any remote PR identity changed mid-read."""
        if (
            second.number != first.number
            or second.html_url != first.html_url
            or second.state != "open"
            or second.base_branch != default_branch
            or second.base_branch != first.base_branch
            or second.base_sha != first.base_sha
            or second.head_sha != first.head_sha
            or second.mergeable is not True
            or second.merge_commit_sha != first.merge_commit_sha
        ):
            raise SnapshotAcquisitionError(
                "snapshot_changed_during_acquisition",
                "The pull request changed while its snapshot was being acquired.",
            )
