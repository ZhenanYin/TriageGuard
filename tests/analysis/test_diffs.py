"""Tests for reproducible OpenMRS Core diff artifacts."""

import hashlib
from datetime import UTC, datetime

import pytest

from triageguard.analysis import DiffBuilder as PublicDiffBuilder
from triageguard.analysis import DiffBuildError as PublicDiffBuildError
from triageguard.analysis import parse_patch as public_parse_patch
from triageguard.analysis.diffs import DiffBuilder, DiffBuildError, parse_patch
from triageguard.domain.pr_analysis import PullRequestSnapshot
from triageguard.provenance import canonical_sha256
from triageguard.sources.git import GitCommandError


def test_parse_patch_records_one_modified_file_and_its_hunk() -> None:
    """A normal Git patch becomes a structured, hash-bound diff artifact."""
    old_sha = "b" * 40
    new_sha = "c" * 40
    patch_bytes = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -10 +10 @@\n"
        b"-return oldValue;\n"
        b"+return newValue;\n"
    )
    numstat_bytes = b"1\t1\tapi/PatientService.java\0"

    artifact = parse_patch(
        kind="integration_diff",
        old_sha=old_sha,
        new_sha=new_sha,
        patch_bytes=patch_bytes,
        numstat_bytes=numstat_bytes,
        git_version="2.47.1",
    )

    assert artifact.kind == "integration_diff"
    assert artifact.old_revision == old_sha
    assert artifact.new_revision == new_sha
    assert artifact.patch_sha256 == hashlib.sha256(patch_bytes).hexdigest()
    assert len(artifact.artifact_sha256) == 64
    assert len(artifact.files) == 1

    changed_file = artifact.files[0]
    assert changed_file.status == "modified"
    assert changed_file.old_path == "api/PatientService.java"
    assert changed_file.new_path == "api/PatientService.java"
    assert changed_file.binary is False
    assert (changed_file.additions, changed_file.deletions) == (1, 1)
    assert [
        (
            hunk.old_start,
            hunk.old_count,
            hunk.new_start,
            hunk.new_count,
        )
        for hunk in changed_file.hunks
    ] == [(10, 1, 10, 1)]


def test_binary_rename_is_explicitly_recorded() -> None:
    """A renamed binary file must never be treated as ordinary text source."""
    artifact = parse_patch(
        kind="integration_diff",
        old_sha="b" * 40,
        new_sha="c" * 40,
        patch_bytes=(
            b"diff --git a/old/logo.bin b/new/logo.bin\n"
            b"similarity index 50%\n"
            b"rename from old/logo.bin\n"
            b"rename to new/logo.bin\n"
            b"Binary files a/old/logo.bin and b/new/logo.bin differ\n"
        ),
        numstat_bytes=b"-\t-\t\0old/logo.bin\0new/logo.bin\0",
        git_version="2.47.1",
    )

    changed_file = artifact.files[0]
    assert changed_file.status == "renamed"
    assert changed_file.old_path == "old/logo.bin"
    assert changed_file.new_path == "new/logo.bin"
    assert changed_file.binary is True
    assert (changed_file.additions, changed_file.deletions) == (0, 0)
    assert changed_file.hunks == ()


def _snapshot() -> PullRequestSnapshot:
    """Create one fixed snapshot with four distinct revisions."""
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
        acquired_at=datetime(2026, 8, 12, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="3" * 64,
    )


def test_builder_uses_the_three_approved_frozen_comparisons() -> None:
    """The artifact order is author, primary integration, then base drift."""
    snapshot = _snapshot()
    patch_bytes = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -10 +10 @@\n"
        b"-return oldValue;\n"
        b"+return newValue;\n"
    )
    numstat_bytes = b"1\t1\tapi/PatientService.java\0"

    class RecordingStore:
        def __init__(self) -> None:
            self.diff_calls: list[tuple[str, str]] = []

        def diff(self, old_sha: str, new_sha: str) -> tuple[bytes, bytes]:
            self.diff_calls.append((old_sha, new_sha))
            return patch_bytes, numstat_bytes

        def git_version(self) -> str:
            return "2.47.1"

    store = RecordingStore()
    artifacts = DiffBuilder(store).build_all(snapshot)

    assert [
        (artifact.kind, artifact.old_revision, artifact.new_revision)
        for artifact in artifacts
    ] == [
        ("author_diff", snapshot.merge_base_sha, snapshot.head_sha),
        ("integration_diff", snapshot.base_sha, snapshot.candidate_sha),
        ("base_drift_diff", snapshot.merge_base_sha, snapshot.base_sha),
    ]
    assert store.diff_calls == [
        (snapshot.merge_base_sha, snapshot.head_sha),
        (snapshot.base_sha, snapshot.candidate_sha),
        (snapshot.merge_base_sha, snapshot.base_sha),
    ]


def test_parse_patch_rejects_more_files_than_the_approved_limit() -> None:
    """A partial file list is not valid security evidence."""
    patch_bytes = (
        b"diff --git a/api/A.java b/api/A.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/A.java\n"
        b"+++ b/api/A.java\n"
        b"@@ -1 +1 @@\n"
        b"-oldA\n"
        b"+newA\n"
        b"diff --git a/api/B.java b/api/B.java\n"
        b"index 3333333..4444444 100644\n"
        b"--- a/api/B.java\n"
        b"+++ b/api/B.java\n"
        b"@@ -1 +1 @@\n"
        b"-oldB\n"
        b"+newB\n"
    )
    numstat_bytes = b"1\t1\tapi/A.java\x001\t1\tapi/B.java\x00"

    with pytest.raises(DiffBuildError, match="analysis_limit_exceeded") as error:
        parse_patch(
            kind="integration_diff",
            old_sha="b" * 40,
            new_sha="c" * 40,
            patch_bytes=patch_bytes,
            numstat_bytes=numstat_bytes,
            git_version="2.47.1",
            max_files=1,
            max_bytes=1_000_000,
        )

    assert error.value.reason_code == "analysis_limit_exceeded"


def test_parse_patch_rejects_raw_input_larger_than_the_approved_limit() -> None:
    """Oversize raw Git output is rejected before it can be partially analyzed."""
    patch_bytes = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -10 +10 @@\n"
        b"-return oldValue;\n"
        b"+return newValue;\n"
    )
    numstat_bytes = b"1\t1\tapi/PatientService.java\0"

    with pytest.raises(DiffBuildError, match="analysis_limit_exceeded") as error:
        parse_patch(
            kind="integration_diff",
            old_sha="b" * 40,
            new_sha="c" * 40,
            patch_bytes=patch_bytes,
            numstat_bytes=numstat_bytes,
            git_version="2.47.1",
            max_files=1_000,
            max_bytes=len(patch_bytes) + len(numstat_bytes) - 1,
        )

    assert error.value.reason_code == "analysis_limit_exceeded"


def test_parse_patch_matches_quoted_unicode_path_to_its_literal_manifest_path() -> None:
    """Git's quoted patch header must resolve to the same literal UTF-8 file."""
    artifact = parse_patch(
        kind="integration_diff",
        old_sha="b" * 40,
        new_sha="c" * 40,
        patch_bytes=(
            b'diff --git "a/api/Caf\\303\\251.java" "b/api/Caf\\303\\251.java"\n'
            b"index 1111111..2222222 100644\n"
            b'--- "a/api/Caf\\303\\251.java"\n'
            b'+++ "b/api/Caf\\303\\251.java"\n'
            b"@@ -1 +1 @@\n"
            b"-old\n"
            b"+new\n"
        ),
        numstat_bytes=b"1\t1\tapi/Caf\xc3\xa9.java\0",
        git_version="2.47.1",
    )

    changed_file = artifact.files[0]
    assert changed_file.old_path == "api/Café.java"
    assert changed_file.new_path == "api/Café.java"
    assert changed_file.status == "modified"


def test_parse_patch_records_added_deleted_files_and_zero_count_hunks_in_order() -> (
    None
):
    """File status, zero-count ranges, and Git's recorded order stay explicit."""
    artifact = parse_patch(
        kind="integration_diff",
        old_sha="b" * 40,
        new_sha="c" * 40,
        patch_bytes=(
            b"diff --git a/api/New.java b/api/New.java\n"
            b"new file mode 100644\n"
            b"index 0000000..1111111\n"
            b"--- /dev/null\n"
            b"+++ b/api/New.java\n"
            b"@@ -0,0 +1,2 @@\n"
            b"+first line\n"
            b"+second line\n"
            b"diff --git a/api/Old.java b/api/Old.java\n"
            b"deleted file mode 100644\n"
            b"index 2222222..0000000\n"
            b"--- a/api/Old.java\n"
            b"+++ /dev/null\n"
            b"@@ -1,2 +0,0 @@\n"
            b"-first line\n"
            b"-second line\n"
        ),
        numstat_bytes=b"2\t0\tapi/New.java\x000\t2\tapi/Old.java\x00",
        git_version="2.47.1",
    )

    assert [(item.status, item.old_path, item.new_path) for item in artifact.files] == [
        ("added", None, "api/New.java"),
        ("deleted", "api/Old.java", None),
    ]
    assert [
        (
            item.hunks[0].old_start,
            item.hunks[0].old_count,
            item.hunks[0].new_start,
            item.hunks[0].new_count,
        )
        for item in artifact.files
    ] == [
        (0, 0, 1, 2),
        (1, 2, 0, 0),
    ]


def test_parse_patch_rejects_a_malformed_hunk_header() -> None:
    """Unparseable changed-line ranges cannot become incomplete evidence."""
    with pytest.raises(DiffBuildError, match="diff_parse_failed") as error:
        parse_patch(
            kind="integration_diff",
            old_sha="b" * 40,
            new_sha="c" * 40,
            patch_bytes=(
                b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
                b"index 1111111..2222222 100644\n"
                b"--- a/api/PatientService.java\n"
                b"+++ b/api/PatientService.java\n"
                b"@@ -not-a-range +10 @@\n"
                b"-return oldValue;\n"
                b"+return newValue;\n"
            ),
            numstat_bytes=b"1\t1\tapi/PatientService.java\0",
            git_version="2.47.1",
        )

    assert error.value.reason_code == "diff_parse_failed"


def test_parse_patch_rejects_an_unsupported_combined_diff() -> None:
    """Merge-style combined patches are not an approved two-revision artifact."""
    with pytest.raises(DiffBuildError, match="unsupported_combined_diff") as error:
        parse_patch(
            kind="integration_diff",
            old_sha="b" * 40,
            new_sha="c" * 40,
            patch_bytes=(
                b"diff --cc api/PatientService.java\n"
                b"index 1111111,2222222..3333333\n"
                b"--- a/api/PatientService.java\n"
                b"+++ b/api/PatientService.java\n"
                b"@@@ -10,10 -10,10 +10,10 @@@\n"
            ),
            numstat_bytes=b"1\t1\tapi/PatientService.java\0",
            git_version="2.47.1",
        )

    assert error.value.reason_code == "unsupported_combined_diff"


def test_builder_reports_a_missing_primary_integration_artifact() -> None:
    """The workflow must stop if its main B-to-C evidence cannot be built."""
    snapshot = _snapshot()
    patch_bytes = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -10 +10 @@\n"
        b"-return oldValue;\n"
        b"+return newValue;\n"
    )
    numstat_bytes = b"1\t1\tapi/PatientService.java\0"

    class StoreWithMissingIntegration:
        def diff(self, old_sha: str, new_sha: str) -> tuple[bytes, bytes]:
            if (old_sha, new_sha) == (
                snapshot.base_sha,
                snapshot.candidate_sha,
            ):
                raise GitCommandError(
                    "git_command_failed",
                    "Git command failed.",
                )
            return patch_bytes, numstat_bytes

        def git_version(self) -> str:
            return "2.47.1"

    with pytest.raises(
        DiffBuildError,
        match="primary_integration_artifact_missing",
    ) as error:
        DiffBuilder(StoreWithMissingIntegration()).build_all(snapshot)

    assert error.value.reason_code == "primary_integration_artifact_missing"


def test_builder_enforces_its_injected_file_limit_for_every_comparison() -> None:
    """The configured file limit applies before any of the three diffs is accepted."""
    snapshot = _snapshot()
    patch_bytes = (
        b"diff --git a/api/A.java b/api/A.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/A.java\n"
        b"+++ b/api/A.java\n"
        b"@@ -1 +1 @@\n"
        b"-oldA\n"
        b"+newA\n"
        b"diff --git a/api/B.java b/api/B.java\n"
        b"index 3333333..4444444 100644\n"
        b"--- a/api/B.java\n"
        b"+++ b/api/B.java\n"
        b"@@ -1 +1 @@\n"
        b"-oldB\n"
        b"+newB\n"
    )
    numstat_bytes = b"1\t1\tapi/A.java\x001\t1\tapi/B.java\x00"

    class OversizeStore:
        def diff(self, old_sha: str, new_sha: str) -> tuple[bytes, bytes]:
            return patch_bytes, numstat_bytes

        def git_version(self) -> str:
            return "2.47.1"

    with pytest.raises(DiffBuildError, match="analysis_limit_exceeded"):
        DiffBuilder(
            OversizeStore(),
            max_files=1,
            max_bytes=1_000_000,
        ).build_all(snapshot)


def test_artifact_hash_exactly_binds_the_recorded_diff_content() -> None:
    """Changing any recorded diff field would change its artifact SHA-256."""
    artifact = parse_patch(
        kind="integration_diff",
        old_sha="b" * 40,
        new_sha="c" * 40,
        patch_bytes=(
            b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
            b"index 1111111..2222222 100644\n"
            b"--- a/api/PatientService.java\n"
            b"+++ b/api/PatientService.java\n"
            b"@@ -10 +10 @@\n"
            b"-return oldValue;\n"
            b"+return newValue;\n"
        ),
        numstat_bytes=b"1\t1\tapi/PatientService.java\0",
        git_version="2.47.1",
    )

    assert artifact.artifact_sha256 == canonical_sha256(
        artifact.model_dump(mode="json", exclude={"artifact_sha256"})
    )


def test_analysis_package_exports_its_public_diff_tools() -> None:
    """Later workflow code imports Task 4 tools from one stable package boundary."""
    assert PublicDiffBuilder is DiffBuilder
    assert PublicDiffBuildError is DiffBuildError
    assert public_parse_patch is parse_patch


def test_parse_patch_records_an_empty_but_valid_frozen_comparison() -> None:
    """An empty comparison is evidence of no source-file change, not an error."""
    artifact = parse_patch(
        kind="base_drift_diff",
        old_sha="b" * 40,
        new_sha="c" * 40,
        patch_bytes=b"",
        numstat_bytes=b"",
        git_version="2.47.1",
    )

    assert artifact.kind == "base_drift_diff"
    assert artifact.files == ()
    assert artifact.patch_sha256 == hashlib.sha256(b"").hexdigest()


def test_parse_patch_rejects_a_patch_and_manifest_with_different_file_sets() -> None:
    """A partial file parse cannot become an apparently complete artifact."""
    with pytest.raises(DiffBuildError, match="diff_manifest_mismatch") as error:
        parse_patch(
            kind="integration_diff",
            old_sha="b" * 40,
            new_sha="c" * 40,
            patch_bytes=(
                b"diff --git a/api/A.java b/api/A.java\n"
                b"index 1111111..2222222 100644\n"
                b"--- a/api/A.java\n"
                b"+++ b/api/A.java\n"
                b"@@ -1 +1 @@\n"
                b"-oldA\n"
                b"+newA\n"
            ),
            numstat_bytes=b"1\t1\tapi/A.java\x001\t1\tapi/B.java\x00",
            git_version="2.47.1",
        )

    assert error.value.reason_code == "diff_manifest_mismatch"
