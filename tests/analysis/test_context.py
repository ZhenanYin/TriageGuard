"""Tests for bounded, traceable Java evidence context."""

"""Tests for bounded, traceable Java evidence context."""

from datetime import UTC, datetime

import pytest

from triageguard.analysis import ContextBuilder as PublicContextBuilder
from triageguard.analysis import ContextLimits as PublicContextLimits
from triageguard.analysis.context import (
    ContextBuilder,
    ContextBuildError,
    ContextLimits,
    JavaSyntaxExtractor,
)
from triageguard.analysis.diffs import parse_patch
from triageguard.config import Settings
from triageguard.domain.pr_analysis import PullRequestSnapshot
from triageguard.sources.git import GitTreeEntry


def test_java_extractor_finds_security_relevant_structure() -> None:
    """Java syntax is read as code, not guessed from plain text."""
    index = JavaSyntaxExtractor().extract(
        "PatientService.java",
        (
            b"package org.openmrs.api;\n"
            b"import org.openmrs.annotation.Authorized;\n"
            b"class PatientService {\n"
            b'    @Authorized("Delete Patients")\n'
            b"    void purgePatient() {\n"
            b"        dao.deletePatient();\n"
            b"    }\n"
            b"}\n"
        ),
    )

    assert index.package == "org.openmrs.api"
    assert index.annotations == ("Authorized",)
    assert index.methods == ("purgePatient",)
    assert index.invocations == ("deletePatient",)


def test_java_extractor_indexes_code_declarations_but_ignores_comment_text() -> None:
    """Only Java syntax—not comments or strings—becomes searchable evidence."""
    index = JavaSyntaxExtractor().extract(
        "PatientService.java",
        (
            b"package org.openmrs.api;\n"
            b"import org.openmrs.annotation.Authorized;\n"
            b"interface PatientGateway {}\n"
            b"enum DeletionMode { HARD }\n"
            b"record DeletionRequest(String patientId) {}\n"
            b"class PatientService {\n"
            b"    PatientService() {}\n"
            b'    @Authorized("Delete Patients")\n'
            b"    void purgePatient() {\n"
            b"        dao.deletePatient();\n"
            b'        String fake = "void pretendMethod()";\n'
            b"        // @Authorized void inventedMethod() {}\n"
            b"    }\n"
            b"}\n"
        ),
    )

    assert index.imports == ("org.openmrs.annotation.Authorized",)
    assert index.interfaces == ("PatientGateway",)
    assert index.enums == ("DeletionMode",)
    assert index.records == ("DeletionRequest",)
    assert index.classes == ("PatientService",)
    assert index.constructors == ("PatientService",)
    assert index.methods == ("purgePatient",)
    assert index.invocations == ("deletePatient",)


def test_java_extractor_rejects_a_syntax_error() -> None:
    """Incomplete Java must not become apparently trustworthy evidence."""
    with pytest.raises(ContextBuildError, match="java_parse_failed") as error:
        JavaSyntaxExtractor().extract(
            "PatientService.java",
            b"class PatientService { void purgePatient() {",
        )

    assert error.value.reason_code == "java_parse_failed"


def _snapshot() -> PullRequestSnapshot:
    """Return one fixed four-revision PR snapshot for context tests."""
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


def _diffs(snapshot: PullRequestSnapshot) -> tuple[object, ...]:
    """Return the required three comparisons with one integration change."""
    integration_patch = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -4 +4 @@\n"
        b"-        dao.requirePrivilege();\n"
        b"+        dao.deletePatient();\n"
    )
    return (
        parse_patch(
            kind="author_diff",
            old_sha=snapshot.merge_base_sha,
            new_sha=snapshot.head_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version=snapshot.git_version,
        ),
        parse_patch(
            kind="integration_diff",
            old_sha=snapshot.base_sha,
            new_sha=snapshot.candidate_sha,
            patch_bytes=integration_patch,
            numstat_bytes=b"1\t1\tapi/PatientService.java\0",
            git_version=snapshot.git_version,
        ),
        parse_patch(
            kind="base_drift_diff",
            old_sha=snapshot.merge_base_sha,
            new_sha=snapshot.base_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version=snapshot.git_version,
        ),
    )


def test_context_builder_reserves_evidence_for_an_integration_hunk() -> None:
    """The primary predicted-merge change is always represented as evidence."""
    snapshot = _snapshot()
    source = (
        b"package org.openmrs.api;\n"
        b"class PatientService {\n"
        b"    void purgePatient() {\n"
        b"        dao.deletePatient();\n"
        b"    }\n"
        b"}\n"
    )

    class FakeStore:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha == snapshot.candidate_sha
            return (
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha="4" * 40,
                    path="api/PatientService.java",
                ),
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            assert blob_sha == "4" * 40
            assert max_bytes == 1_000
            return source

    limits = ContextLimits(
        max_files=1,
        max_anchors=1,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        store=FakeStore(),
        limits=limits,
    )

    assert bundle.primary_change_represented is True
    assert bundle.selected_file_count == 1
    assert bundle.selected_anchor_count == 1

    anchor = bundle.anchors[0]
    assert anchor.revision_role == "candidate"
    assert anchor.commit_sha == snapshot.candidate_sha
    assert anchor.blob_sha == "4" * 40
    assert anchor.path == "api/PatientService.java"
    assert anchor.change_relation == "integration_change"
    assert "dao.deletePatient();" in anchor.text
    assert anchor.java_symbol == "purgePatient"
    assert anchor.score_components[0].name == "integration_hunk"
    assert anchor.score_components[0].value == 100


@pytest.mark.parametrize(
    ("mode", "object_type"),
    [
        ("120000", "blob"),
        ("160000", "commit"),
    ],
)
def test_context_builder_excludes_links_and_submodules_without_opening_them(
    mode: str,
    object_type: str,
) -> None:
    """A symlink or submodule is recorded, never treated as Java source."""
    snapshot = _snapshot()

    class FakeStore:
        def __init__(self) -> None:
            self.read_calls = 0

        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha == snapshot.candidate_sha
            return (
                GitTreeEntry(
                    mode=mode,
                    object_type=object_type,
                    object_sha="4" * 40,
                    path="api/PatientService.java",
                ),
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            self.read_calls += 1
            raise AssertionError("A link or submodule must never be opened.")

    limits = ContextLimits(
        max_files=1,
        max_anchors=1,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )
    store = FakeStore()

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        store=store,
        limits=limits,
    )

    assert bundle.primary_change_represented is False
    assert bundle.anchors == ()
    assert bundle.excluded_paths == ("api/PatientService.java",)
    assert store.read_calls == 0


def test_context_builder_records_a_binary_primary_change_as_unrepresented() -> None:
    """A binary change is retained as evidence metadata, never source text."""
    snapshot = _snapshot()
    author_diff, _, base_drift_diff = _diffs(snapshot)
    binary_integration = parse_patch(
        kind="integration_diff",
        old_sha=snapshot.base_sha,
        new_sha=snapshot.candidate_sha,
        patch_bytes=(
            b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
            b"--- a/api/PatientService.java\n"
            b"+++ b/api/PatientService.java\n"
            b"Binary files a/api/PatientService.java "
            b"and b/api/PatientService.java differ\n"
        ),
        numstat_bytes=b"-\t-\tapi/PatientService.java\0",
        git_version=snapshot.git_version,
    )

    class FakeStore:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            raise AssertionError("A binary file must not be opened as source.")

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            raise AssertionError("A binary file must not be opened as source.")

    limits = ContextLimits(
        max_files=1,
        max_anchors=1,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=(author_diff, binary_integration, base_drift_diff),
        store=FakeStore(),
        limits=limits,
    )

    assert bundle.primary_change_represented is False
    assert bundle.anchors == ()
    assert bundle.binary_paths == ("api/PatientService.java",)


def test_context_builder_ranks_hunks_from_all_three_frozen_comparisons() -> None:
    """Integration evidence ranks before author change, then base drift."""
    snapshot = _snapshot()

    def patch(old_line: bytes, new_line: bytes) -> bytes:
        return (
            b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
            b"index 1111111..2222222 100644\n"
            b"--- a/api/PatientService.java\n"
            b"+++ b/api/PatientService.java\n"
            b"@@ -4 +4 @@\n"
            b"-        " + old_line + b"\n"
            b"+        " + new_line + b"\n"
        )

    author_diff = parse_patch(
        kind="author_diff",
        old_sha=snapshot.merge_base_sha,
        new_sha=snapshot.head_sha,
        patch_bytes=patch(b"dao.oldAuthorChange();", b"dao.authorChange();"),
        numstat_bytes=b"1\t1\tapi/PatientService.java\0",
        git_version=snapshot.git_version,
    )
    integration_diff = parse_patch(
        kind="integration_diff",
        old_sha=snapshot.base_sha,
        new_sha=snapshot.candidate_sha,
        patch_bytes=patch(b"dao.requirePrivilege();", b"dao.deletePatient();"),
        numstat_bytes=b"1\t1\tapi/PatientService.java\0",
        git_version=snapshot.git_version,
    )
    base_drift_diff = parse_patch(
        kind="base_drift_diff",
        old_sha=snapshot.merge_base_sha,
        new_sha=snapshot.base_sha,
        patch_bytes=patch(b"dao.oldBaseChange();", b"dao.baseDriftChange();"),
        numstat_bytes=b"1\t1\tapi/PatientService.java\0",
        git_version=snapshot.git_version,
    )

    sources = {
        snapshot.head_sha: (
            b"package org.openmrs.api;\n"
            b"class PatientService {\n"
            b"    void purgePatient() {\n"
            b"        dao.authorChange();\n"
            b"    }\n"
            b"}\n"
        ),
        snapshot.candidate_sha: (
            b"package org.openmrs.api;\n"
            b"class PatientService {\n"
            b"    void purgePatient() {\n"
            b"        dao.deletePatient();\n"
            b"    }\n"
            b"}\n"
        ),
        snapshot.base_sha: (
            b"package org.openmrs.api;\n"
            b"class PatientService {\n"
            b"    void purgePatient() {\n"
            b"        dao.baseDriftChange();\n"
            b"    }\n"
            b"}\n"
        ),
    }

    blob_shas = {
        snapshot.head_sha: "4" * 40,
        snapshot.candidate_sha: "5" * 40,
        snapshot.base_sha: "6" * 40,
    }

    class FakeStore:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            return (
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha=blob_shas[commit_sha],
                    path="api/PatientService.java",
                ),
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            commit_sha = next(
                sha
                for sha, expected_blob_sha in blob_shas.items()
                if expected_blob_sha == blob_sha
            )
            return sources[commit_sha]

    limits = ContextLimits(
        max_files=1,
        max_anchors=3,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=(author_diff, integration_diff, base_drift_diff),
        store=FakeStore(),
        limits=limits,
    )

    assert [anchor.change_relation for anchor in bundle.anchors] == [
        "integration_change",
        "author_change",
        "base_drift_change",
    ]
    assert [anchor.revision_role for anchor in bundle.anchors] == [
        "candidate",
        "head",
        "base",
    ]
    assert [anchor.score_components[0].value for anchor in bundle.anchors] == [
        100,
        60,
        40,
    ]


def test_context_builder_adds_scored_repository_context_for_an_exact_identifier() -> (
    None
):
    """Related candidate code is selected by exact identifiers and fixed scores."""
    snapshot = _snapshot()
    primary_source = (
        b"package org.openmrs.api;\n"
        b"class PatientService {\n"
        b"    void purgePatient() {\n"
        b"        dao.deletePatient();\n"
        b"    }\n"
        b"}\n"
    )
    related_source = (
        b"package org.openmrs.api;\n"
        b"import org.openmrs.annotation.Authorized;\n"
        b"class PatientDao {\n"
        b'    @Authorized("Delete Patients")\n'
        b"    void deletePatient() {\n"
        b"        deleteFromDatabase();\n"
        b"    }\n"
        b"}\n"
    )

    class FakeStore:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha == snapshot.candidate_sha
            return (
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha="4" * 40,
                    path="api/PatientService.java",
                ),
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha="5" * 40,
                    path="api/PatientDao.java",
                ),
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            return {
                "4" * 40: primary_source,
                "5" * 40: related_source,
            }[blob_sha]

    limits = ContextLimits(
        max_files=2,
        max_anchors=2,
        max_total_bytes=2_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        store=FakeStore(),
        limits=limits,
    )

    assert [anchor.change_relation for anchor in bundle.anchors] == [
        "integration_change",
        "repository_context",
    ]

    related_anchor = bundle.anchors[1]
    assert related_anchor.path == "api/PatientDao.java"
    assert related_anchor.java_symbol == "deletePatient"
    assert [
        (component.name, component.value)
        for component in related_anchor.score_components
    ] == [
        ("same_symbol", 30),
        ("security_signal", 20),
        ("same_package", 10),
    ]


def test_context_builder_rejects_a_diff_that_is_not_bound_to_the_snapshot() -> None:
    """Evidence from a different revision pair must never enter this run."""
    snapshot = _snapshot()
    author_diff, integration_diff, base_drift_diff = _diffs(snapshot)
    wrong_integration_diff = integration_diff.model_copy(
        update={"old_revision": "9" * 40},
    )

    class StoreMustNotBeRead:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            raise AssertionError("Revision binding must be checked before Git reads.")

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            raise AssertionError("Revision binding must be checked before Git reads.")

    limits = ContextLimits(
        max_files=1,
        max_anchors=1,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    with pytest.raises(ContextBuildError, match="diff_snapshot_mismatch") as error:
        ContextBuilder().build(
            snapshot=snapshot,
            diffs=(author_diff, wrong_integration_diff, base_drift_diff),
            store=StoreMustNotBeRead(),
            limits=limits,
        )

    assert error.value.reason_code == "diff_snapshot_mismatch"


def test_context_limits_copy_every_approved_setting() -> None:
    """One context run receives the exact public bounds chosen for the study."""
    settings = Settings(
        max_context_files=2,
        max_context_anchors=3,
        max_context_bytes=4_000,
        max_context_anchor_lines=5,
        max_context_blob_bytes=6_000,
        max_context_search_identifiers=7,
        max_context_hits_per_identifier=8,
    )

    limits = ContextLimits.from_settings(settings)

    assert limits == ContextLimits(
        max_files=2,
        max_anchors=3,
        max_total_bytes=4_000,
        max_anchor_lines=5,
        max_blob_bytes=6_000,
        max_search_identifiers=7,
        max_hits_per_identifier=8,
    )


def test_context_builder_removes_duplicate_primary_hunk_anchors() -> None:
    """Repeated identical diff hunks produce one stable evidence anchor."""
    snapshot = _snapshot()
    author_diff, _, base_drift_diff = _diffs(snapshot)
    repeated_hunk_patch = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -4 +4 @@\n"
        b"-        dao.requirePrivilege();\n"
        b"+        dao.deletePatient();\n"
        b"@@ -4 +4 @@\n"
        b"-        dao.requirePrivilege();\n"
        b"+        dao.deletePatient();\n"
    )
    integration_diff = parse_patch(
        kind="integration_diff",
        old_sha=snapshot.base_sha,
        new_sha=snapshot.candidate_sha,
        patch_bytes=repeated_hunk_patch,
        numstat_bytes=b"1\t1\tapi/PatientService.java\0",
        git_version=snapshot.git_version,
    )
    source = (
        b"package org.openmrs.api;\n"
        b"class PatientService {\n"
        b"    void purgePatient() {\n"
        b"        dao.deletePatient();\n"
        b"    }\n"
        b"}\n"
    )

    class FakeStore:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha == snapshot.candidate_sha
            return (
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha="4" * 40,
                    path="api/PatientService.java",
                ),
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            return source

    limits = ContextLimits(
        max_files=1,
        max_anchors=1,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=(author_diff, integration_diff, base_drift_diff),
        store=FakeStore(),
        limits=limits,
    )

    assert bundle.primary_change_represented is True
    assert bundle.selected_anchor_count == 1
    assert len(bundle.anchors) == 1


def test_analysis_package_exports_the_context_builder_api() -> None:
    """Later workflow code receives the supported public context API."""
    assert PublicContextBuilder is ContextBuilder
    assert PublicContextLimits is ContextLimits


def test_context_builder_rejects_a_diff_whose_content_hash_was_changed() -> None:
    """A modified diff cannot reuse the hash from a different artifact."""
    snapshot = _snapshot()
    author_diff, integration_diff, base_drift_diff = _diffs(snapshot)
    forged_integration_diff = integration_diff.model_copy(
        update={"files": ()},
    )

    class StoreMustNotBeRead:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            raise AssertionError("Diff hashes must be checked before Git reads.")

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            raise AssertionError("Diff hashes must be checked before Git reads.")

    limits = ContextLimits(
        max_files=1,
        max_anchors=1,
        max_total_bytes=1_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    with pytest.raises(ContextBuildError, match="diff_hash_mismatch") as error:
        ContextBuilder().build(
            snapshot=snapshot,
            diffs=(author_diff, forged_integration_diff, base_drift_diff),
            store=StoreMustNotBeRead(),
            limits=limits,
        )

    assert error.value.reason_code == "diff_hash_mismatch"


def test_repository_context_uses_invocation_test_and_stable_path_scores() -> None:
    """Fixed scoring stays explainable when several exact matches exist."""
    snapshot = _snapshot()
    primary_source = (
        b"package org.openmrs.api;\n"
        b"class PatientService {\n"
        b"    void purgePatient() {\n"
        b"        dao.deletePatient();\n"
        b"    }\n"
        b"}\n"
    )

    def related_source(class_name: str, body: bytes) -> bytes:
        return (
            b"package org.openmrs.api;\n"
            + f"class {class_name} {{\n".encode()
            + b"    void deletePatient() {\n"
            + body
            + b"    }\n"
            + b"}\n"
        )

    sources = {
        "4" * 40: primary_source,
        "5" * 40: related_source(
            "AlphaPatientDao",
            b"        deleteFromDatabase();\n",
        ),
        "6" * 40: related_source(
            "BetaPatientDao",
            b"        deleteFromDatabase();\n",
        ),
        "7" * 40: related_source(
            "PatientAuditTest",
            b"        purgePatient();\n",
        ),
    }
    paths = {
        "4" * 40: "api/PatientService.java",
        "5" * 40: "api/AlphaPatientDao.java",
        "6" * 40: "api/BetaPatientDao.java",
        "7" * 40: "api/PatientAuditTest.java",
    }

    class FakeStore:
        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha == snapshot.candidate_sha
            return tuple(
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha=blob_sha,
                    path=paths[blob_sha],
                )
                for blob_sha in ("4" * 40, "5" * 40, "6" * 40, "7" * 40)
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            return sources[blob_sha]

    limits = ContextLimits(
        max_files=4,
        max_anchors=4,
        max_total_bytes=4_000,
        max_anchor_lines=10,
        max_blob_bytes=1_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    bundle = ContextBuilder().build(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        store=FakeStore(),
        limits=limits,
    )

    assert [anchor.path for anchor in bundle.anchors] == [
        "api/PatientService.java",
        "api/PatientAuditTest.java",
        "api/AlphaPatientDao.java",
        "api/BetaPatientDao.java",
    ]
    assert [
        (component.name, component.value)
        for component in bundle.anchors[1].score_components
    ] == [
        ("same_symbol", 30),
        ("import_or_invocation", 15),
        ("same_package", 10),
        ("related_test", 8),
    ]
    assert [anchor.path for anchor in bundle.anchors[2:]] == [
        "api/AlphaPatientDao.java",
        "api/BetaPatientDao.java",
    ]
