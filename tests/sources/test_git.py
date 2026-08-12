"""Tests for the constrained local-Git command boundary."""

import os
import subprocess
from pathlib import Path

import pytest

from triageguard.sources.git import (
    GitCommandError,
    GitCommandRunner,
    GitObjectStore,
)


def test_git_runner_uses_argument_array_and_hardened_child_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Git must run without a shell, hooks, credentials, file transport, or prompts."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"git version 2.47.1\n",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", "/unchanged-home")

    result = GitCommandRunner().run(["version"], cwd=tmp_path)

    assert result == b"git version 2.47.1\n"
    assert calls[0][0] == [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "credential.helper=",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "fetch.writeCommitGraph=false",
        "version",
    ]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["timeout"] == 60.0

    environment = calls[0][1]["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["LC_ALL"] == "C"
    assert environment["HOME"] == os.environ["HOME"]


def test_git_runner_maps_timeout_to_a_safe_reason_code(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A stalled Git command must not expose its output in an error."""

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(
            command,
            60.0,
            output=b"output-that-must-not-appear",
            stderr=b"stderr-that-must-not-appear",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(GitCommandError, match="timed out") as error:
        GitCommandRunner().run(["version"], cwd=tmp_path)

    assert error.value.reason_code == "git_command_timed_out"
    assert "output-that-must-not-appear" not in str(error.value)
    assert "stderr-that-must-not-appear" not in str(error.value)


def test_object_store_initializes_a_bare_repository(
    tmp_path: Path,
) -> None:
    """The analysis cache stores Git objects without creating a working tree."""
    store_path = tmp_path / "analysis-store"

    store = GitObjectStore(store_path)
    store.initialize()

    assert (store_path / "HEAD").is_file()
    assert (store_path / "objects").is_dir()
    assert not (store_path / ".git").exists()


def test_object_store_resolves_only_allowlisted_local_refs(
    tmp_path: Path,
) -> None:
    """A Git revision lookup cannot interpret an arbitrary user-controlled ref."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
        ) -> bytes:
            self.calls.append(arguments)
            return (b"a" * 40) + b"\n"

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    assert store.resolve_commit("refs/triageguard/base") == "a" * 40
    assert runner.calls == [
        [
            "--git-dir",
            str(store_path),
            "rev-parse",
            "--verify",
            "refs/triageguard/base^{commit}",
        ]
    ]

    with pytest.raises(ValueError, match="allowlisted local ref"):
        store.resolve_commit("HEAD")


def test_object_store_reads_commit_parents_in_git_order(
    tmp_path: Path,
) -> None:
    """Candidate validation depends on Git's exact first-parent/second-parent order."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
        ) -> bytes:
            self.calls.append(arguments)
            return (b"b" * 40) + b" " + (b"c" * 40) + b"\n"

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    assert store.commit_parents("a" * 40) == ("b" * 40, "c" * 40)
    assert runner.calls == [
        [
            "--git-dir",
            str(store_path),
            "show",
            "-s",
            "--format=%P",
            "a" * 40,
        ]
    ]


def test_object_store_reads_the_exact_tree_for_a_commit(
    tmp_path: Path,
) -> None:
    """A snapshot records the tree behind each frozen commit."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
        ) -> bytes:
            self.calls.append(arguments)
            return (b"d" * 40) + b"\n"

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    assert store.tree_sha("a" * 40) == "d" * 40
    assert runner.calls == [
        [
            "--git-dir",
            str(store_path),
            "show",
            "-s",
            "--format=%T",
            "a" * 40,
        ]
    ]


def test_object_store_finds_one_exact_merge_base(
    tmp_path: Path,
) -> None:
    """The author diff begins at the one unambiguous common ancestor."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
        ) -> bytes:
            self.calls.append(arguments)
            return (b"d" * 40) + b"\n"

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    assert store.merge_base("b" * 40, "c" * 40) == "d" * 40
    assert runner.calls == [
        [
            "--git-dir",
            str(store_path),
            "merge-base",
            "--all",
            "b" * 40,
            "c" * 40,
        ]
    ]


def test_object_store_fetches_only_the_base_and_two_pr_refs(
    tmp_path: Path,
) -> None:
    """The local store fetches only the three refs needed for one frozen PR."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], float]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
            *,
            timeout_seconds: float = 60.0,
        ) -> bytes:
            self.calls.append((arguments, timeout_seconds))
            return b""

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    store.fetch_snapshot("main", 7312)

    assert runner.calls == [
        (
            [
                "--git-dir",
                str(store_path),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                "https://github.com/openmrs/openmrs-core.git",
                "refs/heads/main:refs/triageguard/base",
                "refs/pull/7312/head:refs/triageguard/head",
                "refs/pull/7312/merge:refs/triageguard/candidate",
            ],
            180.0,
        )
    ]


def test_object_store_reads_one_bounded_blob_by_object_id(
    tmp_path: Path,
) -> None:
    """Source bytes are read only after their exact Git object size is checked."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], float]] = []
            self.responses = [b"13\n", b"class Patient"]

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
            *,
            timeout_seconds: float = 60.0,
        ) -> bytes:
            self.calls.append((arguments, timeout_seconds))
            return self.responses.pop(0)

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    assert store.read_blob("a" * 40, max_bytes=13) == b"class Patient"
    assert runner.calls == [
        (
            [
                "--git-dir",
                str(store_path),
                "cat-file",
                "-s",
                "a" * 40,
            ],
            60.0,
        ),
        (
            [
                "--git-dir",
                str(store_path),
                "cat-file",
                "blob",
                "a" * 40,
            ],
            60.0,
        ),
    ]


def test_object_store_lists_exact_tree_entries_without_using_paths_as_refs(
    tmp_path: Path,
) -> None:
    """A tree listing retains modes, types, object IDs, and literal paths."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
            *,
            timeout_seconds: float = 60.0,
        ) -> bytes:
            self.calls.append(arguments)
            return (
                b"100644 blob "
                + (b"b" * 40)
                + b"\tapi/Patient.java\0"
                + b"160000 commit "
                + (b"c" * 40)
                + b"\tmodules/example-module\0"
            )

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    entries = store.list_tree("a" * 40)

    assert [
        (entry.mode, entry.object_type, entry.object_sha, entry.path)
        for entry in entries
    ] == [
        ("100644", "blob", "b" * 40, "api/Patient.java"),
        ("160000", "commit", "c" * 40, "modules/example-module"),
    ]
    assert runner.calls == [
        [
            "--git-dir",
            str(store_path),
            "ls-tree",
            "-r",
            "-z",
            "a" * 40,
        ]
    ]


def test_object_store_records_the_git_version_used_for_acquisition(
    tmp_path: Path,
) -> None:
    """A frozen snapshot records the Git version that read its objects."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
            *,
            timeout_seconds: float = 60.0,
        ) -> bytes:
            self.calls.append(arguments)
            return b"git version 2.47.1\n"

    runner = RecordingRunner()
    store = GitObjectStore(tmp_path / "analysis-store", runner=runner)

    assert store.git_version() == "2.47.1"
    assert runner.calls == [["version"]]


def test_object_store_rejects_ambiguous_multiple_merge_bases(
    tmp_path: Path,
) -> None:
    """A reproducible snapshot must not silently choose between merge bases."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
            *,
            timeout_seconds: float = 60.0,
        ) -> bytes:
            self.calls.append(arguments)
            return (b"d" * 40) + b"\n" + (b"e" * 40) + b"\n"

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    with pytest.raises(GitCommandError, match="one full merge-base SHA") as error:
        store.merge_base("b" * 40, "c" * 40)

    assert error.value.reason_code == "git_command_invalid_output"
    assert runner.calls == [
        [
            "--git-dir",
            str(store_path),
            "merge-base",
            "--all",
            "b" * 40,
            "c" * 40,
        ]
    ]


def test_object_store_reads_one_diff_and_matching_numstat_with_fixed_options(
    tmp_path: Path,
) -> None:
    """Each frozen comparison uses fixed, non-executable Git diff options."""

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], float]] = []

        def run(
            self,
            arguments: list[str],
            cwd: Path | None = None,
            *,
            timeout_seconds: float = 60.0,
        ) -> bytes:
            self.calls.append((arguments, timeout_seconds))
            if "--numstat" in arguments:
                return b"1\t1\tapi/PatientService.java\0"
            return b"diff --git a/api/PatientService.java b/api/PatientService.java\n"

    runner = RecordingRunner()
    store_path = tmp_path / "analysis-store"
    store = GitObjectStore(store_path, runner=runner)

    patch_bytes, numstat_bytes = store.diff("a" * 40, "b" * 40)

    assert patch_bytes.startswith(b"diff --git ")
    assert numstat_bytes == b"1\t1\tapi/PatientService.java\0"
    assert runner.calls == [
        (
            [
                "--git-dir",
                str(store_path),
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--find-renames=50%",
                "--unified=3",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "a" * 40,
                "b" * 40,
                "--",
            ],
            60.0,
        ),
        (
            [
                "--git-dir",
                str(store_path),
                "diff",
                "--numstat",
                "-z",
                "--find-renames=50%",
                "a" * 40,
                "b" * 40,
                "--",
            ],
            60.0,
        ),
    ]
