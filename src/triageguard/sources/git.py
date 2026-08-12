"""Constrained local-Git commands for frozen Milestone 2 evidence."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_LOCAL_REFS = frozenset(
    {
        "refs/triageguard/base",
        "refs/triageguard/head",
        "refs/triageguard/candidate",
    }
)


class GitCommandError(RuntimeError):
    """A safe local-Git failure without copied command output."""

    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.reason_code = reason_code
        self.safe_message = safe_message


class GitCommandRunner:
    """Run a small allowlisted Git argument array without invoking a shell."""

    def run(
        self,
        arguments: Sequence[str],
        cwd: Path | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> bytes:
        """Run one Git command with fixed security controls."""
        if not arguments:
            raise ValueError("Git arguments must not be empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("Git timeout must be a positive number")
        if any(not isinstance(argument, str) or not argument for argument in arguments):
            raise TypeError("Git arguments must be non-empty strings")

        command = [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "fetch.writeCommitGraph=false",
            *arguments,
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            }
        )

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                shell=False,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise GitCommandError(
                "git_command_timed_out",
                "Git command timed out.",
            ) from error
        except OSError as error:
            raise GitCommandError(
                "git_command_start_failed",
                "Git command could not start.",
            ) from error

        if completed.returncode != 0:
            raise GitCommandError(
                "git_command_failed",
                "Git command failed.",
            )

        if not isinstance(completed.stdout, bytes):
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return byte output.",
            )

        return completed.stdout


@dataclass(frozen=True)
class GitTreeEntry:
    """One literal entry from a frozen Git tree."""

    mode: Literal["040000", "100644", "100755", "120000", "160000"]
    object_type: Literal["blob", "tree", "commit"]
    object_sha: str
    path: str


class GitObjectStore:
    """A local bare repository containing only frozen analysis Git objects."""

    def __init__(
        self,
        root: Path,
        runner: GitCommandRunner | None = None,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("Git object-store root must be a Path")

        self._root = root
        self._runner = runner if runner is not None else GitCommandRunner()

    @property
    def root(self) -> Path:
        """Return the local bare-repository directory."""
        return self._root

    def initialize(self) -> None:
        """Create the empty bare repository without a checked-out working tree."""
        self._root.parent.mkdir(parents=True, exist_ok=True)
        self._runner.run(["init", "--bare", str(self._root)])

    def resolve_commit(self, ref: str) -> str:
        """Resolve only an exact SHA or one of TriageGuard's local snapshot refs."""
        if not isinstance(ref, str):
            raise TypeError("Git ref must be a string")
        if ref not in _ALLOWED_LOCAL_REFS and _FULL_COMMIT_SHA.fullmatch(ref) is None:
            raise ValueError(
                "Git ref must be an allowlisted local ref or full commit SHA"
            )

        output = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "rev-parse",
                "--verify",
                f"{ref}^{{commit}}",
            ]
        )

        try:
            resolved = output.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return an ASCII commit SHA.",
            ) from error

        if _FULL_COMMIT_SHA.fullmatch(resolved) is None:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return a full commit SHA.",
            )

        return resolved

    def commit_parents(self, commit_sha: str) -> tuple[str, ...]:
        """Return a commit's parent SHAs in Git's recorded order."""
        if _FULL_COMMIT_SHA.fullmatch(commit_sha) is None:
            raise ValueError("commit SHA must be a full 40-character lowercase SHA")

        output = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "show",
                "-s",
                "--format=%P",
                commit_sha,
            ]
        )

        try:
            parent_text = output.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return ASCII parent SHAs.",
            ) from error

        if not parent_text:
            return ()

        parents = tuple(parent_text.split())
        if any(_FULL_COMMIT_SHA.fullmatch(parent) is None for parent in parents):
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return full parent SHAs.",
            )

        return parents

    def tree_sha(self, commit_sha: str) -> str:
        """Return the exact tree SHA recorded by one frozen commit."""
        if _FULL_COMMIT_SHA.fullmatch(commit_sha) is None:
            raise ValueError("commit SHA must be a full 40-character lowercase SHA")

        output = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "show",
                "-s",
                "--format=%T",
                commit_sha,
            ]
        )

        try:
            tree_sha = output.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return an ASCII tree SHA.",
            ) from error

        if _FULL_COMMIT_SHA.fullmatch(tree_sha) is None:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return a full tree SHA.",
            )

        return tree_sha

    def merge_base(self, base_sha: str, head_sha: str) -> str:
        """Return the one shared ancestor of the frozen base and PR-head commits."""
        if _FULL_COMMIT_SHA.fullmatch(base_sha) is None:
            raise ValueError("base SHA must be a full 40-character lowercase SHA")
        if _FULL_COMMIT_SHA.fullmatch(head_sha) is None:
            raise ValueError("head SHA must be a full 40-character lowercase SHA")

        output = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "merge-base",
                "--all",
                base_sha,
                head_sha,
            ]
        )

        try:
            merge_base_sha = output.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return an ASCII merge-base SHA.",
            ) from error

        if _FULL_COMMIT_SHA.fullmatch(merge_base_sha) is None:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return one full merge-base SHA.",
            )

        return merge_base_sha

    def fetch_snapshot(self, base_branch: str, pull_number: int) -> None:
        """Fetch only the frozen base, PR head, and GitHub merge candidate refs."""
        if (
            not isinstance(base_branch, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", base_branch)
            or ".." in base_branch
            or "//" in base_branch
            or base_branch.endswith(("/", "."))
        ):
            raise ValueError("base branch must be a safe Git branch name")
        if isinstance(pull_number, bool) or not isinstance(pull_number, int):
            raise TypeError("pull request number must be an integer")
        if pull_number <= 0:
            raise ValueError("pull request number must be positive")

        self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                "https://github.com/openmrs/openmrs-core.git",
                f"refs/heads/{base_branch}:refs/triageguard/base",
                f"refs/pull/{pull_number}/head:refs/triageguard/head",
                f"refs/pull/{pull_number}/merge:refs/triageguard/candidate",
            ],
            timeout_seconds=180.0,
        )

    def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
        """Read one exact blob only after enforcing its configured byte limit."""
        if _FULL_COMMIT_SHA.fullmatch(blob_sha) is None:
            raise ValueError("blob SHA must be a full 40-character lowercase SHA")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")

        size_output = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "cat-file",
                "-s",
                blob_sha,
            ]
        )
        try:
            size_text = size_output.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return an ASCII blob size.",
            ) from error

        if re.fullmatch(r"(?:0|[1-9][0-9]*)", size_text) is None:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return a valid blob size.",
            )

        blob_size = int(size_text)
        if blob_size > max_bytes:
            raise GitCommandError(
                "blob_too_large",
                "Git blob exceeds the configured byte limit.",
            )

        blob = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "cat-file",
                "blob",
                blob_sha,
            ]
        )
        if len(blob) != blob_size:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git blob size did not match the declared size.",
            )

        return blob

    def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
        """Return literal entries from one exact frozen commit tree."""
        if _FULL_COMMIT_SHA.fullmatch(commit_sha) is None:
            raise ValueError("commit SHA must be a full 40-character lowercase SHA")

        output = self._runner.run(
            [
                "--git-dir",
                str(self._root),
                "ls-tree",
                "-r",
                "-z",
                commit_sha,
            ]
        )
        if not output:
            return ()
        if not output.endswith(b"\0"):
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command returned an incomplete tree listing.",
            )

        entries: list[GitTreeEntry] = []
        allowed_modes = {"040000", "100644", "100755", "120000", "160000"}
        allowed_types = {"blob", "tree", "commit"}

        for raw_entry in output[:-1].split(b"\0"):
            metadata, separator, path_bytes = raw_entry.partition(b"\t")
            fields = metadata.split(b" ")

            if not separator or len(fields) != 3 or not path_bytes:
                raise GitCommandError(
                    "git_command_invalid_output",
                    "Git command returned an invalid tree entry.",
                )

            try:
                mode = fields[0].decode("ascii")
                object_type = fields[1].decode("ascii")
                object_sha = fields[2].decode("ascii")
                path = path_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise GitCommandError(
                    "git_command_invalid_output",
                    "Git command returned a non-text tree entry.",
                ) from error

            if (
                mode not in allowed_modes
                or object_type not in allowed_types
                or _FULL_COMMIT_SHA.fullmatch(object_sha) is None
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
            ):
                raise GitCommandError(
                    "git_command_invalid_output",
                    "Git command returned an invalid tree entry.",
                )

            if (
                (mode == "160000" and object_type != "commit")
                or (mode == "040000" and object_type != "tree")
                or (mode in {"100644", "100755", "120000"} and object_type != "blob")
            ):
                raise GitCommandError(
                    "git_command_invalid_output",
                    "Git command returned an inconsistent tree entry.",
                )

            entries.append(
                GitTreeEntry(
                    mode=mode,
                    object_type=object_type,
                    object_sha=object_sha,
                    path=path,
                )
            )

        return tuple(entries)

    def git_version(self) -> str:
        """Return the installed Git version used to freeze this evidence."""
        output = self._runner.run(["version"])

        try:
            version_text = output.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return an ASCII version.",
            ) from error

        if re.fullmatch(r"git version [ -~]+", version_text) is None:
            raise GitCommandError(
                "git_command_invalid_output",
                "Git command did not return a valid version.",
            )

        return version_text.removeprefix("git version ")
