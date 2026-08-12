"""Parse exact local Git diffs into immutable research artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from triageguard.config import DEFAULT_MAX_DIFF_BYTES, DEFAULT_MAX_DIFF_FILES
from triageguard.domain.pr_analysis import (
    DiffArtifact,
    DiffFile,
    DiffHunk,
    PullRequestSnapshot,
)
from triageguard.provenance import canonical_sha256
from triageguard.sources.git import GitCommandError

_FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_HUNK_HEADER = re.compile(
    rb"^@@ -(?P<old_start>[0-9]+)(?:,(?P<old_count>[0-9]+))? "
    rb"\+(?P<new_start>[0-9]+)(?:,(?P<new_count>[0-9]+))? @@"
)
_DIFF_KINDS = {"author_diff", "integration_diff", "base_drift_diff"}


class DiffBuildError(RuntimeError):
    """A safe, typed failure while building a reproducible diff artifact."""

    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(f"{reason_code}: {safe_message}")
        self.reason_code = reason_code
        self.safe_message = safe_message


class _DiffStore(Protocol):
    """The narrow local-Git operations required to build frozen diffs."""

    def diff(self, old_sha: str, new_sha: str) -> tuple[bytes, bytes]:
        """Return exact patch and numstat bytes for one frozen comparison."""

    def git_version(self) -> str:
        """Return the Git version that generated those bytes."""


class DiffBuilder:
    """Build the three approved reproducible comparisons from one snapshot."""

    def __init__(
        self,
        store: _DiffStore,
        *,
        max_files: int = DEFAULT_MAX_DIFF_FILES,
        max_bytes: int = DEFAULT_MAX_DIFF_BYTES,
    ) -> None:
        _validate_limits(max_files=max_files, max_bytes=max_bytes)
        self._store = store
        self._max_files = max_files
        self._max_bytes = max_bytes

    def build_all(
        self,
        snapshot: PullRequestSnapshot,
    ) -> tuple[DiffArtifact, DiffArtifact, DiffArtifact]:
        """Build author, integration, and base-drift artifacts in that order."""
        git_version = self._store.git_version()
        comparisons = (
            ("author_diff", snapshot.merge_base_sha, snapshot.head_sha),
            ("integration_diff", snapshot.base_sha, snapshot.candidate_sha),
            ("base_drift_diff", snapshot.merge_base_sha, snapshot.base_sha),
        )

        artifacts: list[DiffArtifact] = []
        for kind, old_sha, new_sha in comparisons:
            try:
                patch_bytes, numstat_bytes = self._store.diff(old_sha, new_sha)
            except GitCommandError as error:
                if kind == "integration_diff":
                    raise DiffBuildError(
                        "primary_integration_artifact_missing",
                        "The primary integration diff could not be produced.",
                    ) from error
                raise DiffBuildError(
                    "diff_build_failed",
                    "A required frozen diff could not be produced.",
                ) from error

            artifacts.append(
                parse_patch(
                    kind=kind,
                    old_sha=old_sha,
                    new_sha=new_sha,
                    patch_bytes=patch_bytes,
                    numstat_bytes=numstat_bytes,
                    git_version=git_version,
                    max_files=self._max_files,
                    max_bytes=self._max_bytes,
                )
            )

        return artifacts[0], artifacts[1], artifacts[2]


@dataclass(frozen=True)
class _NumstatEntry:
    """One file record from Git's null-separated numstat output."""

    additions: int
    deletions: int
    old_path: str | None
    new_path: str
    binary: bool


def parse_patch(
    *,
    kind: Literal["author_diff", "integration_diff", "base_drift_diff"],
    old_sha: str,
    new_sha: str,
    patch_bytes: bytes,
    numstat_bytes: bytes,
    git_version: str,
    max_files: int = DEFAULT_MAX_DIFF_FILES,
    max_bytes: int = DEFAULT_MAX_DIFF_BYTES,
) -> DiffArtifact:
    """Convert one exact Git patch and matching numstat into a frozen artifact."""
    _validate_inputs(kind, old_sha, new_sha, patch_bytes, numstat_bytes, git_version)
    if (
        patch_bytes.startswith((b"diff --cc ", b"diff --combined "))
        or b"\ndiff --cc " in patch_bytes
        or b"\ndiff --combined " in patch_bytes
    ):
        raise DiffBuildError(
            "unsupported_combined_diff",
            "Git combined diffs are not supported for this analysis.",
        )
    _validate_limits(max_files=max_files, max_bytes=max_bytes)
    if len(patch_bytes) + len(numstat_bytes) > max_bytes:
        raise DiffBuildError(
            "analysis_limit_exceeded",
            "Raw Git diff input exceeds the configured byte limit.",
        )
    if patch_bytes == b"" and numstat_bytes == b"":
        files: tuple[DiffFile, ...] = ()
    else:
        numstats = _parse_numstat(numstat_bytes)
        files = tuple(_parse_file_blocks(patch_bytes, numstats))

        if {file.new_path or file.old_path for file in files} != {
            entry.new_path for entry in numstats
        }:
            raise DiffBuildError(
                "diff_manifest_mismatch",
                "Patch and numstat did not describe the same changed files.",
            )

    if len(files) > max_files:
        raise DiffBuildError(
            "analysis_limit_exceeded",
            "Git diff exceeds the configured file limit.",
        )

    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    git_arguments = (
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--find-renames=50%",
        "--unified=3",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        old_sha,
        new_sha,
        "--",
    )
    artifact_content = {
        "kind": kind,
        "old_revision": old_sha,
        "new_revision": new_sha,
        "git_arguments": git_arguments,
        "git_version": git_version,
        "files": [item.model_dump(mode="json") for item in files],
        "patch_sha256": patch_sha256,
    }

    return DiffArtifact(
        **artifact_content,
        artifact_sha256=canonical_sha256(artifact_content),
    )


def _validate_inputs(
    kind: str,
    old_sha: str,
    new_sha: str,
    patch_bytes: bytes,
    numstat_bytes: bytes,
    git_version: str,
) -> None:
    """Reject malformed parser inputs before interpreting repository data."""
    if kind not in _DIFF_KINDS:
        raise ValueError("diff kind must be an approved comparison kind")
    if _FULL_COMMIT_SHA.fullmatch(old_sha) is None:
        raise ValueError("old SHA must be a full 40-character lowercase SHA")
    if _FULL_COMMIT_SHA.fullmatch(new_sha) is None:
        raise ValueError("new SHA must be a full 40-character lowercase SHA")
    if old_sha == new_sha:
        raise ValueError("diff revisions must be distinct")
    if not isinstance(patch_bytes, bytes) or not isinstance(numstat_bytes, bytes):
        raise TypeError("patch and numstat values must be bytes")
    if not isinstance(git_version, str) or not git_version:
        raise ValueError("Git version must be a non-empty string")


def _validate_limits(*, max_files: int, max_bytes: int) -> None:
    """Reject disabled or malformed parser resource limits."""
    for field_name, value in (
        ("max_files", max_files),
        ("max_bytes", max_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")


def _parse_numstat(numstat_bytes: bytes) -> tuple[_NumstatEntry, ...]:
    """Parse Git's documented null-separated numstat file manifest."""
    if not numstat_bytes.endswith(b"\0"):
        raise DiffBuildError(
            "diff_parse_failed",
            "Git numstat output was incomplete.",
        )

    records = numstat_bytes[:-1].split(b"\0")
    entries: list[_NumstatEntry] = []
    index = 0

    while index < len(records):
        fields = records[index].split(b"\t")
        if len(fields) != 3:
            raise DiffBuildError(
                "diff_parse_failed",
                "Git numstat output contained an invalid entry.",
            )

        additions_text, deletions_text, path_bytes = fields
        binary = additions_text == b"-" and deletions_text == b"-"

        if binary:
            additions = 0
            deletions = 0
        else:
            try:
                additions = int(additions_text)
                deletions = int(deletions_text)
            except ValueError as error:
                raise DiffBuildError(
                    "diff_parse_failed",
                    "Git numstat output contained invalid line counts.",
                ) from error

            if additions < 0 or deletions < 0:
                raise DiffBuildError(
                    "diff_parse_failed",
                    "Git numstat output contained invalid line counts.",
                )

        if path_bytes:
            path = _decode_path(path_bytes)
            entries.append(
                _NumstatEntry(
                    additions=additions,
                    deletions=deletions,
                    old_path=None,
                    new_path=path,
                    binary=binary,
                )
            )
            index += 1
            continue

        if index + 2 >= len(records):
            raise DiffBuildError(
                "diff_parse_failed",
                "Git numstat output contained an incomplete rename entry.",
            )

        old_path = _decode_path(records[index + 1])
        new_path = _decode_path(records[index + 2])
        entries.append(
            _NumstatEntry(
                additions=additions,
                deletions=deletions,
                old_path=old_path,
                new_path=new_path,
                binary=binary,
            )
        )
        index += 3

    if not entries or len({entry.new_path for entry in entries}) != len(entries):
        raise DiffBuildError(
            "diff_parse_failed",
            "Git numstat output contained duplicate or empty file paths.",
        )

    return tuple(entries)


def _parse_file_blocks(
    patch_bytes: bytes,
    numstats: tuple[_NumstatEntry, ...],
) -> Iterable[DiffFile]:
    """Parse Git file blocks in their exact recorded order."""
    if not patch_bytes.startswith(b"diff --git "):
        raise DiffBuildError(
            "diff_parse_failed",
            "Git patch did not begin with a file header.",
        )

    for raw_block in patch_bytes.split(b"diff --git ")[1:]:
        block = b"diff --git " + raw_block
        lines = block.splitlines()

        if not lines:
            raise DiffBuildError(
                "diff_parse_failed",
                "Git patch contained an incomplete file block.",
            )

        rename_from = _rename_path(lines, b"rename from ")
        rename_to = _rename_path(lines, b"rename to ")

        if (rename_from is None) != (rename_to is None):
            raise DiffBuildError(
                "diff_parse_failed",
                "Git patch contained an incomplete rename record.",
            )

        if rename_from is not None and rename_to is not None:
            old_path = rename_from
            new_path = rename_to
            status = "renamed"
        else:
            old_path = _header_path(lines, b"--- ", b"a/")
            new_path = _header_path(lines, b"+++ ", b"b/")
            status = _file_status(old_path, new_path)

        entry = _matching_numstat(numstats, old_path, new_path)
        binary = any(
            line.startswith(b"Binary files ") or line == b"GIT binary patch"
            for line in lines
        )

        if binary != entry.binary:
            raise DiffBuildError(
                "diff_manifest_mismatch",
                "Git patch binary status did not match its numstat manifest.",
            )

        hunks = tuple(_parse_hunks(lines))
        if binary and hunks:
            raise DiffBuildError(
                "diff_parse_failed",
                "Git patch contained text hunks for a binary file.",
            )

        yield DiffFile(
            status=status,
            old_path=old_path,
            new_path=new_path,
            binary=binary,
            additions=entry.additions,
            deletions=entry.deletions,
            hunks=hunks,
            content_sha256=hashlib.sha256(block).hexdigest(),
        )


def _matching_numstat(
    numstats: tuple[_NumstatEntry, ...],
    old_path: str | None,
    new_path: str | None,
) -> _NumstatEntry:
    """Find the one manifest entry matching one patch file identity."""
    visible_path = new_path or old_path
    if visible_path is None:
        raise DiffBuildError(
            "diff_parse_failed",
            "Git patch did not identify a changed file.",
        )

    matches = [
        entry
        for entry in numstats
        if (
            entry.old_path == old_path and entry.new_path == new_path
            if entry.old_path is not None
            else entry.new_path == visible_path
        )
    ]
    if len(matches) != 1:
        raise DiffBuildError(
            "diff_manifest_mismatch",
            "Git patch file did not match exactly one numstat entry.",
        )

    return matches[0]


def _rename_path(lines: list[bytes], prefix: bytes) -> str | None:
    """Read one literal Git rename path, when a block records a rename."""
    matching_lines = [line for line in lines if line.startswith(prefix)]
    if not matching_lines:
        return None
    if len(matching_lines) != 1:
        raise DiffBuildError(
            "diff_parse_failed",
            "Git patch contained repeated rename metadata.",
        )

    return _decode_path(_decode_patch_path_bytes(matching_lines[0][len(prefix) :]))


def _decode_patch_path_bytes(value: bytes) -> bytes:
    """Decode Git's optional C-style quoted path form into literal bytes."""
    if not value.startswith(b'"'):
        return value
    if len(value) < 2 or not value.endswith(b'"'):
        raise DiffBuildError(
            "diff_parse_failed",
            "Git patch contained an incomplete quoted file path.",
        )

    return _decode_git_c_quoted_bytes(value[1:-1])


def _decode_git_c_quoted_bytes(value: bytes) -> bytes:
    """Decode the limited escape sequences Git uses for quoted path names."""
    escapes = {
        ord("a"): b"\a",
        ord("b"): b"\b",
        ord("f"): b"\f",
        ord("n"): b"\n",
        ord("r"): b"\r",
        ord("t"): b"\t",
        ord("v"): b"\v",
        ord("\\"): b"\\",
        ord('"'): b'"',
    }
    result = bytearray()
    index = 0

    while index < len(value):
        byte = value[index]
        if byte != ord("\\"):
            result.append(byte)
            index += 1
            continue

        if index + 1 >= len(value):
            raise DiffBuildError(
                "diff_parse_failed",
                "Git patch contained an incomplete quoted file path.",
            )

        escaped = value[index + 1]
        if escaped in escapes:
            result.extend(escapes[escaped])
            index += 2
            continue

        octal = value[index + 1 : index + 4]
        if len(octal) != 3 or any(character not in b"01234567" for character in octal):
            raise DiffBuildError(
                "diff_parse_failed",
                "Git patch contained an invalid quoted file path.",
            )

        result.append(int(octal, 8))
        index += 4

    return bytes(result)


def _header_path(
    lines: list[bytes],
    prefix: bytes,
    required_path_prefix: bytes,
) -> str | None:
    """Read an ordinary Git --- or +++ path header, including /dev/null."""
    matching_lines = [line for line in lines if line.startswith(prefix)]
    if len(matching_lines) != 1:
        raise DiffBuildError(
            "diff_parse_failed",
            "Git patch contained invalid file path headers.",
        )

    value = _decode_patch_path_bytes(matching_lines[0][len(prefix) :])
    if value == b"/dev/null":
        return None
    if not value.startswith(required_path_prefix):
        raise DiffBuildError(
            "diff_parse_failed",
            "Git patch contained an unsupported file path.",
        )

    return _decode_path(value[len(required_path_prefix) :])


def _decode_path(path_bytes: bytes) -> str:
    """Decode one literal repository-relative path without interpreting it."""
    try:
        path = path_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DiffBuildError(
            "diff_parse_failed",
            "Git output contained a non-text file path.",
        ) from error

    if (
        not path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise DiffBuildError(
            "diff_parse_failed",
            "Git output contained an invalid file path.",
        )

    return path


def _parse_hunks(lines: list[bytes]) -> Iterable[DiffHunk]:
    """Read each normal unified-diff hunk range without dropping malformed ones."""
    for line in lines:
        if not line.startswith(b"@@"):
            continue

        match = _HUNK_HEADER.match(line)
        if match is None:
            raise DiffBuildError(
                "diff_parse_failed",
                "Git patch contained an invalid hunk header.",
            )

        yield DiffHunk(
            old_start=int(match.group("old_start")),
            old_count=int(match.group("old_count") or b"1"),
            new_start=int(match.group("new_start")),
            new_count=int(match.group("new_count") or b"1"),
        )


def _file_status(
    old_path: str | None,
    new_path: str | None,
) -> Literal["added", "modified", "deleted", "renamed", "copied", "type_changed"]:
    """Derive the status of a normal non-rename patch block."""
    if old_path is None:
        return "added"
    if new_path is None:
        return "deleted"
    return "modified"
