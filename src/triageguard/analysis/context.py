"""Bounded Java syntax extraction for frozen pull-request evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from triageguard.config import Settings
from triageguard.domain.pr_analysis import (
    ContextAnchor,
    ContextBundle,
    ContextScoreComponent,
    DiffArtifact,
    DiffFile,
    DiffHunk,
    PullRequestSnapshot,
)
from triageguard.provenance import canonical_sha256
from triageguard.sources.git import GitTreeEntry


class ContextBuildError(RuntimeError):
    """A safe, structured reason why frozen code evidence cannot be built."""

    def __init__(self, reason_code: str, safe_message: str) -> None:
        self.reason_code = reason_code
        self.safe_message = safe_message
        super().__init__(f"{reason_code}: {safe_message}")


@dataclass(frozen=True)
class JavaSymbol:
    """One named Java declaration and its exact source-line span."""

    name: str
    kind: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class JavaFileIndex:
    """Security-relevant Java syntax recorded from one immutable source blob."""

    path: str
    symbols: tuple[JavaSymbol, ...]
    package: str | None
    imports: tuple[str, ...]
    annotations: tuple[str, ...]
    classes: tuple[str, ...]
    interfaces: tuple[str, ...]
    enums: tuple[str, ...]
    records: tuple[str, ...]
    constructors: tuple[str, ...]
    methods: tuple[str, ...]
    invocations: tuple[str, ...]


class JavaSyntaxExtractor:
    """Read Java syntax with Tree-sitter instead of guessing from raw text."""

    def __init__(self) -> None:
        language = Language(tree_sitter_java.language())
        self._parser = Parser(language)

    def extract(self, path: str, source: bytes) -> JavaFileIndex:
        """Extract a small, deterministic index from one Java source file."""
        if not path:
            raise ValueError("Java source path must not be empty")

        tree = self._parser.parse(source)
        if tree.root_node.has_error:
            raise ContextBuildError(
                "java_parse_failed",
                "Java source could not be parsed exactly.",
            )

        package: str | None = None
        imports: list[str] = []
        annotations: list[str] = []
        classes: list[str] = []
        interfaces: list[str] = []
        enums: list[str] = []
        records: list[str] = []
        constructors: list[str] = []
        methods: list[str] = []
        invocations: list[str] = []
        symbols: list[JavaSymbol] = []

        for node in _walk_named_nodes(tree.root_node):
            if node.type == "package_declaration":
                package = _declaration_name(source, node, "package")
            elif node.type == "import_declaration":
                imports.append(_declaration_name(source, node, "import"))
            elif node.type == "annotation":
                name = _field_text(source, node, "name")
                if name is not None:
                    annotations.append(name)
            elif node.type == "class_declaration":
                name = _field_text(source, node, "name")
                if name is not None:
                    classes.append(name)
                    symbols.append(_symbol_from_node(source, node, name, "class"))
            elif node.type == "interface_declaration":
                name = _field_text(source, node, "name")
                if name is not None:
                    interfaces.append(name)
                    symbols.append(_symbol_from_node(source, node, name, "interface"))
            elif node.type == "enum_declaration":
                name = _field_text(source, node, "name")
                if name is not None:
                    enums.append(name)
                    symbols.append(_symbol_from_node(source, node, name, "enum"))
            elif node.type == "record_declaration":
                name = _field_text(source, node, "name")
                if name is not None:
                    records.append(name)
                    symbols.append(_symbol_from_node(source, node, name, "record"))
            elif node.type == "constructor_declaration":
                name = _field_text(source, node, "name")
                if name is not None:
                    constructors.append(name)
                    symbols.append(_symbol_from_node(source, node, name, "constructor"))
            elif node.type == "method_declaration":
                name = _field_text(source, node, "name")
                if name is not None:
                    methods.append(name)
                    symbols.append(_symbol_from_node(source, node, name, "method"))
            elif node.type == "method_invocation":
                name = _field_text(source, node, "name")
                if name is not None:
                    invocations.append(name)

        return JavaFileIndex(
            path=path,
            symbols=tuple(symbols),
            package=package,
            imports=tuple(imports),
            annotations=tuple(annotations),
            classes=tuple(classes),
            interfaces=tuple(interfaces),
            enums=tuple(enums),
            records=tuple(records),
            constructors=tuple(constructors),
            methods=tuple(methods),
            invocations=tuple(invocations),
        )


def _walk_named_nodes(node: Node) -> Iterator[Node]:
    """Yield one syntax node and all its named descendants in source order."""
    yield node
    for child in node.named_children:
        yield from _walk_named_nodes(child)


def _field_text(source: bytes, node: Node, field_name: str) -> str | None:
    """Decode one Tree-sitter field from the frozen source bytes."""
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    return source[child.start_byte : child.end_byte].decode("utf-8")


def _declaration_name(source: bytes, node: Node, keyword: str) -> str:
    """Return the name within a Tree-sitter declaration node."""
    text = source[node.start_byte : node.end_byte].decode("utf-8")
    return text.removeprefix(keyword).removesuffix(";").strip()


@dataclass(frozen=True)
class ContextLimits:
    """The exact resource limits used to construct one context bundle."""

    max_files: int
    max_anchors: int
    max_total_bytes: int
    max_anchor_lines: int
    max_blob_bytes: int
    max_search_identifiers: int
    max_hits_per_identifier: int

    def __post_init__(self) -> None:
        for name in (
            "max_files",
            "max_anchors",
            "max_total_bytes",
            "max_anchor_lines",
            "max_blob_bytes",
            "max_search_identifiers",
            "max_hits_per_identifier",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_settings(cls, settings: Settings) -> ContextLimits:
        """Copy the secret-free context bounds from one project setting set."""
        return cls(
            max_files=settings.max_context_files,
            max_anchors=settings.max_context_anchors,
            max_total_bytes=settings.max_context_bytes,
            max_anchor_lines=settings.max_context_anchor_lines,
            max_blob_bytes=settings.max_context_blob_bytes,
            max_search_identifiers=settings.max_context_search_identifiers,
            max_hits_per_identifier=settings.max_context_hits_per_identifier,
        )


class _ContextStore(Protocol):
    """The safe, read-only Git operations needed to build context."""

    def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
        """List entries in one already-frozen commit."""

    def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
        """Read one already-frozen blob under a fixed byte limit."""


class ContextBuilder:
    """Build bounded, content-addressed Java evidence from frozen diffs."""

    def __init__(self) -> None:
        self._extractor = JavaSyntaxExtractor()

    def build(
        self,
        *,
        snapshot: PullRequestSnapshot,
        diffs: Sequence[DiffArtifact],
        store: _ContextStore,
        limits: ContextLimits,
    ) -> ContextBundle:
        """Build ranked Java evidence from all three frozen comparisons."""
        artifacts_by_kind = {artifact.kind: artifact for artifact in diffs}
        expected_kinds = {
            "author_diff",
            "integration_diff",
            "base_drift_diff",
        }
        if len(diffs) != 3 or set(artifacts_by_kind) != expected_kinds:
            raise ContextBuildError(
                "diff_inventory_invalid",
                "Exactly one frozen diff of each required comparison is needed.",
            )

        expected_revisions = {
            "author_diff": (snapshot.merge_base_sha, snapshot.head_sha),
            "integration_diff": (snapshot.base_sha, snapshot.candidate_sha),
            "base_drift_diff": (snapshot.merge_base_sha, snapshot.base_sha),
        }
        for kind, expected_pair in expected_revisions.items():
            artifact = artifacts_by_kind[kind]
            if (artifact.old_revision, artifact.new_revision) != expected_pair:
                raise ContextBuildError(
                    "diff_snapshot_mismatch",
                    "A frozen diff did not match the pull-request snapshot.",
                )

        for artifact in diffs:
            expected_artifact_hash = canonical_sha256(
                artifact.model_dump(
                    mode="json",
                    exclude={"artifact_sha256"},
                )
            )
            if artifact.artifact_sha256 != expected_artifact_hash:
                raise ContextBuildError(
                    "diff_hash_mismatch",
                    "A frozen diff did not match its recorded content hash.",
                )

        anchors: list[ContextAnchor] = []
        excluded_paths: list[str] = []
        binary_paths: list[str] = []
        primary_anchor_count = 0
        excluded_primary_hunks = 0
        primary_change_excluded = False
        primary_indexes: list[JavaFileIndex] = []
        known_anchor_ids: set[str] = set()

        for kind in (
            "integration_diff",
            "author_diff",
            "base_drift_diff",
        ):
            artifact = artifacts_by_kind[kind]
            change_relation, score_name, score_value = _comparison_details(kind)
            is_primary = kind == "integration_diff"

            for diff_file in artifact.files:
                if diff_file.binary:
                    binary_paths.append(_visible_path(diff_file))
                    if is_primary:
                        primary_change_excluded = True
                    continue

                if not diff_file.hunks:
                    continue

                path, commit_sha, revision_role, start_kind = _source_for_diff(
                    snapshot,
                    kind,
                    diff_file,
                )
                if not path.endswith(".java"):
                    excluded_paths.append(path)
                    if is_primary:
                        excluded_primary_hunks += len(diff_file.hunks)
                        primary_change_excluded = True
                    continue

                entry = _tree_entry(store, commit_sha, path)
                if entry is None:
                    if is_primary:
                        raise ContextBuildError(
                            "primary_change_not_represented",
                            "A primary integration source file was unavailable.",
                        )
                    excluded_paths.append(path)
                    continue

                if entry.object_type != "blob" or entry.mode not in {
                    "100644",
                    "100755",
                }:
                    excluded_paths.append(path)
                    if is_primary:
                        excluded_primary_hunks += len(diff_file.hunks)
                        primary_change_excluded = True
                    continue

                source = store.read_blob(
                    entry.object_sha,
                    max_bytes=limits.max_blob_bytes,
                )
                if len(source) > limits.max_blob_bytes:
                    raise ContextBuildError(
                        "blob_limit_exceeded",
                        "A source blob exceeded the configured byte limit.",
                    )

                index = self._extractor.extract(path, source)
                if is_primary:
                    primary_indexes.append(index)

                for hunk in diff_file.hunks:
                    start_line, line_count = _hunk_range(hunk, start_kind)
                    anchor = _anchor_for_hunk(
                        snapshot=snapshot,
                        path=path,
                        commit_sha=commit_sha,
                        blob_sha=entry.object_sha,
                        revision_role=revision_role,
                        source=source,
                        start_line=start_line,
                        line_count=line_count,
                        limits=limits,
                        java_symbol=_symbol_for_range(
                            index,
                            start_line,
                            start_line + line_count - 1,
                        ),
                        change_relation=change_relation,
                        selection_reason=score_name.replace("_", " "),
                        score_components=(
                            ContextScoreComponent(
                                name=score_name,
                                value=score_value,
                            ),
                        ),
                    )

                    if is_primary:
                        primary_anchor_count += 1
                        if anchor.anchor_id in known_anchor_ids:
                            continue
                        _ensure_primary_budget(anchors, anchor, limits)
                        anchors.append(anchor)
                        known_anchor_ids.add(anchor.anchor_id)
                    elif (
                        anchor.anchor_id not in known_anchor_ids
                        and _fits_context_budget(anchors, anchor, limits)
                    ):
                        anchors.append(anchor)
                        known_anchor_ids.add(anchor.anchor_id)

        integration = artifacts_by_kind["integration_diff"]
        required_primary_hunks = sum(
            len(diff_file.hunks)
            for diff_file in integration.files
            if not diff_file.binary
        )
        if primary_anchor_count + excluded_primary_hunks != required_primary_hunks:
            raise ContextBuildError(
                "primary_change_not_represented",
                "Every non-binary primary integration hunk requires evidence.",
            )

        known_anchor_ids.update(anchor.anchor_id for anchor in anchors)
        for candidate in _repository_context_anchors(
            snapshot=snapshot,
            store=store,
            limits=limits,
            primary_indexes=primary_indexes,
            selected_paths={anchor.path for anchor in anchors},
            extractor=self._extractor,
        ):
            if candidate.anchor_id in known_anchor_ids:
                continue
            if _fits_context_budget(anchors, candidate, limits):
                anchors.append(candidate)
                known_anchor_ids.add(candidate.anchor_id)

        anchors.sort(
            key=lambda anchor: (
                -sum(component.value for component in anchor.score_components),
                anchor.revision_role,
                anchor.path,
                anchor.start_line,
                anchor.anchor_id,
            )
        )
        selected_bytes = sum(len(anchor.text.encode("utf-8")) for anchor in anchors)

        return ContextBundle.from_content(
            snapshot_key=snapshot.snapshot_key,
            anchors=anchors,
            selected_file_count=len({anchor.path for anchor in anchors}),
            selected_anchor_count=len(anchors),
            selected_bytes=selected_bytes,
            max_files=limits.max_files,
            max_anchors=limits.max_anchors,
            max_bytes=limits.max_total_bytes,
            max_anchor_lines=limits.max_anchor_lines,
            max_blob_bytes=limits.max_blob_bytes,
            max_search_identifiers=limits.max_search_identifiers,
            max_hits_per_identifier=limits.max_hits_per_identifier,
            excluded_paths=tuple(dict.fromkeys(excluded_paths)),
            binary_paths=tuple(dict.fromkeys(binary_paths)),
            truncated_anchor_ids=tuple(
                anchor.anchor_id for anchor in anchors if anchor.truncated
            ),
            primary_change_represented=(
                not primary_change_excluded
                and primary_anchor_count == required_primary_hunks
            ),
        )


def _comparison_details(kind: str) -> tuple[str, str, int]:
    """Return the fixed relation and deterministic score for one comparison."""
    details = {
        "integration_diff": ("integration_change", "integration_hunk", 100),
        "author_diff": ("author_change", "author_hunk", 60),
        "base_drift_diff": ("base_drift_change", "base_drift_hunk", 40),
    }
    try:
        return details[kind]
    except KeyError as error:
        raise ContextBuildError(
            "diff_inventory_invalid",
            "A frozen diff had an unsupported comparison kind.",
        ) from error


def _visible_path(diff_file: DiffFile) -> str:
    """Return the available path for one changed file."""
    path = diff_file.new_path or diff_file.old_path
    if path is None:
        raise ContextBuildError(
            "primary_change_not_represented",
            "A changed file did not retain a usable path.",
        )
    return path


def _source_for_diff(
    snapshot: PullRequestSnapshot,
    kind: str,
    diff_file: DiffFile,
) -> tuple[str, str, str, str]:
    """Choose the frozen revision that contains one changed file side."""
    if kind == "author_diff":
        new_commit, new_role = snapshot.head_sha, "head"
        old_commit, old_role = snapshot.merge_base_sha, "merge_base"
    elif kind == "integration_diff":
        new_commit, new_role = snapshot.candidate_sha, "candidate"
        old_commit, old_role = snapshot.base_sha, "base"
    elif kind == "base_drift_diff":
        new_commit, new_role = snapshot.base_sha, "base"
        old_commit, old_role = snapshot.merge_base_sha, "merge_base"
    else:
        raise ContextBuildError(
            "diff_inventory_invalid",
            "A frozen diff had an unsupported comparison kind.",
        )

    if diff_file.new_path is not None:
        return diff_file.new_path, new_commit, new_role, "new"
    if diff_file.old_path is not None:
        return diff_file.old_path, old_commit, old_role, "old"

    raise ContextBuildError(
        "primary_change_not_represented",
        "A changed file did not retain a usable path.",
    )


def _tree_entry(
    store: _ContextStore,
    commit_sha: str,
    path: str,
) -> GitTreeEntry | None:
    """Find one literal path in one frozen commit tree."""
    return next(
        (entry for entry in store.list_tree(commit_sha) if entry.path == path),
        None,
    )


def _hunk_range(hunk: DiffHunk, start_kind: str) -> tuple[int, int]:
    """Choose the side of a diff that exists in the frozen source revision."""
    if start_kind == "new" and hunk.new_count > 0:
        return hunk.new_start, hunk.new_count
    if start_kind == "old" and hunk.old_count > 0:
        return hunk.old_start, hunk.old_count

    raise ContextBuildError(
        "primary_change_not_represented",
        "A changed hunk had no readable source lines.",
    )


def _anchor_for_hunk(
    *,
    snapshot: PullRequestSnapshot,
    path: str,
    commit_sha: str,
    blob_sha: str,
    revision_role: str,
    source: bytes,
    start_line: int,
    line_count: int,
    limits: ContextLimits,
    java_symbol: str | None,
    change_relation: str,
    selection_reason: str,
    score_components: tuple[ContextScoreComponent, ...],
) -> ContextAnchor:
    """Create one exact, bounded evidence anchor from source lines."""
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextBuildError(
            "source_decode_failed",
            "A Java source blob was not valid UTF-8.",
        ) from error

    source_lines = source_text.splitlines(keepends=True)
    if start_line <= 0 or start_line > len(source_lines):
        raise ContextBuildError(
            "primary_change_not_represented",
            "A changed hunk was outside its source file.",
        )

    available_count = len(source_lines) - start_line + 1
    selected_count = min(line_count, limits.max_anchor_lines)
    if selected_count > available_count:
        raise ContextBuildError(
            "primary_change_not_represented",
            "A changed hunk extended beyond its source file.",
        )

    text = "".join(source_lines[start_line - 1 : start_line - 1 + selected_count])
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = {
        "snapshot_key": snapshot.snapshot_key,
        "commit_sha": commit_sha,
        "blob_sha": blob_sha,
        "path": path,
        "start_line": start_line,
        "end_line": start_line + selected_count - 1,
        "text_sha256": text_sha256,
        "change_relation": change_relation,
    }

    return ContextAnchor(
        anchor_id=f"anchor-{canonical_sha256(identity)[:16]}",
        revision_role=revision_role,
        commit_sha=commit_sha,
        blob_sha=blob_sha,
        path=path,
        java_symbol=java_symbol,
        start_line=start_line,
        end_line=start_line + selected_count - 1,
        text=text,
        text_sha256=text_sha256,
        selection_reason=selection_reason,
        score_components=score_components,
        change_relation=change_relation,
        truncated=selected_count < line_count,
    )


def _fits_context_budget(
    anchors: Sequence[ContextAnchor],
    candidate: ContextAnchor,
    limits: ContextLimits,
) -> bool:
    """Return whether one optional context anchor still fits all limits."""
    next_paths = {anchor.path for anchor in anchors} | {candidate.path}
    next_bytes = sum(len(anchor.text.encode("utf-8")) for anchor in anchors)
    next_bytes += len(candidate.text.encode("utf-8"))

    return (
        len(next_paths) <= limits.max_files
        and len(anchors) + 1 <= limits.max_anchors
        and next_bytes <= limits.max_total_bytes
    )


def _ensure_primary_budget(
    anchors: Sequence[ContextAnchor],
    candidate: ContextAnchor,
    limits: ContextLimits,
) -> None:
    """Reject rather than partially represent primary integration evidence."""
    if not _fits_context_budget(anchors, candidate, limits):
        raise ContextBuildError(
            "primary_change_not_represented",
            "Configured context limits cannot represent every primary change.",
        )


def _symbol_from_node(
    source: bytes,
    node: Node,
    name: str,
    kind: str,
) -> JavaSymbol:
    """Record one declaration span from safe Tree-sitter byte offsets.

    The macOS Tree-sitter Java binding can crash Python while resolving a
    node's ``start_point`` or ``end_point``. Byte offsets describe the same
    source positions without crossing that unsafe native-code boundary.
    """
    start_line = source.count(b"\n", 0, node.start_byte) + 1
    end_line = max(start_line, source.count(b"\n", 0, node.end_byte) + 1)
    return JavaSymbol(
        name=name,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
    )


def _symbol_for_range(
    index: JavaFileIndex,
    start_line: int,
    end_line: int,
) -> str | None:
    """Return the smallest declaration that contains an evidence excerpt."""
    containing = [
        symbol
        for symbol in index.symbols
        if symbol.start_line <= start_line and end_line <= symbol.end_line
    ]
    if not containing:
        return None

    return min(
        containing,
        key=lambda symbol: (
            symbol.end_line - symbol.start_line,
            symbol.start_line,
            symbol.name,
        ),
    ).name


def _repository_context_anchors(
    *,
    snapshot: PullRequestSnapshot,
    store: _ContextStore,
    limits: ContextLimits,
    primary_indexes: Sequence[JavaFileIndex],
    selected_paths: set[str],
    extractor: JavaSyntaxExtractor,
) -> tuple[ContextAnchor, ...]:
    """Find bounded candidate-repository context using exact Java identifiers."""
    identifiers = _search_identifiers(
        primary_indexes,
        max_identifiers=limits.max_search_identifiers,
    )
    if not identifiers:
        return ()

    primary_packages = {
        index.package for index in primary_indexes if index.package is not None
    }
    identifier_hits = {identifier: 0 for identifier in identifiers}
    candidates: list[ContextAnchor] = []

    for entry in sorted(
        store.list_tree(snapshot.candidate_sha), key=lambda item: item.path
    ):
        if (
            entry.path in selected_paths
            or not entry.path.endswith(".java")
            or entry.object_type != "blob"
            or entry.mode not in {"100644", "100755"}
        ):
            continue

        source = store.read_blob(
            entry.object_sha,
            max_bytes=limits.max_blob_bytes,
        )
        if len(source) > limits.max_blob_bytes:
            raise ContextBuildError(
                "blob_limit_exceeded",
                "A source blob exceeded the configured byte limit.",
            )

        if not any(identifier.encode("utf-8") in source for identifier in identifiers):
            continue

        try:
            index = extractor.extract(entry.path, source)
        except ContextBuildError as error:
            if error.reason_code == "java_parse_failed":
                continue
            raise
        shared_identifiers = [
            identifier
            for identifier in identifiers
            if identifier in _index_identifiers(index)
            and identifier_hits[identifier] < limits.max_hits_per_identifier
        ]
        if not shared_identifiers:
            continue

        symbol = _matching_symbol(index, shared_identifiers)
        if symbol is None:
            continue

        score_components = [
            ContextScoreComponent(name="same_symbol", value=30),
        ]
        if _has_security_signal(index):
            score_components.append(
                ContextScoreComponent(name="security_signal", value=20),
            )
        if _has_import_or_invocation_match(index, identifiers):
            score_components.append(
                ContextScoreComponent(name="import_or_invocation", value=15),
            )
        if index.package in primary_packages:
            score_components.append(
                ContextScoreComponent(name="same_package", value=10),
            )
        if entry.path.endswith("Test.java"):
            score_components.append(
                ContextScoreComponent(name="related_test", value=8),
            )

        candidates.append(
            _anchor_for_hunk(
                snapshot=snapshot,
                path=entry.path,
                commit_sha=snapshot.candidate_sha,
                blob_sha=entry.object_sha,
                revision_role="candidate",
                source=source,
                start_line=symbol.start_line,
                line_count=symbol.end_line - symbol.start_line + 1,
                limits=limits,
                java_symbol=symbol.name,
                change_relation="repository_context",
                selection_reason="repository context",
                score_components=tuple(score_components),
            )
        )
        for identifier in shared_identifiers:
            identifier_hits[identifier] += 1

    return tuple(candidates)


def _search_identifiers(
    indexes: Sequence[JavaFileIndex],
    *,
    max_identifiers: int,
) -> tuple[str, ...]:
    """Return a stable, bounded list of exact Java identifiers to search."""
    identifiers: list[str] = []

    for index in indexes:
        for identifier in _index_identifiers(index):
            if identifier not in identifiers:
                identifiers.append(identifier)
            if len(identifiers) == max_identifiers:
                return tuple(identifiers)

    return tuple(identifiers)


def _index_identifiers(index: JavaFileIndex) -> tuple[str, ...]:
    """Return searchable exact identifiers from a parsed Java file."""
    imported_names = tuple(
        imported_name.rsplit(".", maxsplit=1)[-1] for imported_name in index.imports
    )
    return (
        index.classes
        + index.interfaces
        + index.enums
        + index.records
        + index.constructors
        + index.methods
        + index.invocations
        + imported_names
    )


def _matching_symbol(
    index: JavaFileIndex,
    identifiers: Sequence[str],
) -> JavaSymbol | None:
    """Choose the earliest smallest declaration matching an exact identifier."""
    matches = [symbol for symbol in index.symbols if symbol.name in set(identifiers)]
    if not matches:
        return None

    return min(
        matches,
        key=lambda symbol: (
            symbol.end_line - symbol.start_line,
            symbol.start_line,
            symbol.name,
        ),
    )


def _has_security_signal(index: JavaFileIndex) -> bool:
    """Identify a small, fixed set of security-relevant Java names."""
    security_terms = (
        "access",
        "auth",
        "authorize",
        "permission",
        "privilege",
        "security",
    )
    names = index.annotations + _index_identifiers(index)
    return any(term in name.casefold() for name in names for term in security_terms)


def _has_import_or_invocation_match(
    index: JavaFileIndex,
    identifiers: Sequence[str],
) -> bool:
    """Detect one exact searched identifier used as an import or invocation."""
    import_names = {
        imported_name.rsplit(".", maxsplit=1)[-1] for imported_name in index.imports
    }
    return bool((import_names | set(index.invocations)) & set(identifiers))
