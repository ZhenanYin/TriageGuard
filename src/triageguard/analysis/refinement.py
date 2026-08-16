"""Deterministically add only already-frozen Java evidence to a context."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from triageguard.analysis.context import (
    ContextBuildError,
    ContextLimits,
    JavaFileIndex,
    JavaSyntaxExtractor,
)
from triageguard.domain import (
    ContextAnchor,
    ContextBundle,
    ContextRefinement,
    ContextScoreComponent,
    PullRequestSnapshot,
    TestabilityAssessment,
)
from triageguard.provenance import canonical_sha256
from triageguard.sources.git import GitTreeEntry


class FrozenEvidenceRefinementError(RuntimeError):
    """The saved evidence could not be safely refined."""


class _FrozenStore(Protocol):
    """Read only Git objects that were already frozen for this snapshot."""

    def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
        """List literal paths from one frozen commit."""

    def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
        """Read one bounded frozen blob."""


class FrozenContextRefiner:
    """Find additional Java anchors without acquiring any new repository state."""

    def __init__(self) -> None:
        self._extractor = JavaSyntaxExtractor()

    def refine(
        self,
        *,
        snapshot: PullRequestSnapshot,
        context: ContextBundle,
        assessment: TestabilityAssessment,
        store: _FrozenStore,
        limits: ContextLimits,
        created_at: datetime,
    ) -> tuple[ContextBundle, ContextRefinement]:
        """Return a successor context or an immutable exhausted-refinement record."""
        _validate_inputs(
            snapshot=snapshot,
            context=context,
            assessment=assessment,
            limits=limits,
            created_at=created_at,
        )

        anchors = list(context.anchors)
        added: list[ContextAnchor] = []
        unresolved_needs = list(assessment.evidence_needs)

        for revision_role, commit_sha in _frozen_revisions(snapshot):
            if not unresolved_needs:
                break

            entries = tuple(
                entry
                for entry in sorted(
                    store.list_tree(commit_sha), key=lambda item: item.path
                )
                if entry.object_type == "blob"
                and entry.mode in {"100644", "100755"}
                and entry.path.endswith(".java")
            )

            for need in tuple(unresolved_needs):
                candidate = _find_anchor_for_need(
                    snapshot=snapshot,
                    revision_role=revision_role,
                    commit_sha=commit_sha,
                    entries=entries,
                    need=need,
                    existing_anchors=tuple(anchors),
                    store=store,
                    limits=limits,
                    extractor=self._extractor,
                )
                if candidate is None:
                    continue
                if not _fits_limits(anchors, candidate, limits):
                    continue

                anchors.append(candidate)
                added.append(candidate)
                unresolved_needs.remove(need)

        if not added:
            refinement = ContextRefinement.from_content(
                snapshot_key=snapshot.snapshot_key,
                parent_context_sha256=context.context_sha256,
                refined_context_sha256=context.context_sha256,
                evidence_need_ids=tuple(
                    need.need_id for need in assessment.evidence_needs
                ),
                added_anchor_ids=(),
                exhausted=True,
                created_at=created_at,
            )
            return context, refinement

        refined_context = ContextBundle.from_content(
            snapshot_key=context.snapshot_key,
            anchors=tuple(anchors),
            selected_file_count=len({anchor.path for anchor in anchors}),
            selected_anchor_count=len(anchors),
            selected_bytes=sum(len(anchor.text.encode("utf-8")) for anchor in anchors),
            max_files=context.max_files,
            max_anchors=context.max_anchors,
            max_bytes=context.max_bytes,
            max_anchor_lines=context.max_anchor_lines,
            max_blob_bytes=context.max_blob_bytes,
            max_search_identifiers=context.max_search_identifiers,
            max_hits_per_identifier=context.max_hits_per_identifier,
            excluded_paths=context.excluded_paths,
            binary_paths=context.binary_paths,
            truncated_anchor_ids=tuple(
                anchor.anchor_id for anchor in anchors if anchor.truncated
            ),
            primary_change_represented=context.primary_change_represented,
        )
        refinement = ContextRefinement.from_content(
            snapshot_key=snapshot.snapshot_key,
            parent_context_sha256=context.context_sha256,
            refined_context_sha256=refined_context.context_sha256,
            evidence_need_ids=tuple(need.need_id for need in assessment.evidence_needs),
            added_anchor_ids=tuple(anchor.anchor_id for anchor in added),
            exhausted=False,
            created_at=created_at,
        )
        return refined_context, refinement


def _validate_inputs(
    *,
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    assessment: TestabilityAssessment,
    limits: ContextLimits,
    created_at: datetime,
) -> None:
    """Reject evidence or limits that do not belong to this exact frozen context."""
    if context.snapshot_key != snapshot.snapshot_key:
        raise FrozenEvidenceRefinementError(
            "context snapshot key must match the frozen snapshot"
        )
    if assessment.snapshot_key != snapshot.snapshot_key:
        raise FrozenEvidenceRefinementError(
            "testability snapshot key must match the frozen snapshot"
        )
    if assessment.context_sha256 != context.context_sha256:
        raise FrozenEvidenceRefinementError(
            "testability context hash must match the current frozen context"
        )
    if assessment.decision != "needs_more_frozen_evidence":
        raise FrozenEvidenceRefinementError(
            "only a needs-more-frozen-evidence assessment may request refinement"
        )
    if not _is_utc(created_at):
        raise FrozenEvidenceRefinementError(
            "context refinement time must be timezone-aware UTC"
        )

    context_limits = (
        context.max_files,
        context.max_anchors,
        context.max_bytes,
        context.max_anchor_lines,
        context.max_blob_bytes,
        context.max_search_identifiers,
        context.max_hits_per_identifier,
    )
    supplied_limits = (
        limits.max_files,
        limits.max_anchors,
        limits.max_total_bytes,
        limits.max_anchor_lines,
        limits.max_blob_bytes,
        limits.max_search_identifiers,
        limits.max_hits_per_identifier,
    )
    if context_limits != supplied_limits:
        raise FrozenEvidenceRefinementError(
            "refinement limits must exactly match the original context limits"
        )


def _frozen_revisions(
    snapshot: PullRequestSnapshot,
) -> tuple[tuple[str, str], ...]:
    """Search a deterministic order of the four already-saved revisions."""
    return (
        ("candidate", snapshot.candidate_sha),
        ("head", snapshot.head_sha),
        ("base", snapshot.base_sha),
        ("merge_base", snapshot.merge_base_sha),
    )


def _find_anchor_for_need(
    *,
    snapshot: PullRequestSnapshot,
    revision_role: str,
    commit_sha: str,
    entries: Sequence[GitTreeEntry],
    need: object,
    existing_anchors: Sequence[ContextAnchor],
    store: _FrozenStore,
    limits: ContextLimits,
    extractor: JavaSyntaxExtractor,
) -> ContextAnchor | None:
    """Return the first deterministic new Java anchor satisfying one evidence need."""
    search_terms = need.search_terms
    category = need.category

    for entry in entries:
        source = store.read_blob(entry.object_sha, max_bytes=limits.max_blob_bytes)
        if len(source) > limits.max_blob_bytes:
            raise FrozenEvidenceRefinementError(
                "a frozen Java blob exceeded the approved byte limit"
            )

        try:
            source_text = source.decode("utf-8")
            index = extractor.extract(entry.path, source)
        except (ContextBuildError, UnicodeDecodeError):
            continue

        for term in search_terms:
            if term not in _indexed_identifiers(index):
                continue

            line_number = _code_line_with_term(source_text, term)
            if line_number is None or _line_is_already_represented(
                existing_anchors,
                commit_sha=commit_sha,
                path=entry.path,
                line_number=line_number,
            ):
                continue

            return _anchor_for_term(
                snapshot=snapshot,
                revision_role=revision_role,
                commit_sha=commit_sha,
                blob_sha=entry.object_sha,
                path=entry.path,
                source_text=source_text,
                line_number=line_number,
                term=term,
                category=category,
                limits=limits,
                index=index,
            )

    return None


def _indexed_identifiers(index: JavaFileIndex) -> set[str]:
    """Return syntax-derived names; comment-only text never becomes evidence."""
    return {
        *index.imports,
        *index.annotations,
        *index.classes,
        *index.interfaces,
        *index.enums,
        *index.records,
        *index.constructors,
        *index.methods,
        *index.invocations,
    }


def _code_line_with_term(source_text: str, term: str) -> int | None:
    """Find a source line containing an already syntax-confirmed identifier."""
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        code_before_comment = line.split("//", maxsplit=1)[0]
        if term in code_before_comment:
            return line_number
    return None


def _line_is_already_represented(
    anchors: Sequence[ContextAnchor],
    *,
    commit_sha: str,
    path: str,
    line_number: int,
) -> bool:
    """Avoid adding a second anchor that covers the same frozen source line."""
    return any(
        anchor.commit_sha == commit_sha
        and anchor.path == path
        and anchor.start_line <= line_number <= anchor.end_line
        for anchor in anchors
    )


def _anchor_for_term(
    *,
    snapshot: PullRequestSnapshot,
    revision_role: str,
    commit_sha: str,
    blob_sha: str,
    path: str,
    source_text: str,
    line_number: int,
    term: str,
    category: str,
    limits: ContextLimits,
    index: JavaFileIndex,
) -> ContextAnchor:
    """Create one bounded repository-context anchor around a code identifier."""
    lines = source_text.splitlines(keepends=True)
    start_line = max(1, line_number - 1)
    end_line = min(len(lines), start_line + limits.max_anchor_lines - 1)
    text = "".join(lines[start_line - 1 : end_line])
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = {
        "snapshot_key": snapshot.snapshot_key,
        "commit_sha": commit_sha,
        "blob_sha": blob_sha,
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "text_sha256": text_sha256,
        "change_relation": "repository_context",
    }
    java_symbol = next(
        (
            symbol.name
            for symbol in index.symbols
            if symbol.start_line <= line_number <= symbol.end_line
        ),
        None,
    )

    return ContextAnchor(
        anchor_id=f"anchor-{canonical_sha256(identity)[:16]}",
        revision_role=revision_role,
        commit_sha=commit_sha,
        blob_sha=blob_sha,
        path=path,
        java_symbol=java_symbol,
        start_line=start_line,
        end_line=end_line,
        text=text,
        text_sha256=text_sha256,
        selection_reason=f"testability evidence need: {category}",
        score_components=(
            ContextScoreComponent(
                name="testability_evidence_need",
                value=80.0,
            ),
        ),
        change_relation="repository_context",
        truncated=False,
    )


def _fits_limits(
    anchors: Sequence[ContextAnchor],
    candidate: ContextAnchor,
    limits: ContextLimits,
) -> bool:
    """Return whether one additional frozen anchor fits the original limits."""
    paths = {anchor.path for anchor in anchors} | {candidate.path}
    selected_bytes = sum(len(anchor.text.encode("utf-8")) for anchor in anchors)
    selected_bytes += len(candidate.text.encode("utf-8"))

    return (
        len(paths) <= limits.max_files
        and len(anchors) + 1 <= limits.max_anchors
        and selected_bytes <= limits.max_total_bytes
    )


def _is_utc(value: datetime) -> bool:
    """Return whether a timestamp is timezone-aware UTC."""
    offset = value.utcoffset()
    return (
        value.tzinfo is not None and offset is not None and offset.total_seconds() == 0
    )
