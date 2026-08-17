"""Bounded catalog-first retrieval from one immutable M/B/H/C snapshot."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from triageguard.analysis.context import ContextLimits
from triageguard.domain import (
    ContextAnchor,
    ContextBundle,
    EvidenceRefinementResult,
    FrozenEvidenceNeed,
    PullRequestSnapshot,
)
from triageguard.provenance import canonical_sha256


class _ContextRefiner(Protocol):
    def refine(
        self,
        *,
        snapshot: PullRequestSnapshot,
        context: ContextBundle,
        needs: Sequence[FrozenEvidenceNeed],
        store: object,
        limits: ContextLimits,
        created_at: datetime,
    ) -> tuple[ContextBundle, object]: ...


@dataclass(frozen=True)
class FrozenEvidenceResolution:
    """One resolver outcome plus the context available to the next model call."""

    context: ContextBundle
    refinement: EvidenceRefinementResult

    @property
    def priority_anchor_ids(self) -> tuple[str, ...]:
        return self.refinement.priority_anchor_ids

    @property
    def added_anchor_ids(self) -> tuple[str, ...]:
        return self.refinement.added_anchor_ids

    @property
    def round_number(self) -> int:
        return self.refinement.round_number

    @property
    def exhausted(self) -> bool:
        return self.refinement.exhausted

    @property
    def reason_code(self) -> str:
        return self.refinement.reason_code


class FrozenEvidenceResolver:
    """Resolve precise needs from the catalog before opening frozen Git blobs."""

    def __init__(self, context_refiner: _ContextRefiner) -> None:
        self._context_refiner = context_refiner

    def resolve(
        self,
        *,
        snapshot: PullRequestSnapshot,
        context: ContextBundle,
        needs: Sequence[FrozenEvidenceNeed],
        store: object,
        limits: ContextLimits,
        completed_rounds: int,
        max_rounds: int,
        created_at: datetime,
    ) -> FrozenEvidenceResolution:
        normalized = _validate_resolution_inputs(
            snapshot=snapshot,
            context=context,
            needs=needs,
            limits=limits,
            completed_rounds=completed_rounds,
            max_rounds=max_rounds,
        )
        requested_need_sha256 = canonical_sha256(
            [need.model_dump(mode="json") for need in normalized]
        )
        round_number = completed_rounds + 1
        if completed_rounds >= max_rounds:
            return _resolution(
                context=context,
                requested_need_sha256=requested_need_sha256,
                priority_anchor_ids=(),
                added_anchor_ids=(),
                round_number=round_number,
                exhausted=True,
                reason_code="refinement_round_limit_reached",
            )

        priority_anchor_ids: list[str] = []
        unresolved: list[FrozenEvidenceNeed] = []
        for need in normalized:
            supporting = set(need.supporting_anchor_ids)
            match = next(
                (
                    anchor.anchor_id
                    for anchor in context.anchors
                    if anchor.anchor_id not in supporting
                    and any(
                        _anchor_has_exact_term(anchor, term)
                        for term in need.search_terms
                    )
                ),
                None,
            )
            if match is None:
                unresolved.append(need)
            elif match not in priority_anchor_ids:
                priority_anchor_ids.append(match)

        refined_context = context
        added_anchor_ids: tuple[str, ...] = ()
        if unresolved:
            refined_context, legacy_refinement = self._context_refiner.refine(
                snapshot=snapshot,
                context=context,
                needs=tuple(unresolved),
                store=store,
                limits=limits,
                created_at=created_at,
            )
            added_anchor_ids = tuple(legacy_refinement.added_anchor_ids)

        if added_anchor_ids:
            return _resolution(
                context=refined_context,
                parent_context=context,
                requested_need_sha256=requested_need_sha256,
                priority_anchor_ids=tuple(priority_anchor_ids),
                added_anchor_ids=added_anchor_ids,
                round_number=round_number,
                exhausted=False,
                reason_code="frozen_context_extended",
            )
        if priority_anchor_ids:
            return _resolution(
                context=context,
                requested_need_sha256=requested_need_sha256,
                priority_anchor_ids=tuple(priority_anchor_ids),
                added_anchor_ids=(),
                round_number=round_number,
                exhausted=False,
                reason_code="catalog_evidence_prioritized",
            )
        return _resolution(
            context=context,
            requested_need_sha256=requested_need_sha256,
            priority_anchor_ids=(),
            added_anchor_ids=(),
            round_number=round_number,
            exhausted=True,
            reason_code="frozen_evidence_exhausted",
        )


def _validate_resolution_inputs(
    *,
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    needs: Sequence[FrozenEvidenceNeed],
    limits: ContextLimits,
    completed_rounds: int,
    max_rounds: int,
) -> tuple[FrozenEvidenceNeed, ...]:
    if context.snapshot_key != snapshot.snapshot_key:
        raise ValueError("refinement context must match the frozen snapshot")
    if type(completed_rounds) is not int or completed_rounds < 0:
        raise ValueError("completed refinement rounds must be a non-negative integer")
    if type(max_rounds) is not int or max_rounds <= 0:
        raise ValueError("maximum refinement rounds must be a positive integer")
    normalized = tuple(FrozenEvidenceNeed.model_validate(need) for need in needs)
    if not normalized:
        raise ValueError("refinement requires at least one structured evidence need")
    if len({need.need_id for need in normalized}) != len(normalized):
        raise ValueError("refinement evidence need IDs must be unique")
    if (
        sum(len(need.search_terms) for need in normalized)
        > limits.max_search_identifiers
    ):
        raise ValueError("refinement exceeds the frozen search identifier limit")
    catalog = {anchor.anchor_id for anchor in context.anchors}
    if any(
        anchor_id not in catalog
        for need in normalized
        for anchor_id in need.supporting_anchor_ids
    ):
        raise ValueError("refinement needs must cite the current frozen catalog")
    return normalized


def _anchor_has_exact_term(anchor: ContextAnchor, term: str) -> bool:
    searchable = "\n".join(
        value
        for value in (anchor.path, anchor.java_symbol, anchor.text)
        if value is not None
    )
    pattern = rf"(?<![A-Za-z0-9_$]){re.escape(term)}(?![A-Za-z0-9_$])"
    return re.search(pattern, searchable) is not None


def _resolution(
    *,
    context: ContextBundle,
    requested_need_sha256: str,
    priority_anchor_ids: tuple[str, ...],
    added_anchor_ids: tuple[str, ...],
    round_number: int,
    exhausted: bool,
    reason_code: str,
    parent_context: ContextBundle | None = None,
) -> FrozenEvidenceResolution:
    parent = parent_context or context
    refinement = EvidenceRefinementResult.from_content(
        parent_context_sha256=parent.context_sha256,
        successor_context_sha256=context.context_sha256,
        requested_need_sha256=requested_need_sha256,
        priority_anchor_ids=priority_anchor_ids,
        added_anchor_ids=added_anchor_ids,
        round_number=round_number,
        exhausted=exhausted,
        reason_code=reason_code,
    )
    return FrozenEvidenceResolution(context=context, refinement=refinement)
