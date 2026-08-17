"""Deterministic whole-anchor selection under an exact provider-body budget."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from triageguard.domain.pr_analysis import ContextAnchor, ContextBundle
from triageguard.evidence.model_envelope import (
    EvidenceArtifactBinding,
    ModelEvidenceEnvelope,
    ModelEvidenceStage,
    OmittedEvidenceAnchor,
    VisibleEvidenceAnchor,
)
from triageguard.llm.gateway import ModelRequest
from triageguard.llm.request_budget import (
    ProviderRequestBudget,
    groq_request_body_bytes,
)
from triageguard.provenance import canonical_sha256

_RELATION_RANK = {
    "integration_change": 0,
    "author_change": 1,
    "repository_context": 2,
    "base_drift_change": 3,
}
_POLICY_VERSION = {
    "risk_hypothesis": "risk-evidence-v1",
    "testability_assessment": "testability-evidence-v1",
    "gherkin_generation": "gherkin-evidence-v1",
}


class ModelEvidenceBudgetError(RuntimeError):
    """The exact request cannot include every mandatory frozen anchor."""


@dataclass(frozen=True)
class EnvelopeBuildResult:
    """One immutable evidence envelope and the exact request measured from it."""

    envelope: ModelEvidenceEnvelope
    request: ModelRequest
    request_body_bytes: int


class EvidenceEnvelopeBuilder:
    """Select complete context anchors using one stable stage-aware ranking."""

    def build(
        self,
        *,
        stage: ModelEvidenceStage,
        context: ContextBundle,
        comparison_bindings: tuple[EvidenceArtifactBinding, ...],
        input_bindings: tuple[EvidenceArtifactBinding, ...],
        required_anchor_ids: tuple[str, ...],
        priority_terms: tuple[str, ...],
        budget: ProviderRequestBudget,
        request_factory: Callable[[ModelEvidenceEnvelope], ModelRequest],
        priority_anchor_ids: tuple[str, ...] = (),
    ) -> EnvelopeBuildResult:
        """Build one exact request without slicing or mutating frozen context."""
        catalog = {anchor.anchor_id: anchor for anchor in context.anchors}
        if len(required_anchor_ids) != len(set(required_anchor_ids)):
            raise ValueError("required anchor IDs must be unique")
        unknown_required = set(required_anchor_ids) - set(catalog)
        if unknown_required:
            unknown = ", ".join(sorted(unknown_required))
            raise ValueError(f"required anchor IDs are absent from context: {unknown}")
        if len(priority_anchor_ids) != len(set(priority_anchor_ids)):
            raise ValueError("priority anchor IDs must be unique")
        unknown_priority = set(priority_anchor_ids) - set(catalog)
        if unknown_priority:
            unknown = ", ".join(sorted(unknown_priority))
            raise ValueError(f"priority anchor IDs are absent from context: {unknown}")
        if any(not isinstance(term, str) or not term for term in priority_terms):
            raise ValueError("priority terms must be non-empty strings")
        if len(priority_terms) != len(set(priority_terms)):
            raise ValueError("priority terms must be unique")

        required = {*required_anchor_ids, *priority_anchor_ids}
        if stage == "risk_hypothesis":
            # ContextBuilder can create base_drift_change anchors only from changed
            # diff hunks; an unchanged canonical drift comparison has no such anchor.
            for relation in ("integration_change", "author_change"):
                relation_anchors = [
                    anchor
                    for anchor in context.anchors
                    if anchor.change_relation == relation
                ]
                if relation_anchors:
                    required.add(
                        min(
                            relation_anchors,
                            key=lambda anchor: self._rank_anchor(
                                anchor,
                                required=set(),
                                priority_terms=(),
                            ),
                        ).anchor_id
                    )

        ranked = sorted(
            context.anchors,
            key=lambda anchor: self._rank_anchor(
                anchor,
                required=required,
                priority_terms=priority_terms,
            ),
        )
        visible: list[ContextAnchor] = []
        for anchor in ranked:
            candidate_visible = (*visible, anchor)
            candidate = self._materialize(
                stage=stage,
                context=context,
                comparison_bindings=comparison_bindings,
                input_bindings=input_bindings,
                visible_anchors=candidate_visible,
                required=required,
                priority_terms=priority_terms,
                budget=budget,
                request_factory=request_factory,
            )
            if candidate.request_body_bytes <= budget.max_body_bytes:
                visible.append(anchor)
                continue
            if anchor.anchor_id in required:
                raise ModelEvidenceBudgetError(
                    f"required anchor {anchor.anchor_id} cannot fit the model request budget"
                )

        result = self._materialize(
            stage=stage,
            context=context,
            comparison_bindings=comparison_bindings,
            input_bindings=input_bindings,
            visible_anchors=tuple(visible),
            required=required,
            priority_terms=priority_terms,
            budget=budget,
            request_factory=request_factory,
        )
        if result.request_body_bytes > budget.max_body_bytes:
            raise ModelEvidenceBudgetError(
                "model request metadata cannot fit the declared provider budget"
            )
        return result

    @staticmethod
    def _rank_anchor(
        anchor: ContextAnchor,
        *,
        required: set[str],
        priority_terms: tuple[str, ...],
    ) -> tuple[object, ...]:
        priority_matches = sum(
            _anchor_has_exact_term(anchor, term) for term in priority_terms
        )
        score = sum(component.value for component in anchor.score_components)
        return (
            anchor.anchor_id not in required,
            -priority_matches,
            _RELATION_RANK[anchor.change_relation],
            -score,
            anchor.path,
            anchor.start_line,
            anchor.anchor_id,
        )

    def _materialize(
        self,
        *,
        stage: ModelEvidenceStage,
        context: ContextBundle,
        comparison_bindings: tuple[EvidenceArtifactBinding, ...],
        input_bindings: tuple[EvidenceArtifactBinding, ...],
        visible_anchors: tuple[ContextAnchor, ...],
        required: set[str],
        priority_terms: tuple[str, ...],
        budget: ProviderRequestBudget,
        request_factory: Callable[[ModelEvidenceEnvelope], ModelRequest],
    ) -> EnvelopeBuildResult:
        visible_ids = {anchor.anchor_id for anchor in visible_anchors}
        omissions = tuple(
            OmittedEvidenceAnchor(
                anchor_id=anchor.anchor_id,
                reason="request_budget",
            )
            for anchor in context.anchors
            if anchor.anchor_id not in visible_ids
        )

        def envelope_for_schema(schema_sha256: str) -> ModelEvidenceEnvelope:
            return ModelEvidenceEnvelope.from_content(
                stage=stage,
                snapshot_key=context.snapshot_key,
                context_sha256=context.context_sha256,
                comparison_bindings=comparison_bindings,
                input_bindings=input_bindings,
                visible_anchors=tuple(
                    _visible_anchor(
                        anchor,
                        required=required,
                        priority_terms=priority_terms,
                    )
                    for anchor in visible_anchors
                ),
                omitted_anchors=omissions,
                catalog_anchor_ids=tuple(
                    anchor.anchor_id for anchor in context.anchors
                ),
                max_request_body_bytes=budget.max_body_bytes,
                selection_policy_version=_POLICY_VERSION[stage],
                output_schema_sha256=schema_sha256,
            )

        provisional_request = request_factory(envelope_for_schema("0" * 64))
        schema_sha256 = canonical_sha256(provisional_request.output_schema)
        envelope = envelope_for_schema(schema_sha256)
        request = request_factory(envelope)
        if request.purpose != stage:
            raise ValueError("model request purpose must match evidence envelope stage")
        if canonical_sha256(request.output_schema) != schema_sha256:
            raise ValueError("model request factory changed its output schema")
        request_body_bytes = groq_request_body_bytes(
            request=request,
            model=budget.model,
        )
        return EnvelopeBuildResult(
            envelope=envelope,
            request=request,
            request_body_bytes=request_body_bytes,
        )


def _anchor_has_exact_term(anchor: ContextAnchor, term: str) -> bool:
    searchable = "\n".join(
        value
        for value in (anchor.path, anchor.java_symbol, anchor.text)
        if value is not None
    )
    pattern = rf"(?<![A-Za-z0-9_$]){re.escape(term)}(?![A-Za-z0-9_$])"
    return re.search(pattern, searchable) is not None


def _visible_anchor(
    anchor: ContextAnchor,
    *,
    required: set[str],
    priority_terms: tuple[str, ...],
) -> VisibleEvidenceAnchor:
    visible = VisibleEvidenceAnchor.from_context_anchor(anchor)
    if anchor.anchor_id in required:
        reason = "required_by_stage"
    elif any(_anchor_has_exact_term(anchor, term) for term in priority_terms):
        reason = "exact_priority_term"
    else:
        reason = "deterministic_context_rank"
    return visible.model_copy(update={"selection_reason": reason})
