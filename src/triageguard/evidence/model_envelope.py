"""Immutable record of the exact frozen evidence made visible to one model call."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from triageguard.domain.models import ResearchArtifact
from triageguard.domain.pr_analysis import ContextAnchor, ContextBundle, Sha256
from triageguard.provenance import canonical_sha256

ModelEvidenceStage = Literal[
    "risk_hypothesis",
    "testability_assessment",
    "gherkin_generation",
]


class ModelEvidencePreflightStop(ResearchArtifact):
    """Safe durable provenance for a request stopped before an envelope exists."""

    schema_version: Literal[1] = 1
    stage: ModelEvidenceStage
    snapshot_key: Sha256
    context_sha256: Sha256
    reason_code: Literal["model_request_too_large"]
    request_body_bytes: StrictInt = Field(gt=0)
    max_request_body_bytes: StrictInt = Field(gt=0)
    catalog_anchor_count: StrictInt = Field(ge=0)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_overflow(self) -> ModelEvidencePreflightStop:
        if self.request_body_bytes <= self.max_request_body_bytes:
            raise ValueError("preflight stop requires a request over its byte limit")
        return self


class EvidenceArtifactBinding(ResearchArtifact):
    """Name and content identity of one immutable upstream artifact."""

    name: StrictStr = Field(min_length=1)
    sha256: Sha256


class VisibleEvidenceAnchor(ResearchArtifact):
    """One complete context anchor exposed verbatim to a model."""

    anchor_id: StrictStr = Field(min_length=1)
    revision_role: Literal["merge_base", "base", "head", "candidate"]
    path: StrictStr = Field(min_length=1)
    java_symbol: StrictStr | None
    start_line: StrictInt = Field(gt=0)
    end_line: StrictInt = Field(gt=0)
    change_relation: Literal[
        "author_change",
        "integration_change",
        "base_drift_change",
        "repository_context",
    ]
    visible_text: StrictStr = Field(min_length=1)
    source_text_sha256: Sha256
    visible_text_sha256: Sha256
    selection_reason: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def validate_exact_source_text(self) -> VisibleEvidenceAnchor:
        if self.end_line < self.start_line:
            raise ValueError("visible anchor end_line must not precede start_line")
        digest = hashlib.sha256(self.visible_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != digest or self.visible_text_sha256 != digest:
            raise ValueError(
                "visible anchor text SHA-256 values must match its exact text"
            )
        return self

    @classmethod
    def from_context_anchor(cls, anchor: ContextAnchor) -> VisibleEvidenceAnchor:
        """Copy one complete frozen anchor without slicing or rewriting its text."""
        return cls(
            anchor_id=anchor.anchor_id,
            revision_role=anchor.revision_role,
            path=anchor.path,
            java_symbol=anchor.java_symbol,
            start_line=anchor.start_line,
            end_line=anchor.end_line,
            change_relation=anchor.change_relation,
            visible_text=anchor.text,
            source_text_sha256=anchor.text_sha256,
            visible_text_sha256=hashlib.sha256(anchor.text.encode("utf-8")).hexdigest(),
            selection_reason=anchor.selection_reason,
        )


class OmittedEvidenceAnchor(ResearchArtifact):
    """One catalog anchor withheld from a specific model request."""

    anchor_id: StrictStr = Field(min_length=1)
    reason: Literal["request_budget", "stage_irrelevant", "superseded"]


class ModelEvidenceEnvelope(ResearchArtifact):
    """Complete, self-hashed partition of evidence visible to one model stage."""

    stage: ModelEvidenceStage
    snapshot_key: Sha256
    context_sha256: Sha256
    comparison_bindings: tuple[EvidenceArtifactBinding, ...]
    input_bindings: tuple[EvidenceArtifactBinding, ...]
    visible_anchors: tuple[VisibleEvidenceAnchor, ...]
    omitted_anchors: tuple[OmittedEvidenceAnchor, ...]
    catalog_anchor_ids: tuple[StrictStr, ...]
    max_request_body_bytes: StrictInt = Field(gt=0)
    selection_policy_version: StrictStr = Field(min_length=1)
    output_schema_sha256: Sha256
    envelope_sha256: Sha256

    @model_validator(mode="after")
    def validate_integrity(self) -> ModelEvidenceEnvelope:
        comparison_names = [binding.name for binding in self.comparison_bindings]
        input_names = [binding.name for binding in self.input_bindings]
        binding_names = [
            binding.name
            for binding in (*self.comparison_bindings, *self.input_bindings)
        ]
        if len(binding_names) != len(set(binding_names)):
            raise ValueError("evidence artifact binding names must be unique")

        visible_ids = [anchor.anchor_id for anchor in self.visible_anchors]
        omitted_ids = [anchor.anchor_id for anchor in self.omitted_anchors]
        catalog_ids = list(self.catalog_anchor_ids)
        if (
            comparison_names != sorted(comparison_names)
            or input_names != sorted(input_names)
            or omitted_ids != sorted(omitted_ids)
            or catalog_ids != sorted(catalog_ids)
        ):
            raise ValueError(
                "evidence bindings, omissions, and catalog require canonical order"
            )
        if len(visible_ids) != len(set(visible_ids)):
            raise ValueError("visible evidence anchor IDs must be unique")
        if len(omitted_ids) != len(set(omitted_ids)):
            raise ValueError("omitted evidence anchor IDs must be unique")
        if len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("catalog evidence anchor IDs must be unique")
        if set(visible_ids) & set(omitted_ids) or set(visible_ids + omitted_ids) != set(
            catalog_ids
        ):
            raise ValueError(
                "visible and omitted evidence must exactly partition the catalog"
            )

        content = self.model_dump(mode="json", exclude={"envelope_sha256"})
        if self.envelope_sha256 != canonical_sha256(content):
            raise ValueError(
                "envelope SHA-256 must match canonical model evidence content"
            )
        return self

    @classmethod
    def from_content(cls, **values: object) -> ModelEvidenceEnvelope:
        """Normalize unordered inventories and derive the canonical envelope hash."""
        comparison_bindings = tuple(
            sorted(
                (
                    EvidenceArtifactBinding.model_validate(binding)
                    for binding in values.get("comparison_bindings", ())
                ),
                key=lambda binding: binding.name,
            )
        )
        input_bindings = tuple(
            sorted(
                (
                    EvidenceArtifactBinding.model_validate(binding)
                    for binding in values.get("input_bindings", ())
                ),
                key=lambda binding: binding.name,
            )
        )
        visible_anchors = tuple(
            VisibleEvidenceAnchor.model_validate(anchor)
            for anchor in values.get("visible_anchors", ())
        )
        omitted_anchors = tuple(
            sorted(
                (
                    OmittedEvidenceAnchor.model_validate(anchor)
                    for anchor in values.get("omitted_anchors", ())
                ),
                key=lambda anchor: anchor.anchor_id,
            )
        )
        catalog_value = values.get("catalog_anchor_ids")
        if catalog_value is None:
            catalog_value = tuple(
                anchor.anchor_id for anchor in (*visible_anchors, *omitted_anchors)
            )
        catalog_anchor_ids = tuple(sorted(catalog_value))
        normalized = {
            **values,
            "comparison_bindings": comparison_bindings,
            "input_bindings": input_bindings,
            "visible_anchors": visible_anchors,
            "omitted_anchors": omitted_anchors,
            "catalog_anchor_ids": catalog_anchor_ids,
        }
        normalized.pop("envelope_sha256", None)
        provisional = cls.model_construct(**normalized)
        content = provisional.model_dump(mode="json", exclude={"envelope_sha256"})
        return cls.model_validate(
            {**normalized, "envelope_sha256": canonical_sha256(content)}
        )


def validate_envelope_binding(
    *,
    envelope: ModelEvidenceEnvelope,
    stage: ModelEvidenceStage,
    context: ContextBundle,
    comparison_bindings: tuple[EvidenceArtifactBinding, ...],
    input_bindings: tuple[EvidenceArtifactBinding, ...],
    output_schema_sha256: Sha256,
    max_request_body_bytes: int | None = None,
) -> ModelEvidenceEnvelope:
    """Revalidate an envelope and bind every visible byte to frozen context."""
    normalized = ModelEvidenceEnvelope.model_validate(envelope.model_dump(mode="json"))
    expected_comparisons = tuple(
        sorted(comparison_bindings, key=lambda binding: binding.name)
    )
    expected_inputs = tuple(sorted(input_bindings, key=lambda binding: binding.name))
    if normalized.stage != stage:
        raise ValueError("evidence envelope stage does not match the model operation")
    if (
        normalized.snapshot_key != context.snapshot_key
        or normalized.context_sha256 != context.context_sha256
    ):
        raise ValueError("evidence envelope does not bind the frozen context")
    if normalized.comparison_bindings != expected_comparisons:
        raise ValueError("evidence envelope does not bind the frozen comparisons")
    if normalized.input_bindings != expected_inputs:
        raise ValueError("evidence envelope does not bind the declared inputs")
    if normalized.output_schema_sha256 != output_schema_sha256:
        raise ValueError("evidence envelope does not bind the output schema")
    if (
        max_request_body_bytes is not None
        and normalized.max_request_body_bytes != max_request_body_bytes
    ):
        raise ValueError("evidence envelope does not bind the request budget")

    catalog = {anchor.anchor_id: anchor for anchor in context.anchors}
    if normalized.catalog_anchor_ids != tuple(sorted(catalog)):
        raise ValueError("evidence envelope catalog does not match frozen context")
    for visible in normalized.visible_anchors:
        source = catalog.get(visible.anchor_id)
        if source is None or (
            visible.revision_role != source.revision_role
            or visible.path != source.path
            or visible.java_symbol != source.java_symbol
            or visible.start_line != source.start_line
            or visible.end_line != source.end_line
            or visible.change_relation != source.change_relation
            or visible.visible_text != source.text
            or visible.source_text_sha256 != source.text_sha256
        ):
            raise ValueError("visible evidence anchor does not match its frozen source")
    return normalized
