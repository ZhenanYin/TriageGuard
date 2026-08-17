"""Immutable record of the exact frozen evidence made visible to one model call."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from triageguard.domain.models import ResearchArtifact
from triageguard.domain.pr_analysis import ContextAnchor, Sha256
from triageguard.provenance import canonical_sha256

ModelEvidenceStage = Literal[
    "risk_hypothesis",
    "testability_assessment",
    "gherkin_generation",
]


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
