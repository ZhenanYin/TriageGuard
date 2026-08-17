"""Integrity contracts for immutable, model-visible evidence envelopes."""

import hashlib

import pytest
from pydantic import ValidationError

from triageguard.domain import ContextAnchor, ContextBundle, ContextScoreComponent
from triageguard.evidence.model_envelope import (
    EvidenceArtifactBinding,
    ModelEvidenceEnvelope,
    ModelEvidencePreflightStop,
    OmittedEvidenceAnchor,
    VisibleEvidenceAnchor,
)
from triageguard.provenance import canonical_sha256


def test_preflight_stop_records_only_a_bound_local_budget_overflow() -> None:
    """A pre-provider stop must bind its exact prepared evidence and byte policy."""
    stop = ModelEvidencePreflightStop(
        stage="risk_hypothesis",
        snapshot_key="a" * 64,
        context_sha256="b" * 64,
        reason_code="model_request_too_large",
        request_body_bytes=12_589,
        max_request_body_bytes=7_000,
        catalog_anchor_count=54,
        observed_at="2026-08-17T17:30:00Z",
    )

    assert stop.request_body_bytes == 12_589
    assert stop.max_request_body_bytes == 7_000
    assert stop.catalog_anchor_count == 54


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("snapshot_key", "not-a-hash"),
        ("reason_code", "groq_non_retryable_error"),
        ("request_body_bytes", 7_000),
        ("catalog_anchor_count", -1),
    ],
)
def test_preflight_stop_rejects_unbound_or_nonoverflow_content(
    field_name: str,
    replacement: object,
) -> None:
    """Invalid local-stop provenance must fail before it can be persisted."""
    payload = {
        "stage": "risk_hypothesis",
        "snapshot_key": "a" * 64,
        "context_sha256": "b" * 64,
        "reason_code": "model_request_too_large",
        "request_body_bytes": 12_589,
        "max_request_body_bytes": 7_000,
        "catalog_anchor_count": 54,
        "observed_at": "2026-08-17T17:30:00Z",
    }
    payload[field_name] = replacement

    with pytest.raises(ValidationError):
        ModelEvidencePreflightStop.model_validate(payload)


def _anchor(
    anchor_id: str,
    *,
    relation: str,
    text: str,
    path: str,
    line: int,
) -> ContextAnchor:
    return ContextAnchor(
        anchor_id=anchor_id,
        revision_role="candidate",
        commit_sha="d" * 40,
        blob_sha=hashlib.sha1(anchor_id.encode()).hexdigest(),
        path=path,
        java_symbol="deletePatient",
        start_line=line,
        end_line=line,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        selection_reason=f"selected {relation}",
        score_components=(ContextScoreComponent(name="security", value=10.0),),
        change_relation=relation,
        truncated=False,
    )


def _context_with_three_anchors() -> ContextBundle:
    anchors = (
        _anchor(
            "anchor-integration",
            relation="integration_change",
            text="authorize(actor);\ndeletePatient(patient);\n",
            path="api/PatientService.java",
            line=20,
        ),
        _anchor(
            "anchor-author",
            relation="author_change",
            text="patientDao.deletePatient(patient);\n",
            path="dao/PatientDao.java",
            line=40,
        ),
        _anchor(
            "anchor-context",
            relation="repository_context",
            text='requirePrivilege("Delete Patients");\n',
            path="api/Authorization.java",
            line=60,
        ),
    )
    return ContextBundle.from_content(
        snapshot_key="a" * 64,
        anchors=anchors,
        selected_file_count=3,
        selected_anchor_count=3,
        selected_bytes=sum(len(anchor.text.encode()) for anchor in anchors),
        max_files=10,
        max_anchors=10,
        max_bytes=10_000,
        max_anchor_lines=20,
        max_blob_bytes=10_000,
        max_search_identifiers=20,
        max_hits_per_identifier=10,
        primary_change_represented=True,
    )


def _comparison_bindings() -> tuple[EvidenceArtifactBinding, ...]:
    return (
        EvidenceArtifactBinding(name="author_diff", sha256="1" * 64),
        EvidenceArtifactBinding(name="base_drift_diff", sha256="2" * 64),
        EvidenceArtifactBinding(name="integration_diff", sha256="3" * 64),
    )


def _envelope() -> ModelEvidenceEnvelope:
    context = _context_with_three_anchors()
    return ModelEvidenceEnvelope.from_content(
        stage="risk_hypothesis",
        snapshot_key=context.snapshot_key,
        context_sha256=context.context_sha256,
        comparison_bindings=_comparison_bindings(),
        input_bindings=(),
        visible_anchors=(
            VisibleEvidenceAnchor.from_context_anchor(context.anchors[0]),
        ),
        omitted_anchors=(
            OmittedEvidenceAnchor(
                anchor_id=context.anchors[1].anchor_id,
                reason="request_budget",
            ),
            OmittedEvidenceAnchor(
                anchor_id=context.anchors[2].anchor_id,
                reason="request_budget",
            ),
        ),
        catalog_anchor_ids=tuple(anchor.anchor_id for anchor in context.anchors),
        max_request_body_bytes=7_000,
        selection_policy_version="risk-evidence-v1",
        output_schema_sha256="f" * 64,
    )


def test_envelope_hashes_exact_visible_text_and_partitions_catalog() -> None:
    """Dropping source text or a catalog disposition must invalidate the envelope."""
    context = _context_with_three_anchors()

    envelope = _envelope()

    visible = envelope.visible_anchors[0]
    assert visible.visible_text == context.anchors[0].text
    assert visible.source_text_sha256 == context.anchors[0].text_sha256
    assert (
        visible.visible_text_sha256
        == hashlib.sha256(context.anchors[0].text.encode()).hexdigest()
    )
    assert set(envelope.catalog_anchor_ids) == {
        anchor.anchor_id for anchor in context.anchors
    }
    assert len(envelope.envelope_sha256) == 64


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("stage", "testability_assessment"),
        ("max_request_body_bytes", 6_999),
        ("selection_policy_version", "risk-evidence-v2"),
        ("output_schema_sha256", "e" * 64),
        ("envelope_sha256", "0" * 64),
    ],
)
def test_envelope_rejects_tampered_top_level_content(
    field_name: str,
    replacement: object,
) -> None:
    """Changing a request-defining field without re-derivation must be detected."""
    payload = _envelope().model_dump(mode="json")
    payload[field_name] = replacement

    with pytest.raises(ValidationError, match="envelope SHA-256"):
        ModelEvidenceEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("visible_text", "deleteEverything();\n"),
        ("source_text_sha256", "e" * 64),
        ("visible_text_sha256", "e" * 64),
    ],
)
def test_visible_anchor_rejects_tampered_text_or_hashes(
    field_name: str,
    replacement: str,
) -> None:
    """A model-visible excerpt must remain byte-identical to its frozen source."""
    payload = _envelope().model_dump(mode="json")
    payload["visible_anchors"][0][field_name] = replacement

    with pytest.raises(ValidationError, match="text SHA-256"):
        ModelEvidenceEnvelope.model_validate(payload)


def test_envelope_hash_binds_omission_reasons_and_input_artifacts() -> None:
    """Changing hidden evidence or upstream inputs must invalidate provenance."""
    omitted_payload = _envelope().model_dump(mode="json")
    omitted_payload["omitted_anchors"][0]["reason"] = "stage_irrelevant"
    input_payload = _envelope().model_dump(mode="json")
    input_payload["input_bindings"] = [{"name": "reviewed_risk", "sha256": "4" * 64}]

    for payload in (omitted_payload, input_payload):
        with pytest.raises(ValidationError, match="envelope SHA-256"):
            ModelEvidenceEnvelope.model_validate(payload)


def test_envelope_hash_binds_the_order_of_model_visible_anchors() -> None:
    """Changing prompt evidence order without re-derivation must be detected."""
    context = _context_with_three_anchors()
    envelope = ModelEvidenceEnvelope.from_content(
        stage="risk_hypothesis",
        snapshot_key=context.snapshot_key,
        context_sha256=context.context_sha256,
        comparison_bindings=_comparison_bindings(),
        input_bindings=(),
        visible_anchors=tuple(
            VisibleEvidenceAnchor.from_context_anchor(anchor)
            for anchor in context.anchors[:2]
        ),
        omitted_anchors=(
            OmittedEvidenceAnchor(
                anchor_id=context.anchors[2].anchor_id,
                reason="request_budget",
            ),
        ),
        catalog_anchor_ids=tuple(anchor.anchor_id for anchor in context.anchors),
        max_request_body_bytes=7_000,
        selection_policy_version="risk-evidence-v1",
        output_schema_sha256="f" * 64,
    )
    payload = envelope.model_dump(mode="json")
    payload["visible_anchors"] = list(reversed(payload["visible_anchors"]))

    with pytest.raises(ValidationError, match="envelope SHA-256"):
        ModelEvidenceEnvelope.model_validate(payload)


def test_envelope_requires_unique_binding_names_and_an_exact_partition() -> None:
    """Ambiguous inputs or undisposed catalog anchors must never reach a model."""
    context = _context_with_three_anchors()
    duplicate_bindings = (
        EvidenceArtifactBinding(name="integration_diff", sha256="1" * 64),
        EvidenceArtifactBinding(name="integration_diff", sha256="2" * 64),
    )

    with pytest.raises(ValidationError, match="binding names"):
        ModelEvidenceEnvelope.from_content(
            stage="risk_hypothesis",
            snapshot_key=context.snapshot_key,
            context_sha256=context.context_sha256,
            comparison_bindings=duplicate_bindings,
            input_bindings=(),
            visible_anchors=(
                VisibleEvidenceAnchor.from_context_anchor(context.anchors[0]),
            ),
            omitted_anchors=(),
            catalog_anchor_ids=tuple(anchor.anchor_id for anchor in context.anchors),
            max_request_body_bytes=7_000,
            selection_policy_version="risk-evidence-v1",
            output_schema_sha256="f" * 64,
        )

    with pytest.raises(ValidationError, match="partition"):
        ModelEvidenceEnvelope.from_content(
            stage="risk_hypothesis",
            snapshot_key=context.snapshot_key,
            context_sha256=context.context_sha256,
            comparison_bindings=_comparison_bindings(),
            input_bindings=(),
            visible_anchors=(
                VisibleEvidenceAnchor.from_context_anchor(context.anchors[0]),
            ),
            omitted_anchors=(),
            catalog_anchor_ids=tuple(anchor.anchor_id for anchor in context.anchors),
            max_request_body_bytes=7_000,
            selection_policy_version="risk-evidence-v1",
            output_schema_sha256="f" * 64,
        )


def test_envelope_factory_normalizes_unordered_bindings_and_catalog_dispositions() -> (
    None
):
    """Equivalent input ordering must produce one reproducible envelope identity."""
    first = _envelope()
    payload = first.model_dump(mode="python", exclude={"envelope_sha256"})

    second = ModelEvidenceEnvelope.from_content(
        **{
            **payload,
            "comparison_bindings": tuple(reversed(payload["comparison_bindings"])),
            "omitted_anchors": tuple(reversed(payload["omitted_anchors"])),
            "catalog_anchor_ids": tuple(reversed(payload["catalog_anchor_ids"])),
        }
    )

    assert second == first


def test_direct_validation_rejects_noncanonical_inventory_order() -> None:
    """Recomputed hashes must not permit multiple serializations of one inventory."""
    payload = _envelope().model_dump(mode="json")
    payload["comparison_bindings"] = list(reversed(payload["comparison_bindings"]))
    payload["omitted_anchors"] = list(reversed(payload["omitted_anchors"]))
    payload["catalog_anchor_ids"] = list(reversed(payload["catalog_anchor_ids"]))
    payload["envelope_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "envelope_sha256"}
    )

    with pytest.raises(ValidationError, match="canonical order"):
        ModelEvidenceEnvelope.model_validate(payload)
