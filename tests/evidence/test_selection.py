"""Deterministic whole-anchor selection for bounded model evidence."""

import hashlib

import pytest

from triageguard.domain import ContextAnchor, ContextBundle, ContextScoreComponent
from triageguard.evidence.model_envelope import EvidenceArtifactBinding
from triageguard.evidence.selection import (
    EvidenceEnvelopeBuilder,
    ModelEvidenceBudgetError,
)
from triageguard.llm import ModelRequest, ProviderRequestBudget, groq_request_body_bytes
from triageguard.provenance import canonical_sha256


def _anchor(
    anchor_id: str,
    relation: str,
    text: str,
    line: int,
    score: float,
) -> ContextAnchor:
    return ContextAnchor(
        anchor_id=anchor_id,
        revision_role="candidate",
        commit_sha="d" * 40,
        blob_sha=hashlib.sha1(anchor_id.encode()).hexdigest(),
        path=f"api/{anchor_id}.java",
        java_symbol=anchor_id.removeprefix("anchor-"),
        start_line=line,
        end_line=line + text.count("\n") - 1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        selection_reason=f"{relation} evidence",
        score_components=(ContextScoreComponent(name="security", value=score),),
        change_relation=relation,
        truncated=False,
    )


def _context(
    *,
    large_repository_anchor: bool = False,
    large_author_anchor: bool = False,
) -> ContextBundle:
    author_text = (
        "patientDao.deletePatient(patient);\n" * 400
        if large_author_anchor
        else "patientDao.deletePatient(patient);\n"
    )
    repository_text = (
        "verifyAuthorization();\n" * 400
        if large_repository_anchor
        else "verifyAuthorization();\n"
    )
    anchors = (
        _anchor(
            "anchor-integration",
            "integration_change",
            "authorize(actor);\ndeletePatient(patient);\n",
            10,
            50.0,
        ),
        _anchor(
            "anchor-author",
            "author_change",
            author_text,
            30,
            40.0,
        ),
        _anchor(
            "anchor-context",
            "repository_context",
            repository_text,
            50,
            30.0,
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
        max_bytes=20_000,
        max_anchor_lines=500,
        max_blob_bytes=20_000,
        max_search_identifiers=20,
        max_hits_per_identifier=10,
        primary_change_represented=True,
    )


def _comparison_bindings() -> tuple[EvidenceArtifactBinding, ...]:
    return (
        EvidenceArtifactBinding(name="author_diff", sha256="1" * 64),
        EvidenceArtifactBinding(name="integration_diff", sha256="2" * 64),
        EvidenceArtifactBinding(name="base_drift_diff", sha256="3" * 64),
    )


def _request_factory(envelope) -> ModelRequest:
    return ModelRequest(
        purpose=envelope.stage,
        system_prompt="Use only the exact evidence envelope.",
        payload={"evidence_envelope": envelope.model_dump(mode="json")},
        output_schema={
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
            "additionalProperties": False,
        },
        max_output_tokens=2_048,
    )


def _budget(max_bytes: int) -> ProviderRequestBudget:
    return ProviderRequestBudget(
        provider="groq",
        model="openai/gpt-oss-120b",
        max_body_bytes=max_bytes,
    )


def test_selector_ranks_complete_anchors_and_measures_the_exact_request() -> None:
    """Reordering or slicing selected evidence must change this observable result."""
    context = _context()

    result = EvidenceEnvelopeBuilder().build(
        stage="risk_hypothesis",
        context=context,
        comparison_bindings=_comparison_bindings(),
        input_bindings=(),
        required_anchor_ids=(),
        priority_terms=(),
        budget=_budget(20_000),
        request_factory=_request_factory,
    )

    assert [anchor.anchor_id for anchor in result.envelope.visible_anchors] == [
        "anchor-integration",
        "anchor-author",
        "anchor-context",
    ]
    assert [anchor.visible_text for anchor in result.envelope.visible_anchors] == [
        anchor.text for anchor in context.anchors
    ]
    assert result.request_body_bytes == groq_request_body_bytes(
        request=result.request,
        model="openai/gpt-oss-120b",
    )
    assert result.request_body_bytes <= 20_000
    assert result.envelope.output_schema_sha256 == canonical_sha256(
        result.request.output_schema
    )


def test_risk_selector_reserves_integration_then_author_before_optional_context() -> (
    None
):
    """A large optional anchor must not displace either primary change relation."""
    context = _context(large_repository_anchor=True)
    original = context.model_dump(mode="json")

    result = EvidenceEnvelopeBuilder().build(
        stage="risk_hypothesis",
        context=context,
        comparison_bindings=_comparison_bindings(),
        input_bindings=(),
        required_anchor_ids=(),
        priority_terms=(),
        budget=_budget(7_000),
        request_factory=_request_factory,
    )

    assert {anchor.anchor_id for anchor in result.envelope.visible_anchors} == {
        "anchor-integration",
        "anchor-author",
    }
    assert len(result.envelope.omitted_anchors) == 1
    assert result.envelope.omitted_anchors[0].anchor_id == "anchor-context"
    assert result.envelope.omitted_anchors[0].reason == "request_budget"
    assert context.model_dump(mode="json") == original


def test_risk_selector_fails_when_a_reserved_author_anchor_cannot_fit() -> None:
    """The author comparison must not be silently dropped to satisfy the budget."""
    context = _context(large_author_anchor=True)

    with pytest.raises(
        ModelEvidenceBudgetError,
        match="anchor-author",
    ) as captured:
        EvidenceEnvelopeBuilder().build(
            stage="risk_hypothesis",
            context=context,
            comparison_bindings=_comparison_bindings(),
            input_bindings=(),
            required_anchor_ids=(),
            priority_terms=(),
            budget=_budget(7_000),
            request_factory=_request_factory,
        )

    assert captured.value.stage == "risk_hypothesis"
    assert captured.value.request_body_bytes > 7_000
    assert captured.value.limit_bytes == 7_000
    assert captured.value.reason_code == "model_request_too_large"


def test_required_citation_that_cannot_fit_fails_instead_of_being_omitted() -> None:
    """A reviewed citation must never silently disappear from testability evidence."""
    context = _context(large_repository_anchor=True)

    with pytest.raises(ModelEvidenceBudgetError, match="anchor-context"):
        EvidenceEnvelopeBuilder().build(
            stage="testability_assessment",
            context=context,
            comparison_bindings=_comparison_bindings(),
            input_bindings=(
                EvidenceArtifactBinding(name="reviewed_risk", sha256="4" * 64),
            ),
            required_anchor_ids=("anchor-context",),
            priority_terms=("verifyAuthorization",),
            budget=_budget(7_000),
            request_factory=_request_factory,
        )


def test_gherkin_selector_requires_the_union_of_review_and_testability_bindings() -> (
    None
):
    """Removing any caller-supplied setup/action/observable citation must fail."""
    context = _context()

    result = EvidenceEnvelopeBuilder().build(
        stage="gherkin_generation",
        context=context,
        comparison_bindings=_comparison_bindings(),
        input_bindings=(
            EvidenceArtifactBinding(name="reviewed_risk", sha256="4" * 64),
            EvidenceArtifactBinding(name="testability_assessment", sha256="5" * 64),
        ),
        required_anchor_ids=(
            "anchor-author",
            "anchor-integration",
            "anchor-context",
        ),
        priority_terms=("deletePatient", "verifyAuthorization"),
        budget=_budget(20_000),
        request_factory=_request_factory,
    )

    assert {anchor.anchor_id for anchor in result.envelope.visible_anchors} == {
        "anchor-integration",
        "anchor-author",
        "anchor-context",
    }


def test_gherkin_selector_does_not_omit_an_oversized_union_binding() -> None:
    """Every reviewed setup/action/observable citation remains mandatory."""
    context = _context(large_repository_anchor=True)

    with pytest.raises(ModelEvidenceBudgetError, match="anchor-context"):
        EvidenceEnvelopeBuilder().build(
            stage="gherkin_generation",
            context=context,
            comparison_bindings=_comparison_bindings(),
            input_bindings=(
                EvidenceArtifactBinding(name="reviewed_risk", sha256="4" * 64),
                EvidenceArtifactBinding(
                    name="testability_assessment",
                    sha256="5" * 64,
                ),
            ),
            required_anchor_ids=(
                "anchor-author",
                "anchor-integration",
                "anchor-context",
            ),
            priority_terms=("deletePatient", "verifyAuthorization"),
            budget=_budget(7_000),
            request_factory=_request_factory,
        )


def test_selector_rejects_unknown_or_duplicate_required_anchor_ids() -> None:
    """A misspelled citation must fail locally rather than weaken the evidence set."""
    context = _context()

    for required in (("anchor-missing",), ("anchor-author", "anchor-author")):
        with pytest.raises(ValueError, match="required anchor"):
            EvidenceEnvelopeBuilder().build(
                stage="testability_assessment",
                context=context,
                comparison_bindings=_comparison_bindings(),
                input_bindings=(),
                required_anchor_ids=required,
                priority_terms=(),
                budget=_budget(20_000),
                request_factory=_request_factory,
            )


def test_priority_terms_deterministically_rank_optional_exact_matches_first() -> None:
    """An exact requested identifier must outrank unrelated optional evidence."""
    context = _context()
    builder = EvidenceEnvelopeBuilder()
    arguments = {
        "stage": "testability_assessment",
        "context": context,
        "comparison_bindings": _comparison_bindings(),
        "input_bindings": (),
        "required_anchor_ids": (),
        "priority_terms": ("verifyAuthorization",),
        "budget": _budget(20_000),
        "request_factory": _request_factory,
    }

    first = builder.build(**arguments)
    second = builder.build(**arguments)

    assert first == second
    assert first.envelope.visible_anchors[0].anchor_id == "anchor-context"
