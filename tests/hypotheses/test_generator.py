"""Tests for the structured, evidence-bound risk-proposal operation."""

import hashlib
from datetime import UTC, datetime

import pytest

from triageguard.analysis.diffs import parse_patch
from triageguard.domain.pr_analysis import (
    ContextAnchor,
    ContextBundle,
    PullRequestSnapshot,
)
from triageguard.evidence import (
    EvidenceArtifactBinding,
    ModelEvidenceBudgetError,
    ModelEvidenceEnvelope,
)
from triageguard.hypotheses import generator as risk_generator
from triageguard.hypotheses.generator import (
    build_risk_evidence,
    build_risk_request,
    generate_risk_assessment,
)
from triageguard.llm import ModelOutputInvalid
from triageguard.llm.replay_gateway import ReplayGateway
from triageguard.llm.request_budget import ProviderRequestBudget
from triageguard.provenance import canonical_json, canonical_sha256


def _snapshot() -> PullRequestSnapshot:
    """Return one frozen OpenMRS Core pull-request snapshot."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=7312,
        pull_url="https://github.com/openmrs/openmrs-core/pull/7312",
        state="open",
        default_branch="main",
        base_branch="main",
        merge_base_sha="a" * 40,
        base_sha="b" * 40,
        head_sha="c" * 40,
        candidate_sha="d" * 40,
        merge_base_tree_sha="e" * 40,
        base_tree_sha="f" * 40,
        head_tree_sha="1" * 40,
        candidate_tree_sha="2" * 40,
        acquired_at=datetime(2026, 8, 12, tzinfo=UTC),
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="3" * 64,
    )


def _diffs(snapshot: PullRequestSnapshot) -> tuple[object, ...]:
    """Return the three required frozen comparisons."""
    integration_patch = (
        b"diff --git a/api/PatientService.java b/api/PatientService.java\n"
        b"index 1111111..2222222 100644\n"
        b"--- a/api/PatientService.java\n"
        b"+++ b/api/PatientService.java\n"
        b"@@ -4 +4 @@\n"
        b"-        dao.requirePrivilege();\n"
        b"+        dao.deletePatient();\n"
    )
    return (
        parse_patch(
            kind="author_diff",
            old_sha=snapshot.merge_base_sha,
            new_sha=snapshot.head_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version=snapshot.git_version,
        ),
        parse_patch(
            kind="integration_diff",
            old_sha=snapshot.base_sha,
            new_sha=snapshot.candidate_sha,
            patch_bytes=integration_patch,
            numstat_bytes=b"1\t1\tapi/PatientService.java\0",
            git_version=snapshot.git_version,
        ),
        parse_patch(
            kind="base_drift_diff",
            old_sha=snapshot.merge_base_sha,
            new_sha=snapshot.base_sha,
            patch_bytes=b"",
            numstat_bytes=b"",
            git_version=snapshot.git_version,
        ),
    )


def _context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Return one traceable integration-change anchor."""
    text = "void purgePatient() {\n    dao.deletePatient();\n}\n"
    anchor = ContextAnchor(
        anchor_id="anchor-integration",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="4" * 40,
        path="api/PatientService.java",
        java_symbol="purgePatient",
        start_line=3,
        end_line=5,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        selection_reason="integration hunk",
        score_components=[],
        change_relation="integration_change",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(anchor,),
        selected_file_count=1,
        selected_anchor_count=1,
        selected_bytes=len(text.encode("utf-8")),
        max_files=40,
        max_anchors=80,
        max_bytes=160_000,
        max_anchor_lines=120,
        max_blob_bytes=1_000_000,
        max_search_identifiers=100,
        max_hits_per_identifier=20,
        primary_change_represented=True,
    )


def _envelope(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    diffs: tuple[object, ...],
) -> ModelEvidenceEnvelope:
    """Expose the exact complete anchor used by the risk model."""
    return ModelEvidenceEnvelope.from_content(
        stage="risk_hypothesis",
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        comparison_bindings=tuple(
            EvidenceArtifactBinding(name=diff.kind, sha256=diff.artifact_sha256)
            for diff in diffs
        ),
        input_bindings=(),
        visible_anchors=tuple(
            {
                "anchor_id": anchor.anchor_id,
                "revision_role": anchor.revision_role,
                "path": anchor.path,
                "java_symbol": anchor.java_symbol,
                "start_line": anchor.start_line,
                "end_line": anchor.end_line,
                "change_relation": anchor.change_relation,
                "visible_text": anchor.text,
                "source_text_sha256": anchor.text_sha256,
                "visible_text_sha256": anchor.text_sha256,
                "selection_reason": "required_by_stage",
            }
            for anchor in context.anchors
        ),
        omitted_anchors=(),
        catalog_anchor_ids=tuple(anchor.anchor_id for anchor in context.anchors),
        max_request_body_bytes=7_000,
        selection_policy_version="risk-evidence-v1",
        output_schema_sha256=canonical_sha256(risk_generator.RISK_OUTPUT_SCHEMA),
    )


def test_risk_request_contains_the_exact_immutable_evidence_envelope() -> None:
    """The risk model must see and echo the envelope that defines visibility."""
    snapshot = _snapshot()
    diffs = _diffs(snapshot)
    context = _context(snapshot)
    envelope = _envelope(snapshot, context, diffs)

    request = build_risk_request(
        snapshot=snapshot,
        diffs=diffs,
        context=context,
        evidence_envelope=envelope,
    )

    assert request.payload["evidence_envelope"] == envelope.model_dump(mode="json")
    assert (
        request.output_schema["properties"]["evidence_envelope_sha256"]["pattern"]
        == "^[0-9a-f]{64}$"
    )
    assert "text_truncated_for_model" not in canonical_json(request.payload)


def test_risk_evidence_builder_measures_the_exact_whole_anchor_request() -> None:
    """The real risk schema and complete anchor fit the declared wire budget."""
    snapshot = _snapshot()
    context = _context(snapshot)
    result = build_risk_evidence(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        context=context,
        budget=ProviderRequestBudget(
            provider="groq",
            model="openai/gpt-oss-120b",
            max_body_bytes=7_000,
        ),
    )

    assert result.request_body_bytes <= result.envelope.max_request_body_bytes
    assert result.envelope.visible_anchors[0].visible_text == context.anchors[0].text
    assert result.envelope.visible_anchors[0].source_text_sha256 == (
        context.anchors[0].text_sha256
    )


def test_risk_request_contains_only_frozen_evidence() -> None:
    """The model receives immutable evidence, never process configuration."""
    snapshot = _snapshot()
    diffs = _diffs(snapshot)
    context = _context(snapshot)
    envelope = _envelope(snapshot, context, diffs)

    request = build_risk_request(
        snapshot=snapshot,
        diffs=diffs,
        context=context,
        evidence_envelope=envelope,
    )

    serialized = canonical_json(request.payload)
    assert request.purpose == "risk_hypothesis"
    assert set(request.payload) == {
        "snapshot_key",
        "context_sha256",
        "comparisons",
        "evidence_envelope",
        "output_rule",
    }
    assert request.payload["snapshot_key"] == snapshot.snapshot_key
    assert request.payload["context_sha256"] == context.context_sha256
    assert [item["comparison"] for item in request.payload["comparisons"]] == [
        "author_change",
        "merge_impact",
        "main_branch_drift",
    ]
    assert snapshot.snapshot_key in serialized
    assert context.context_sha256 in serialized
    assert snapshot.merge_base_sha not in serialized
    assert "GROQ_API_KEY" not in serialized
    assert "GITHUB_TOKEN" not in serialized
    assert request.output_schema["additionalProperties"] is False
    assert "Use one readable paragraph per hypothesis" in request.system_prompt


def test_risk_request_rejects_a_required_anchor_instead_of_slicing_it() -> None:
    """A required whole anchor that cannot fit fails before any model call."""
    snapshot = _snapshot()
    source_text = "x" * 4_000
    relations = (
        "integration_change",
        "author_change",
        "base_drift_change",
    )
    anchors = tuple(
        ContextAnchor(
            anchor_id=f"anchor-{relation}-{index}",
            revision_role="candidate",
            commit_sha=snapshot.candidate_sha,
            blob_sha="4" * 40,
            path=f"api/{relation}-{index}.java",
            java_symbol="riskTarget",
            start_line=1,
            end_line=1,
            text=f"{relation} {source_text}",
            text_sha256=hashlib.sha256(
                f"{relation} {source_text}".encode()
            ).hexdigest(),
            selection_reason="synthetic large frozen evidence",
            score_components=(),
            change_relation=relation,
            truncated=False,
        )
        for index, relation in enumerate(
            (*relations, *("repository_context",) * 10),
        )
    )
    context = ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=anchors,
        selected_file_count=len(anchors),
        selected_anchor_count=len(anchors),
        selected_bytes=sum(len(anchor.text.encode("utf-8")) for anchor in anchors),
        max_files=40,
        max_anchors=80,
        max_bytes=160_000,
        max_anchor_lines=120,
        max_blob_bytes=1_000_000,
        max_search_identifiers=100,
        max_hits_per_identifier=20,
        primary_change_represented=True,
    )

    with pytest.raises(
        ModelEvidenceBudgetError,
        match="required anchor",
    ):
        build_risk_evidence(
            snapshot=snapshot,
            diffs=_diffs(snapshot),
            context=context,
            budget=ProviderRequestBudget(
                provider="groq",
                model="openai/gpt-oss-120b",
                max_body_bytes=7_000,
            ),
        )


def _model_response(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    evidence_envelope: ModelEvidenceEnvelope,
    outcome: str,
) -> dict[str, object]:
    """Return one schema-valid prerecorded model response."""
    response: dict[str, object] = {
        "snapshot_key": snapshot.snapshot_key,
        "context_sha256": context.context_sha256,
        "evidence_envelope_sha256": evidence_envelope.envelope_sha256,
        "outcome": outcome,
        "hypotheses": [],
        "rationale": None,
        "security_relevant_areas": [],
        "supporting_anchor_ids": ["anchor-integration"],
        "coverage_limitations": [],
        "reason_code": None,
        "missing_evidence": [],
        "evidence_needs": [],
        "generated_at": "2026-08-12T00:00:00Z",
    }

    if outcome == "risks_proposed":
        response["hypotheses"] = [
            {
                "claim_status": "unconfirmed_risk_hypothesis",
                "title": "Patient deletion may bypass an expected authorization check",
                "explanation": (
                    "The integration excerpt replaces one service call with "
                    "deletePatient, so the surrounding authorization behavior "
                    "needs an executable check."
                ),
                "actor": "An authenticated OpenMRS user",
                "preconditions": [
                    "The user can reach the patient-deletion service path."
                ],
                "action": "The user requests deletion of a patient record.",
                "protected_asset": "Patient records",
                "security_property": "Authorization",
                "expected_secure_behavior": (
                    "The deletion path enforces the required authorization "
                    "before deleting a patient record."
                ),
                "possible_failure": (
                    "The changed path may delete a patient record without the "
                    "expected authorization enforcement."
                ),
                "observables": [
                    "The deletion attempt is rejected when authorization is absent."
                ],
                "code_identifiers": [
                    "purgePatient",
                    "deletePatient",
                ],
                "evidence_bindings": [
                    {
                        "claim_field": "explanation",
                        "observable_index": None,
                        "anchor_ids": ["anchor-integration"],
                    },
                    {
                        "claim_field": "actor",
                        "observable_index": None,
                        "anchor_ids": ["anchor-integration"],
                    },
                    {
                        "claim_field": "action",
                        "observable_index": None,
                        "anchor_ids": ["anchor-integration"],
                    },
                    {
                        "claim_field": "expected_secure_behavior",
                        "observable_index": None,
                        "anchor_ids": ["anchor-integration"],
                    },
                    {
                        "claim_field": "possible_failure",
                        "observable_index": None,
                        "anchor_ids": ["anchor-integration"],
                    },
                    {
                        "claim_field": "observable",
                        "observable_index": 0,
                        "anchor_ids": ["anchor-integration"],
                    },
                ],
                "limitations": [
                    "The excerpt does not show the surrounding authorization checks."
                ],
                "missing_evidence": [
                    "The relevant authorization implementation is not in the context."
                ],
                "priority_rationale": (
                    "Patient deletion is a security-relevant operation that needs "
                    "human review and an executable test."
                ),
            }
        ]
    elif outcome == "no_meaningful_security_risk_found":
        response["rationale"] = (
            "The bounded evidence does not show a specific testable security-risk "
            "hypothesis."
        )
        response["security_relevant_areas"] = ["Patient deletion service behavior."]
        response["coverage_limitations"] = [
            "This is not proof of safety because the evidence is bounded."
        ]
    elif outcome == "insufficient_context_to_assess":
        response["reason_code"] = "insufficient_context_to_assess"
        response["missing_evidence"] = [
            "The authorization implementation is not available in the frozen context."
        ]
        response["evidence_needs"] = [
            {
                "need_id": "need-has-privilege",
                "category": "authorization",
                "search_terms": ["hasPrivilege"],
                "explanation": "Find the exact frozen authorization decision.",
                "supporting_anchor_ids": ["anchor-integration"],
            }
        ]
    else:
        raise ValueError("test fixture received an unsupported outcome")

    return response


@pytest.mark.parametrize(
    ("outcome", "expected_hypothesis_count"),
    [
        ("risks_proposed", 1),
        ("no_meaningful_security_risk_found", 0),
        ("insufficient_context_to_assess", 0),
    ],
)
def test_generate_risk_assessment_accepts_each_allowed_outcome(
    outcome: str,
    expected_hypothesis_count: int,
) -> None:
    """Every permitted structured outcome remains an unconfirmed draft."""
    snapshot = _snapshot()
    diffs = _diffs(snapshot)
    context = _context(snapshot)
    evidence_envelope = _envelope(snapshot, context, diffs)
    gateway = ReplayGateway(
        {
            "risk_hypothesis": _model_response(
                snapshot,
                context,
                evidence_envelope,
                outcome,
            )
        }
    )

    assessment, response = generate_risk_assessment(
        snapshot=snapshot,
        diffs=diffs,
        context=context,
        evidence_envelope=evidence_envelope,
        gateway=gateway,
    )

    assert assessment.outcome == outcome
    assert len(assessment.hypotheses) == expected_hypothesis_count
    assert assessment.snapshot_key == snapshot.snapshot_key
    assert assessment.context_sha256 == context.context_sha256
    assert response.provider == "replay"
    assert all(
        "hypothesis_id" not in hypothesis.model_dump(mode="json")
        for hypothesis in assessment.hypotheses
    )


def test_invalid_risk_draft_retains_secret_free_model_failure_provenance() -> None:
    """A schema-valid but incoherent proposal remains attributable for diagnosis."""
    snapshot = _snapshot()
    diffs = _diffs(snapshot)
    context = _context(snapshot)
    evidence_envelope = _envelope(snapshot, context, diffs)
    response = _model_response(
        snapshot,
        context,
        evidence_envelope,
        "risks_proposed",
    )
    response["hypotheses"] = []
    gateway = ReplayGateway({"risk_hypothesis": response})

    with pytest.raises(ModelOutputInvalid) as error:
        generate_risk_assessment(
            snapshot=snapshot,
            diffs=diffs,
            context=context,
            evidence_envelope=evidence_envelope,
            gateway=gateway,
        )

    provenance = error.value.provenance
    assert provenance is not None
    assert provenance.provider == "replay"
    assert provenance.purpose == "risk_hypothesis"
    assert provenance.final_outcome == "invalid_output"
    assert provenance.reason_code == "risk_assessment_invalid"
    assert provenance.response_sha256 is not None
    assert provenance.error_sha256 is not None
    assert provenance.attempts == error.value.attempts


def test_risk_response_must_echo_the_exact_evidence_envelope_hash() -> None:
    """A valid-looking response cannot detach itself from model-visible evidence."""
    snapshot = _snapshot()
    diffs = _diffs(snapshot)
    context = _context(snapshot)
    evidence_envelope = _envelope(snapshot, context, diffs)
    response = _model_response(
        snapshot,
        context,
        evidence_envelope,
        "insufficient_context_to_assess",
    )
    response["evidence_envelope_sha256"] = "f" * 64

    with pytest.raises(ModelOutputInvalid, match="envelope hash"):
        generate_risk_assessment(
            snapshot=snapshot,
            diffs=diffs,
            context=context,
            evidence_envelope=evidence_envelope,
            gateway=ReplayGateway({"risk_hypothesis": response}),
        )
