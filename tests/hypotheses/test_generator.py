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
from triageguard.hypotheses.generator import (
    build_risk_request,
    generate_risk_assessment,
)
from triageguard.llm.replay_gateway import ReplayGateway
from triageguard.provenance import canonical_json


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


def test_risk_request_contains_only_frozen_evidence() -> None:
    """The model receives immutable evidence, never process configuration."""
    snapshot = _snapshot()
    context = _context(snapshot)

    request = build_risk_request(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        context=context,
    )

    serialized = canonical_json(request.payload)
    assert request.purpose == "risk_hypothesis"
    assert set(request.payload) == {
        "snapshot",
        "diff_summaries",
        "context_anchors",
        "context_limits",
        "output_rules",
    }
    assert snapshot.snapshot_key in serialized
    assert context.context_sha256 in serialized
    assert "GROQ_API_KEY" not in serialized
    assert "GITHUB_TOKEN" not in serialized
    assert request.output_schema["additionalProperties"] is False


def _model_response(
    snapshot: PullRequestSnapshot,
    context: ContextBundle,
    outcome: str,
) -> dict[str, object]:
    """Return one schema-valid prerecorded model response."""
    response: dict[str, object] = {
        "snapshot_key": snapshot.snapshot_key,
        "context_sha256": context.context_sha256,
        "outcome": outcome,
        "hypotheses": [],
        "rationale": None,
        "security_relevant_areas": [],
        "supporting_anchor_ids": ["anchor-integration"],
        "coverage_limitations": [],
        "reason_code": None,
        "missing_evidence": [],
        "needed_evidence": [],
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
        response["needed_evidence"] = [
            "The relevant authorization implementation and its tests."
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
    context = _context(snapshot)
    gateway = ReplayGateway(
        {
            "risk_hypothesis": _model_response(
                snapshot,
                context,
                outcome,
            )
        }
    )

    assessment, response = generate_risk_assessment(
        snapshot=snapshot,
        diffs=_diffs(snapshot),
        context=context,
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
