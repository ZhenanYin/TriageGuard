"""Contracts for immutable Milestone 2 PR-analysis artifacts."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from triageguard.domain.pr_analysis import (
    ClaimEvidenceBinding,
    PullRequestSnapshot,
    RiskHypothesis,
)


def test_pr_analysis_contracts_are_available_from_the_domain_package() -> None:
    """Later workflow components should depend on the stable domain boundary."""
    import triageguard.domain as public_domain

    assert public_domain.PullRequestSnapshot is PullRequestSnapshot
    assert all(
        contract is not None
        for contract in (
            public_domain.ContextBundle,
            public_domain.DiffArtifact,
            public_domain.GherkinApproval,
            public_domain.GherkinCandidate,
            public_domain.HumanReviewedRisk,
            public_domain.MilestoneTwoRunRecord,
            public_domain.RiskAssessment,
            public_domain.RiskAssessmentDraft,
            public_domain.SnapshotFreshness,
        )
    )


def snapshot_payload(**changes: object) -> dict[str, object]:
    """Return independent, canonical values for a frozen PR snapshot."""
    payload: dict[str, object] = {
        "snapshot_key": "0" * 64,
        "repository": "openmrs/openmrs-core",
        "pull_number": 123,
        "pull_url": "https://github.com/openmrs/openmrs-core/pull/123",
        "state": "open",
        "default_branch": "main",
        "base_branch": "main",
        "merge_base_sha": "1" * 40,
        "base_sha": "2" * 40,
        "head_sha": "3" * 40,
        "candidate_sha": "4" * 40,
        "merge_base_tree_sha": "5" * 40,
        "base_tree_sha": "6" * 40,
        "head_tree_sha": "7" * 40,
        "candidate_tree_sha": "8" * 40,
        "acquired_at": datetime(2026, 8, 11, tzinfo=UTC),
        "github_api_version": "2026-03-10",
        "git_version": "2.47.1",
        "analysis_config_sha256": "9" * 64,
    }
    payload.update(changes)
    return payload


def test_snapshot_requires_four_distinct_full_commit_shas() -> None:
    """Short or reused frozen revisions would make a PR analysis non-reproducible."""
    with pytest.raises(ValidationError, match="full 40-character"):
        PullRequestSnapshot.model_validate(snapshot_payload(base_sha="abc1234"))

    with pytest.raises(ValidationError, match="distinct"):
        PullRequestSnapshot.model_validate(snapshot_payload(candidate_sha="2" * 40))


def test_snapshot_rejects_non_utc_acquisition_time() -> None:
    """A local timestamp would make the frozen acquisition order ambiguous."""
    with pytest.raises(ValidationError, match="UTC"):
        PullRequestSnapshot.model_validate(
            snapshot_payload(acquired_at=datetime(2026, 8, 11, tzinfo=timezone(timedelta(hours=1))))
        )


def test_risk_hypothesis_derives_stable_unique_citation_ids() -> None:
    """A duplicate citation list must not be persisted separately from bindings."""
    hypothesis = RiskHypothesis(
        claim_status="unconfirmed_risk_hypothesis",
        title="Authorization check may be bypassed",
        explanation="The integration change changes a privilege boundary.",
        actor="authenticated clerk",
        preconditions=["A protected record exists."],
        action="Submit the changed endpoint request.",
        protected_asset="Protected patient record",
        security_property="Authorization is enforced.",
        expected_secure_behavior="The request is denied.",
        possible_failure="The request succeeds without the privilege.",
        observables=["HTTP response", "Persistent record state"],
        code_identifiers=["requirePrivilege"],
        evidence_bindings=[
            ClaimEvidenceBinding(
                claim_field="actor", observable_index=None, anchor_ids=["anchor-b"]
            ),
            ClaimEvidenceBinding(
                claim_field="action", observable_index=None, anchor_ids=["anchor-a"]
            ),
            ClaimEvidenceBinding(
                claim_field="expected_secure_behavior",
                observable_index=None,
                anchor_ids=["anchor-b", "anchor-c"],
            ),
            ClaimEvidenceBinding(
                claim_field="possible_failure",
                observable_index=None,
                anchor_ids=["anchor-c"],
            ),
            ClaimEvidenceBinding(
                claim_field="observable", observable_index=0, anchor_ids=["anchor-a"]
            ),
            ClaimEvidenceBinding(
                claim_field="observable", observable_index=1, anchor_ids=["anchor-c"]
            ),
        ],
        limitations=["Only bounded context was reviewed."],
        missing_evidence=[],
        priority_rationale="The changed check protects patient data.",
        hypothesis_id="risk-1",
    )

    assert hypothesis.citation_anchor_ids == ["anchor-b", "anchor-a", "anchor-c"]
