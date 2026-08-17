"""Tests for deterministic refinement using only already-frozen Java code."""

import hashlib
from datetime import UTC, datetime

from triageguard.analysis.context import ContextLimits
from triageguard.analysis.refinement import FrozenContextRefiner
from triageguard.domain import (
    ContextAnchor,
    ContextBundle,
    FrozenEvidenceNeed,
    PullRequestSnapshot,
)
from triageguard.domain import (
    TestabilityAssessment as Assessment,
)
from triageguard.sources.git import GitTreeEntry

NOW = datetime(2026, 8, 16, tzinfo=UTC)


def _snapshot() -> PullRequestSnapshot:
    """Return one fixed four-revision pull-request snapshot."""
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
        acquired_at=NOW,
        github_api_version="2026-03-10",
        git_version="2.47.1",
        acquisition_tool_version="triageguard/2.0.0",
        analysis_config_sha256="3" * 64,
    )


def _initial_context(snapshot: PullRequestSnapshot) -> ContextBundle:
    """Return the initial evidence with one already-represented code change."""
    text = "dao.deletePatient();\n"
    anchor = ContextAnchor(
        anchor_id="anchor-integration",
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha="4" * 40,
        path="api/PatientService.java",
        java_symbol="purgePatient",
        start_line=4,
        end_line=4,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        selection_reason="integration change",
        score_components=(),
        change_relation="integration_change",
        truncated=False,
    )
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=(anchor,),
        selected_file_count=1,
        selected_anchor_count=1,
        selected_bytes=len(text.encode("utf-8")),
        max_files=2,
        max_anchors=4,
        max_bytes=4_000,
        max_anchor_lines=20,
        max_blob_bytes=4_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
        primary_change_represented=True,
    )


def test_refiner_adds_a_matching_anchor_from_frozen_candidate_code() -> None:
    """A precise evidence need produces a successor context without refetching."""
    snapshot = _snapshot()
    context = _initial_context(snapshot)
    need = FrozenEvidenceNeed(
        need_id="need-delete-authorization",
        category="authorization",
        search_terms=("verifyDeleteAuthorization",),
        explanation="Find the frozen authorization check for patient deletion.",
        supporting_anchor_ids=("anchor-integration",),
    )
    assessment = Assessment.from_content(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        reviewed_risk_sha256="5" * 64,
        evidence_envelope_sha256="6" * 64,
        decision="needs_more_frozen_evidence",
        bindings=(),
        evidence_needs=(need,),
        explanation="The available evidence lacks the authorization check.",
        generated_at=NOW,
        validated_at=NOW,
    )
    source = (
        b"package org.openmrs.api;\n"
        b"class AuthorizationService {\n"
        b"    void verifyDeleteAuthorization() {\n"
        b'        requirePrivilege("Delete Patients");\n'
        b"    }\n"
        b"}\n"
    )

    class FrozenStore:
        """Expose only one blob from the already-frozen candidate commit."""

        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha == snapshot.candidate_sha
            return (
                GitTreeEntry(
                    mode="100644",
                    object_type="blob",
                    object_sha="6" * 40,
                    path="api/AuthorizationService.java",
                ),
            )

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            assert blob_sha == "6" * 40
            assert max_bytes == 4_000
            return source

    limits = ContextLimits(
        max_files=2,
        max_anchors=4,
        max_total_bytes=4_000,
        max_anchor_lines=20,
        max_blob_bytes=4_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    refined_context, refinement = FrozenContextRefiner().refine(
        snapshot=snapshot,
        context=context,
        needs=assessment.evidence_needs,
        store=FrozenStore(),
        limits=limits,
        created_at=NOW,
    )

    assert refinement.snapshot_key == snapshot.snapshot_key
    assert refinement.parent_context_sha256 == context.context_sha256
    assert refinement.refined_context_sha256 == refined_context.context_sha256
    assert refinement.evidence_need_ids == ("need-delete-authorization",)
    assert refinement.exhausted is False
    assert refined_context.selected_anchor_count == 2

    added_anchor = next(
        anchor
        for anchor in refined_context.anchors
        if anchor.anchor_id in refinement.added_anchor_ids
    )
    assert added_anchor.path == "api/AuthorizationService.java"
    assert added_anchor.revision_role == "candidate"
    assert "verifyDeleteAuthorization" in added_anchor.text


def test_refiner_records_exhaustion_when_frozen_revisions_have_no_match() -> None:
    """A failed frozen-only search keeps the original context unchanged."""
    snapshot = _snapshot()
    context = _initial_context(snapshot)
    need = FrozenEvidenceNeed(
        need_id="need-missing-authorization",
        category="authorization",
        search_terms=("missingAuthorizationCheck",),
        explanation="Find the saved authorization check for patient deletion.",
        supporting_anchor_ids=("anchor-integration",),
    )
    assessment = Assessment.from_content(
        snapshot_key=snapshot.snapshot_key,
        context_sha256=context.context_sha256,
        reviewed_risk_sha256="5" * 64,
        evidence_envelope_sha256="6" * 64,
        decision="needs_more_frozen_evidence",
        bindings=(),
        evidence_needs=(need,),
        explanation="The available frozen code lacks the authorization check.",
        generated_at=NOW,
        validated_at=NOW,
    )

    class EmptyFrozenStore:
        """Expose no Java files from any of the four frozen revisions."""

        def __init__(self) -> None:
            self.checked_revisions: list[str] = []

        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            self.checked_revisions.append(commit_sha)
            return ()

        def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
            raise AssertionError("No blob should be opened when every tree is empty.")

    store = EmptyFrozenStore()
    limits = ContextLimits(
        max_files=2,
        max_anchors=4,
        max_total_bytes=4_000,
        max_anchor_lines=20,
        max_blob_bytes=4_000,
        max_search_identifiers=10,
        max_hits_per_identifier=5,
    )

    refined_context, refinement = FrozenContextRefiner().refine(
        snapshot=snapshot,
        context=context,
        needs=assessment.evidence_needs,
        store=store,
        limits=limits,
        created_at=NOW,
    )

    assert refined_context == context
    assert refinement.exhausted is True
    assert refinement.parent_context_sha256 == context.context_sha256
    assert refinement.refined_context_sha256 == context.context_sha256
    assert refinement.added_anchor_ids == ()
    assert store.checked_revisions == [
        snapshot.candidate_sha,
        snapshot.head_sha,
        snapshot.base_sha,
        snapshot.merge_base_sha,
    ]
