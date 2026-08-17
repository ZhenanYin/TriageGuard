"""Tests for bounded retrieval from the already-frozen evidence boundary."""

import hashlib
from datetime import UTC, datetime

import pytest

from triageguard.analysis.context import ContextLimits
from triageguard.analysis.refinement import FrozenContextRefiner
from triageguard.domain import (
    ContextAnchor,
    ContextBundle,
    EvidenceRefinementResult,
    FrozenEvidenceNeed,
    PullRequestSnapshot,
)
from triageguard.evidence import FrozenEvidenceResolver
from triageguard.sources.git import GitTreeEntry

NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _snapshot() -> PullRequestSnapshot:
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


def _anchor(
    snapshot: PullRequestSnapshot,
    *,
    anchor_id: str,
    text: str,
    relation: str,
    path: str,
) -> ContextAnchor:
    return ContextAnchor(
        anchor_id=anchor_id,
        revision_role="candidate",
        commit_sha=snapshot.candidate_sha,
        blob_sha=hashlib.sha1(path.encode()).hexdigest(),
        path=path,
        java_symbol=None,
        start_line=1,
        end_line=1,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        selection_reason="controlled frozen evidence",
        score_components=(),
        change_relation=relation,
        truncated=False,
    )


def _context(
    snapshot: PullRequestSnapshot,
    *anchors: ContextAnchor,
    max_search_identifiers: int = 10,
) -> ContextBundle:
    return ContextBundle.from_content(
        snapshot_key=snapshot.snapshot_key,
        anchors=anchors,
        selected_file_count=len({anchor.path for anchor in anchors}),
        selected_anchor_count=len(anchors),
        selected_bytes=sum(len(anchor.text.encode()) for anchor in anchors),
        max_files=6,
        max_anchors=10,
        max_bytes=8_000,
        max_anchor_lines=20,
        max_blob_bytes=8_000,
        max_search_identifiers=max_search_identifiers,
        max_hits_per_identifier=5,
        primary_change_represented=True,
    )


def _limits(*, max_search_identifiers: int = 10) -> ContextLimits:
    return ContextLimits(
        max_files=6,
        max_anchors=10,
        max_total_bytes=8_000,
        max_anchor_lines=20,
        max_blob_bytes=8_000,
        max_search_identifiers=max_search_identifiers,
        max_hits_per_identifier=5,
    )


def _need(term: str = "hasPrivilege") -> FrozenEvidenceNeed:
    return FrozenEvidenceNeed(
        need_id=f"need-{term}",
        category="authorization",
        search_terms=(term,),
        explanation="Find the exact frozen authorization decision used here.",
        supporting_anchor_ids=("anchor-visible",),
    )


class _NoTreeReads:
    def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
        raise AssertionError("catalog evidence must be used before opening Git trees")

    def read_blob(self, blob_sha: str, *, max_bytes: int) -> bytes:
        raise AssertionError("catalog evidence must be used before opening blobs")


def test_refinement_prioritizes_an_existing_omitted_anchor() -> None:
    """Removing catalog-first resolution must cause an unnecessary Git search."""
    snapshot = _snapshot()
    visible = _anchor(
        snapshot,
        anchor_id="anchor-visible",
        text="service.deletePatient();\n",
        relation="integration_change",
        path="api/PatientService.java",
    )
    omitted = _anchor(
        snapshot,
        anchor_id="anchor-hidden-has-privilege",
        text="if (context.hasPrivilege(DELETE_PATIENTS)) {\n",
        relation="repository_context",
        path="api/AuthorizationContext.java",
    )
    context = _context(snapshot, visible, omitted)

    resolution = FrozenEvidenceResolver(FrozenContextRefiner()).resolve(
        snapshot=snapshot,
        context=context,
        needs=(_need(),),
        store=_NoTreeReads(),
        limits=_limits(),
        completed_rounds=0,
        max_rounds=2,
        created_at=NOW,
    )

    assert resolution.context == context
    assert resolution.priority_anchor_ids == ("anchor-hidden-has-privilege",)
    assert resolution.added_anchor_ids == ()
    assert resolution.exhausted is False
    assert resolution.reason_code == "catalog_evidence_prioritized"
    assert resolution.round_number == 1


def test_refinement_rejects_more_identifiers_than_the_frozen_search_limit() -> None:
    """Model output cannot multiply repository scans beyond the approved bound."""
    snapshot = _snapshot()
    visible = _anchor(
        snapshot,
        anchor_id="anchor-visible",
        text="service.deletePatient();\n",
        relation="integration_change",
        path="api/PatientService.java",
    )
    need = FrozenEvidenceNeed(
        need_id="need-two-identifiers",
        category="authorization",
        search_terms=("hasPrivilege", "requirePrivilege"),
        explanation="Find the exact frozen authorization decision used here.",
        supporting_anchor_ids=("anchor-visible",),
    )

    with pytest.raises(ValueError, match="identifier limit"):
        FrozenEvidenceResolver(FrozenContextRefiner()).resolve(
            snapshot=snapshot,
            context=_context(snapshot, visible, max_search_identifiers=1),
            needs=(need,),
            store=_NoTreeReads(),
            limits=_limits(max_search_identifiers=1),
            completed_rounds=0,
            max_rounds=2,
            created_at=NOW,
        )


def test_refinement_searches_only_the_four_frozen_snapshot_revisions() -> None:
    """Changing the revision allowlist must expose an out-of-snapshot tree read."""
    snapshot = _snapshot()
    context = _context(
        snapshot,
        _anchor(
            snapshot,
            anchor_id="anchor-visible",
            text="service.deletePatient();\n",
            relation="integration_change",
            path="api/PatientService.java",
        ),
    )
    source = (
        b"class AuthorizationService {\n"
        b"  void verifyDeleteAuthorization() { requirePrivilege(); }\n"
        b"}\n"
    )

    class FrozenStore:
        def __init__(self) -> None:
            self.checked: list[str] = []

        def list_tree(self, commit_sha: str) -> tuple[GitTreeEntry, ...]:
            assert commit_sha in {
                snapshot.merge_base_sha,
                snapshot.base_sha,
                snapshot.head_sha,
                snapshot.candidate_sha,
            }
            self.checked.append(commit_sha)
            if commit_sha != snapshot.head_sha:
                return ()
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
            assert max_bytes == 8_000
            return source

    store = FrozenStore()
    resolution = FrozenEvidenceResolver(FrozenContextRefiner()).resolve(
        snapshot=snapshot,
        context=context,
        needs=(_need("verifyDeleteAuthorization"),),
        store=store,
        limits=_limits(),
        completed_rounds=0,
        max_rounds=2,
        created_at=NOW,
    )

    assert resolution.context.context_sha256 != context.context_sha256
    assert len(resolution.added_anchor_ids) == 1
    assert resolution.reason_code == "frozen_context_extended"
    assert store.checked == [snapshot.candidate_sha, snapshot.head_sha]


def test_refinement_round_limit_exhausts_without_reading_git() -> None:
    """Removing the round guard must cause a forbidden third frozen-code search."""
    snapshot = _snapshot()
    context = _context(
        snapshot,
        _anchor(
            snapshot,
            anchor_id="anchor-visible",
            text="service.deletePatient();\n",
            relation="integration_change",
            path="api/PatientService.java",
        ),
    )

    resolution = FrozenEvidenceResolver(FrozenContextRefiner()).resolve(
        snapshot=snapshot,
        context=context,
        needs=(_need("verifyDeleteAuthorization"),),
        store=_NoTreeReads(),
        limits=_limits(),
        completed_rounds=2,
        max_rounds=2,
        created_at=NOW,
    )

    assert resolution.context == context
    assert resolution.exhausted is True
    assert resolution.reason_code == "refinement_round_limit_reached"
    assert resolution.round_number == 3


@pytest.mark.parametrize(
    "term",
    [" ", "authorization", "../secrets", "find something"],
)
def test_frozen_evidence_need_rejects_non_exact_search_terms(term: str) -> None:
    """Weakening exact-term validation must admit vague or path-like searches."""
    with pytest.raises(ValueError, match="exact code identifier"):
        _need(term)


def test_refinement_result_rejects_a_tampered_content_hash() -> None:
    """Accepting a changed result hash would break durable refinement replay."""
    with pytest.raises(ValueError, match="refinement SHA-256"):
        EvidenceRefinementResult(
            parent_context_sha256="a" * 64,
            successor_context_sha256="a" * 64,
            requested_need_sha256="b" * 64,
            priority_anchor_ids=("anchor-hidden",),
            added_anchor_ids=(),
            round_number=1,
            exhausted=False,
            reason_code="catalog_evidence_prioritized",
            refinement_sha256="c" * 64,
        )
