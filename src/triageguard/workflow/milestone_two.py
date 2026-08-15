"""One-way state machine for Milestone 2 OpenMRS Core PR analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

from triageguard.analysis.context import ContextLimits
from triageguard.config import Settings
from triageguard.contracts import (
    apply_gherkin_text_edit,
)
from triageguard.contracts import (
    approve_gherkin as approve_gherkin_candidate,
)
from triageguard.contracts import (
    generate_gherkin as request_gherkin_candidate,
)
from triageguard.domain import (
    ContextBundle,
    DiffArtifact,
    GherkinApproval,
    GherkinCandidate,
    HumanReviewedRisk,
    MilestoneTwoRunRecord,
    MilestoneTwoStatus,
    PullRequestSnapshot,
    RiskAssessment,
    RiskAssessmentDraft,
    SnapshotFreshness,
)
from triageguard.hypotheses import (
    RiskGroundingReport,
    create_human_review,
    generate_risk_assessment,
    validate_risk_assessment,
)
from triageguard.llm import ModelResponse, StructuredModelGateway
from triageguard.research import ArtifactRecorder, RunHandle, RunOwnership


class MilestoneTwoTransitionError(RuntimeError):
    """The user attempted a workflow action before its required earlier step."""


class _State(str, Enum):
    """The only valid one-way states for a Milestone 2 analysis run."""

    NEW = "new"
    PREPARED = "prepared"
    RISKS_READY = "risks_ready"
    RISK_APPROVED = "risk_approved"
    GHERKIN_READY = "gherkin_ready"
    STALE = "stale"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class PreparedPullRequest:
    """The complete frozen evidence package for one supported pull request."""

    snapshot: PullRequestSnapshot
    diffs: tuple[DiffArtifact, DiffArtifact, DiffArtifact]
    context: ContextBundle


class _SnapshotAcquirer(Protocol):
    """The narrow snapshot operations used during workflow analysis."""

    def acquire(self, pr_url: str) -> PullRequestSnapshot:
        """Freeze the exact supported pull-request identity."""

    def recheck(self, snapshot: PullRequestSnapshot) -> SnapshotFreshness:
        """Report whether this frozen snapshot is still current."""


class _DiffBuilder(Protocol):
    """The narrow three-diff operation used during workflow preparation."""

    def build_all(
        self,
        snapshot: PullRequestSnapshot,
    ) -> tuple[DiffArtifact, DiffArtifact, DiffArtifact]:
        """Build author, integration, and base-drift artifacts."""


class _ContextBuilder(Protocol):
    """The narrow bounded-context operation used during workflow preparation."""

    def build(
        self,
        *,
        snapshot: PullRequestSnapshot,
        diffs: tuple[DiffArtifact, DiffArtifact, DiffArtifact],
        store: object,
        limits: ContextLimits,
    ) -> ContextBundle:
        """Build the bounded evidence catalog for this frozen snapshot."""


class MilestoneTwoWorkflow:
    """Coordinate one human-gated Milestone 2 pull-request analysis run."""

    def __init__(
        self,
        *,
        run_id: str,
        settings: Settings,
        recorder: ArtifactRecorder,
        snapshot_acquirer: _SnapshotAcquirer,
        diff_builder: _DiffBuilder,
        context_builder: _ContextBuilder,
        store: object,
        gateway: StructuredModelGateway,
    ) -> None:
        """Create an empty run before any PR, Git, or model operation occurs."""
        self._settings = settings
        self._recorder = recorder
        self._snapshot_acquirer = snapshot_acquirer
        self._diff_builder = diff_builder
        self._context_builder = context_builder
        self._store = store
        self._gateway = gateway

        self._started_at = datetime.now(UTC)
        ownership = RunOwnership.issue(run_id)
        self._run_handle = recorder.start_run(run_id, ownership)
        self._state = _State.NEW
        self._prepared: PreparedPullRequest | None = None
        self._risk_draft: RiskAssessmentDraft | None = None
        self._risk_response: ModelResponse | None = None
        self._risk_assessment: RiskAssessment | None = None
        self._risk_grounding_report: RiskGroundingReport | None = None
        self._freshness: SnapshotFreshness | None = None
        self._human_reviewed_risk: HumanReviewedRisk | None = None
        self._gherkin_candidate: GherkinCandidate | None = None
        self._gherkin_response: ModelResponse | None = None
        self._gherkin_approval: GherkinApproval | None = None
        self._terminal_record: MilestoneTwoRunRecord | None = None

    @property
    def run_handle(self) -> RunHandle:
        """Return the authenticated recorder handle for this analysis run."""
        return self._run_handle

    @property
    def prepared_pull_request(self) -> PreparedPullRequest | None:
        """Return the frozen evidence package after successful preparation."""
        return self._prepared

    @property
    def risk_assessment(self) -> RiskAssessment | None:
        """Return the locally grounded assessment after successful proposal."""
        return self._risk_assessment

    @property
    def human_reviewed_risk(self) -> HumanReviewedRisk | None:
        """Return the selected human-reviewed risk after approval."""
        return self._human_reviewed_risk

    @property
    def gherkin_candidate(self) -> GherkinCandidate | None:
        """Return the locally validated Gherkin candidate after generation."""
        return self._gherkin_candidate

    @property
    def gherkin_approval(self) -> GherkinApproval | None:
        """Return the local approval after terminal Gherkin acceptance."""
        return self._gherkin_approval

    @property
    def terminal_record(self) -> MilestoneTwoRunRecord | None:
        """Return the sealed terminal record after workflow finalization."""
        return self._terminal_record

    def prepare_pr(self, pr_url: str) -> PreparedPullRequest:
        """Freeze one PR and build all evidence required before model generation."""
        if self._state is not _State.NEW:
            raise MilestoneTwoTransitionError(
                "A pull request has already been prepared for this workflow."
            )

        snapshot = self._snapshot_acquirer.acquire(pr_url)
        diffs = self._diff_builder.build_all(snapshot)
        context = self._context_builder.build(
            snapshot=snapshot,
            diffs=diffs,
            store=self._store,
            limits=ContextLimits.from_settings(self._settings),
        )

        prepared = PreparedPullRequest(
            snapshot=snapshot,
            diffs=diffs,
            context=context,
        )
        self._prepared = prepared
        self._state = _State.PREPARED
        return prepared

    def propose_risks(self) -> RiskAssessment:
        """Generate risks from frozen evidence and require local grounding."""
        if self._state is not _State.PREPARED:
            raise MilestoneTwoTransitionError(
                "Cannot propose risks: prepare a pull request before continuing."
            )

        prepared = self._require_prepared("propose risks")
        draft, response = generate_risk_assessment(
            snapshot=prepared.snapshot,
            diffs=prepared.diffs,
            context=prepared.context,
            gateway=self._gateway,
        )
        assessment, grounding_report = validate_risk_assessment(
            draft=draft,
            snapshot=prepared.snapshot,
            context=prepared.context,
        )
        if assessment is None:
            raise MilestoneTwoTransitionError(
                "Cannot propose risks: local risk grounding rejected the model output."
            )

        self._risk_draft = draft
        self._risk_response = response
        self._risk_assessment = assessment
        self._risk_grounding_report = grounding_report
        self._state = _State.RISKS_READY
        return assessment

    def approve_risk(
        self,
        hypothesis_id: str,
        edits: Mapping[str, object],
        selected_anchor_ids: Sequence[str],
    ) -> HumanReviewedRisk:
        """Recheck freshness, then record one human-approved grounded risk."""
        if self._state is not _State.RISKS_READY or self._risk_assessment is None:
            raise MilestoneTwoTransitionError(
                "Cannot approve a risk: propose risks before continuing."
            )

        prepared = self._require_prepared("approve a risk")
        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed after preparation."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: approval requires a current snapshot."
            )

        review = create_human_review(
            assessment=self._risk_assessment,
            hypothesis_id=hypothesis_id,
            edits=dict(edits),
            selected_anchor_ids=tuple(selected_anchor_ids),
            reviewed_at=datetime.now(UTC),
        )
        self._human_reviewed_risk = review
        self._state = _State.RISK_APPROVED
        return review

    def generate_gherkin(self) -> GherkinCandidate:
        """Recheck freshness and generate one candidate for the approved risk."""
        prepared = self._require_prepared("generate Gherkin")
        if self._state is not _State.RISK_APPROVED or self._human_reviewed_risk is None:
            raise MilestoneTwoTransitionError(
                "Cannot generate Gherkin: approve a risk before continuing."
            )
        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed after risk approval."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: Gherkin generation requires "
                "a current snapshot."
            )

        candidate, response = request_gherkin_candidate(
            human_review=self._human_reviewed_risk,
            gateway=self._gateway,
        )
        self._gherkin_candidate = candidate
        self._gherkin_response = response
        self._state = _State.GHERKIN_READY
        return candidate

    def approve_gherkin(self, text: str) -> MilestoneTwoRunRecord:
        """Recheck, validate a human edit, and seal final Gherkin evidence."""
        prepared = self._require_prepared("approve Gherkin")
        if (
            self._state is not _State.GHERKIN_READY
            or self._human_reviewed_risk is None
            or self._gherkin_candidate is None
        ):
            raise MilestoneTwoTransitionError(
                "Cannot approve Gherkin: generate Gherkin before continuing."
            )

        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed before Gherkin approval."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: Gherkin approval requires "
                "a current snapshot."
            )

        edited_candidate = apply_gherkin_text_edit(
            candidate=self._gherkin_candidate,
            text=text,
            human_review=self._human_reviewed_risk,
        )
        approval = approve_gherkin_candidate(
            candidate=edited_candidate,
            human_review=self._human_reviewed_risk,
            approved_at=datetime.now(UTC),
        )
        record = MilestoneTwoRunRecord(
            run_id=self._run_handle.run_id,
            snapshot=prepared.snapshot,
            status=MilestoneTwoStatus.APPROVED_GHERKIN,
            reason_code="gherkin_approved",
            explanation="A human approved the risk-bound Gherkin scenario.",
            started_at=self._started_at,
            finished_at=datetime.now(UTC),
            freshness=freshness,
            risk_assessment=self._risk_assessment,
            human_reviewed_risk=self._human_reviewed_risk,
            gherkin_candidate=edited_candidate,
            gherkin_approval=approval,
        )
        self._recorder.record_event(
            self._run_handle,
            "gherkin_approved",
            {
                "id": approval.candidate_id,
                "gherkin_sha256": approval.candidate_sha256,
            },
        )
        self._recorder.finalize_run(self._run_handle, record)

        self._gherkin_candidate = edited_candidate
        self._gherkin_approval = approval
        self._terminal_record = record
        self._state = _State.FINALIZED
        return record

    def finish_without_risk(self) -> MilestoneTwoRunRecord:
        """Acknowledge a supported non-risk outcome and seal its terminal record."""
        prepared = self._require_prepared("finish without risk")
        if self._state is not _State.RISKS_READY or self._risk_assessment is None:
            raise MilestoneTwoTransitionError(
                "Cannot finish without risk: propose risks before continuing."
            )

        assessment = self._risk_assessment
        if assessment.outcome not in {
            "no_meaningful_security_risk_found",
            "insufficient_context_to_assess",
        }:
            raise MilestoneTwoTransitionError(
                "Cannot finish without risk: the assessment is not a supported "
                "non-risk outcome."
            )

        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed before finalization."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: finalization requires "
                "a current snapshot."
            )

        if assessment.outcome == "no_meaningful_security_risk_found":
            status = MilestoneTwoStatus.NO_MEANINGFUL_SECURITY_RISK_FOUND
            reason_code = "no_meaningful_security_risk_found"
            explanation = assessment.rationale
        else:
            status = MilestoneTwoStatus.INSUFFICIENT_CONTEXT_TO_ASSESS
            reason_code = assessment.reason_code
            explanation = "The bounded evidence was insufficient for a risk assessment."

        if not explanation or not reason_code:
            raise MilestoneTwoTransitionError(
                "Cannot finish without risk: the assessment lacks "
                "its required explanation or reason code."
            )

        record = MilestoneTwoRunRecord(
            run_id=self._run_handle.run_id,
            snapshot=prepared.snapshot,
            status=status,
            reason_code=reason_code,
            explanation=explanation,
            started_at=self._started_at,
            finished_at=datetime.now(UTC),
            freshness=freshness,
            risk_assessment=assessment,
            human_reviewed_risk=None,
            gherkin_candidate=None,
            gherkin_approval=None,
        )
        self._recorder.finalize_run(self._run_handle, record)

        self._terminal_record = record
        self._state = _State.FINALIZED
        return record

    def freshness(self) -> SnapshotFreshness:
        """Recheck and return the frozen PR's currentness without model activity."""
        prepared = self._require_prepared("check freshness")
        if self._state is _State.FINALIZED:
            raise MilestoneTwoTransitionError(
                "Cannot check freshness: the workflow is already finalized."
            )
        if self._state is _State.STALE:
            raise MilestoneTwoTransitionError(
                "Cannot check freshness: the workflow is already stale."
            )

        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
        return freshness

    def _require_prepared(self, action: str) -> PreparedPullRequest:
        """Require the frozen snapshot/evidence stage before downstream actions."""
        if self._state is _State.NEW or self._prepared is None:
            raise MilestoneTwoTransitionError(
                f"Cannot {action}: prepare a pull request before continuing."
            )
        return self._prepared
