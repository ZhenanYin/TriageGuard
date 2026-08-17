"""One-way state machine for Milestone 2 OpenMRS Core PR analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import wraps
from threading import RLock
from typing import Protocol

from triageguard.analysis.context import ContextLimits
from triageguard.config import Settings
from triageguard.contracts import (
    GherkinValidationReport,
    apply_gherkin_text_edit,
    validate_edited_gherkin,
    validate_gherkin_candidate,
)
from triageguard.contracts import (
    approve_gherkin as approve_gherkin_candidate,
)
from triageguard.contracts import (
    generate_gherkin as request_gherkin_candidate,
)
from triageguard.contracts.gherkin_generation import (
    GHERKIN_OUTPUT_SCHEMA,
    build_gherkin_evidence,
    build_gherkin_request,
)
from triageguard.domain import (
    ContextBundle,
    DiffArtifact,
    EvidenceRefinementResult,
    FrozenEvidenceNeed,
    GherkinApproval,
    GherkinCandidate,
    HumanReviewedRisk,
    MilestoneTwoRunRecord,
    MilestoneTwoStatus,
    PullRequestSnapshot,
    RiskAssessment,
    RiskAssessmentDraft,
    SnapshotFreshness,
    TestabilityAssessment,
    TestabilityAssessmentDraft,
)
from triageguard.evidence import (
    EvidenceArtifactBinding,
    FrozenEvidenceResolution,
    FrozenEvidenceResolver,
    ModelEvidenceEnvelope,
    validate_envelope_binding,
)
from triageguard.hypotheses import (
    RISK_OUTPUT_SCHEMA,
    RiskGroundingReport,
    build_risk_evidence,
    build_risk_request,
    create_human_review,
    generate_risk_assessment,
    validate_risk_assessment,
)
from triageguard.llm import (
    ModelFailureProvenance,
    ModelGatewayError,
    ModelResponse,
    StructuredModelGateway,
)
from triageguard.llm.request_budget import ProviderRequestBudget
from triageguard.provenance import canonical_json, canonical_sha256
from triageguard.research import ArtifactRecorder, RunHandle, RunOwnership
from triageguard.research.recorder import (
    ArtifactWriteJournal,
    LifecycleEventType,
    TransformationEvent,
)
from triageguard.testability.generator import (
    TESTABILITY_OUTPUT_SCHEMA,
    build_testability_evidence,
    build_testability_request,
    generate_testability_assessment,
)
from triageguard.testability.validator import (
    validate_testability_assessment,
)


class MilestoneTwoTransitionError(RuntimeError):
    """The user attempted a workflow action before its required earlier step."""


class _State(str, Enum):
    """The only valid one-way states for a Milestone 2 analysis run."""

    NEW = "new"
    PREPARED = "prepared"
    RISKS_READY = "risks_ready"
    RISK_APPROVED = "risk_approved"
    TESTABILITY_READY = "testability_ready"
    EVIDENCE_REFINEMENT_REQUIRED = "evidence_refinement_required"
    GHERKIN_READY = "gherkin_ready"
    GHERKIN_VALIDATED = "gherkin_validated"
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


class _EvidenceRefiner(Protocol):
    """Refine only the already frozen evidence for one prepared pull request."""

    def resolve(
        self,
        *,
        snapshot: PullRequestSnapshot,
        context: ContextBundle,
        needs: Sequence[FrozenEvidenceNeed],
        store: object,
        limits: ContextLimits,
        completed_rounds: int,
        max_rounds: int,
        created_at: datetime,
    ) -> FrozenEvidenceResolution:
        """Return one bounded catalog-first frozen-evidence resolution."""


@dataclass(frozen=True)
class MilestoneTwoDependencies:
    """Exact dependencies required to resume one authenticated workflow run."""

    settings: Settings
    recorder: ArtifactRecorder
    snapshot_acquirer: _SnapshotAcquirer
    diff_builder: _DiffBuilder
    context_builder: _ContextBuilder
    store: object
    gateway: StructuredModelGateway
    evidence_refiner: _EvidenceRefiner | None = None


def _with_workflow_lease(method):
    """Serialize one public operation in-process and through the recorder."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with (
            self._operation_lock,
            self._recorder.workflow_lease(self._run_handle),
        ):
            return method(self, *args, **kwargs)

    return wrapped


def _gherkin_evidence_assessment(
    *,
    candidate: GherkinCandidate,
    human_review: HumanReviewedRisk,
    context: ContextBundle,
    generated_at: datetime,
) -> TestabilityAssessment:
    """Turn an unsupported scenario edit into a bounded frozen-code search."""
    anchors_by_id = {anchor.anchor_id: anchor for anchor in context.anchors}
    candidate_anchor_ids = tuple(
        anchor_id
        for binding in candidate.step_evidence_bindings
        for anchor_id in binding.anchor_ids
        if anchor_id in anchors_by_id
        and anchors_by_id[anchor_id].change_relation == "integration_change"
    )
    review_anchor_ids = tuple(
        anchor_id
        for anchor_id in human_review.reviewed_risk.citation_anchor_ids
        if anchor_id in anchors_by_id
        and anchors_by_id[anchor_id].change_relation == "integration_change"
    )
    supporting_anchor_ids = tuple(
        dict.fromkeys(candidate_anchor_ids + review_anchor_ids)
    )
    search_terms = tuple(dict.fromkeys(human_review.reviewed_risk.code_identifiers))
    if not supporting_anchor_ids or not search_terms:
        raise MilestoneTwoTransitionError(
            "Cannot refine frozen evidence: the approved risk does not provide "
            "a bound code search target."
        )

    need = FrozenEvidenceNeed(
        need_id=(
            "gherkin-edit-"
            + canonical_sha256(
                {
                    "candidate": canonical_sha256(candidate.model_dump(mode="json")),
                    "reviewed_risk": human_review.reviewed_content_sha256,
                    "context": context.context_sha256,
                }
            )[:16]
        ),
        category="entry_point",
        search_terms=search_terms,
        explanation=(
            "The edited scenario needs additional frozen code evidence for an "
            "entry point tied to the approved risk before it can be supported."
        ),
        supporting_anchor_ids=supporting_anchor_ids,
    )
    return TestabilityAssessment.from_content(
        snapshot_key=human_review.snapshot_key,
        context_sha256=context.context_sha256,
        reviewed_risk_sha256=human_review.reviewed_content_sha256,
        evidence_envelope_sha256=candidate.evidence_envelope_sha256,
        decision="needs_more_frozen_evidence",
        bindings=(),
        evidence_needs=(need,),
        explanation=(
            "The edited scenario is not fully supported by the current frozen "
            "code evidence."
        ),
        generated_at=generated_at,
        validated_at=generated_at,
    )


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
        evidence_refiner: _EvidenceRefiner | None = None,
        _run_handle: RunHandle | None = None,
    ) -> None:
        """Create an empty run before any PR, Git, or model operation occurs."""
        self._settings = settings
        self._recorder = recorder
        self._snapshot_acquirer = snapshot_acquirer
        self._diff_builder = diff_builder
        self._context_builder = context_builder
        self._store = store
        self._gateway = gateway
        if evidence_refiner is not None and not hasattr(evidence_refiner, "resolve"):
            self._evidence_refiner = FrozenEvidenceResolver(evidence_refiner)
        else:
            self._evidence_refiner = evidence_refiner
        self._operation_lock = RLock()

        self._started_at = datetime.now(UTC)
        if _run_handle is None:
            ownership = RunOwnership.issue(run_id)
            self._run_handle = recorder.start_run(run_id, ownership)
        else:
            if _run_handle.run_id != run_id:
                raise ValueError("resumed run handle must match run_id")
            self._run_handle = _run_handle
        self._state = _State.NEW
        self._prepared: PreparedPullRequest | None = None
        self._risk_evidence_envelope: ModelEvidenceEnvelope | None = None
        self._risk_draft: RiskAssessmentDraft | None = None
        self._risk_response: ModelResponse | None = None
        self._risk_assessment: RiskAssessment | None = None
        self._risk_grounding_report: RiskGroundingReport | None = None
        self._risk_failure: ModelFailureProvenance | None = None
        self._freshness: SnapshotFreshness | None = None
        self._human_reviewed_risk: HumanReviewedRisk | None = None
        self._testability_draft: TestabilityAssessmentDraft | None = None
        self._testability_evidence_envelope: ModelEvidenceEnvelope | None = None
        self._testability_response: ModelResponse | None = None
        self._testability_assessment: TestabilityAssessment | None = None
        self._context_refinements: list[EvidenceRefinementResult] = []
        self._refinement_priority_anchor_ids: tuple[str, ...] = ()
        self._analysis_round = 0
        self._gherkin_candidate: GherkinCandidate | None = None
        self._gherkin_evidence_envelope: ModelEvidenceEnvelope | None = None
        self._gherkin_response: ModelResponse | None = None
        self._gherkin_validation_report: GherkinValidationReport | None = None
        self._validated_gherkin_text: str | None = None
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
    def risk_evidence_envelope(self) -> ModelEvidenceEnvelope | None:
        """Return the exact frozen evidence visible to risk generation."""
        return self._risk_evidence_envelope

    @property
    def risk_failure(self) -> ModelFailureProvenance | None:
        """Return the latest safe model-failure provenance for a retryable stage."""
        return self._risk_failure

    @property
    def human_reviewed_risk(self) -> HumanReviewedRisk | None:
        """Return the selected human-reviewed risk after approval."""
        return self._human_reviewed_risk

    @property
    def testability_assessment(self) -> TestabilityAssessment | None:
        """Return the current locally validated frozen-evidence decision."""
        return self._testability_assessment

    @property
    def testability_evidence_envelope(self) -> ModelEvidenceEnvelope | None:
        """Return the exact frozen evidence visible to testability generation."""
        return self._testability_evidence_envelope

    @property
    def context_refinements(self) -> tuple[EvidenceRefinementResult, ...]:
        """Return the immutable frozen-evidence refinements in this run."""
        return tuple(self._context_refinements)

    @property
    def gherkin_candidate(self) -> GherkinCandidate | None:
        """Return the locally validated Gherkin candidate after generation."""
        return self._gherkin_candidate

    @property
    def gherkin_evidence_envelope(self) -> ModelEvidenceEnvelope | None:
        """Return the exact frozen evidence visible to Gherkin generation."""
        return self._gherkin_evidence_envelope

    @property
    def gherkin_validation_report(self) -> GherkinValidationReport | None:
        """Return the latest local decision about the edited Gherkin text."""
        return self._gherkin_validation_report

    @property
    def gherkin_approval(self) -> GherkinApproval | None:
        """Return the local approval after terminal Gherkin acceptance."""
        return self._gherkin_approval

    @property
    def terminal_record(self) -> MilestoneTwoRunRecord | None:
        """Return the sealed terminal record after workflow finalization."""
        return self._terminal_record

    @_with_workflow_lease
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
        if self._is_typed_prepared(prepared):
            self._persist_prepared(prepared)
        self._prepared = prepared
        self._state = _State.PREPARED
        return prepared

    @_with_workflow_lease
    def propose_risks(self) -> RiskAssessment:
        """Generate risks from frozen evidence and require local grounding."""
        if self._state is not _State.PREPARED:
            raise MilestoneTwoTransitionError(
                "Cannot propose risks: prepare a pull request before continuing."
            )

        prepared = self._require_prepared("propose risks")
        if self._risk_evidence_envelope is None:
            envelope_result = build_risk_evidence(
                snapshot=prepared.snapshot,
                diffs=prepared.diffs,
                context=prepared.context,
                budget=ProviderRequestBudget.from_settings(self._settings),
                priority_anchor_ids=self._refinement_priority_anchor_ids,
            )
            evidence_envelope = envelope_result.envelope
            if self._is_typed_prepared(prepared):
                self._persist_risk_evidence_envelope(
                    prepared=prepared,
                    evidence_envelope=evidence_envelope,
                )
            self._risk_evidence_envelope = evidence_envelope
        else:
            evidence_envelope = self._risk_evidence_envelope
        if self._risk_draft is not None and self._risk_response is not None:
            draft = self._risk_draft
            response = self._risk_response
            if (
                draft.snapshot_key != prepared.snapshot.snapshot_key
                or draft.context_sha256 != prepared.context.context_sha256
                or draft.evidence_envelope_sha256 != evidence_envelope.envelope_sha256
            ):
                raise MilestoneTwoTransitionError(
                    "A saved risk response is not bound to the prepared evidence."
                )
        else:
            try:
                draft, response = generate_risk_assessment(
                    snapshot=prepared.snapshot,
                    diffs=prepared.diffs,
                    context=prepared.context,
                    evidence_envelope=evidence_envelope,
                    gateway=self._gateway,
                )
            except ModelGatewayError as error:
                if error.provenance is not None:
                    self._record_risk_failure(
                        prepared=prepared,
                        failure=error.provenance,
                    )
                raise
            if self._is_typed_prepared(prepared):
                self._persist_risk_generation(
                    prepared=prepared,
                    draft=draft,
                    response=response,
                )
            self._risk_failure = None

        assessment, grounding_report = validate_risk_assessment(
            draft=draft,
            snapshot=prepared.snapshot,
            context=prepared.context,
            evidence_envelope=evidence_envelope,
        )
        if assessment is None:
            raise MilestoneTwoTransitionError(
                "Cannot propose risks: local risk grounding rejected the model output."
            )

        self._risk_draft = draft
        self._risk_response = response
        self._risk_assessment = assessment
        self._risk_grounding_report = grounding_report
        if getattr(assessment, "outcome", "risks_proposed") == (
            "insufficient_context_to_assess"
        ):
            self._state = _State.EVIDENCE_REFINEMENT_REQUIRED
        else:
            self._state = _State.RISKS_READY
        return assessment

    @_with_workflow_lease
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
        if self._is_typed_prepared(prepared):
            self._persist_human_review(
                prepared=prepared,
                assessment=self._risk_assessment,
                review=review,
                freshness=freshness,
            )
        self._human_reviewed_risk = review
        self._state = _State.RISK_APPROVED
        return review

    @_with_workflow_lease
    def assess_testability(self) -> TestabilityAssessment:
        """Assess whether frozen code evidence can support an executable scenario."""
        if self._state is not _State.RISK_APPROVED or self._human_reviewed_risk is None:
            raise MilestoneTwoTransitionError(
                "Cannot assess testability: approve a risk before continuing."
            )

        prepared = self._require_prepared("assess testability")
        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed before "
                "testability assessment."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: testability assessment requires "
                "a current snapshot."
            )

        comparison_bindings = self._comparison_bindings(prepared)
        if self._is_typed_prepared(prepared):
            envelope_result = build_testability_evidence(
                human_review=self._human_reviewed_risk,
                context=prepared.context,
                comparison_bindings=comparison_bindings,
                budget=ProviderRequestBudget.from_settings(self._settings),
            )
            evidence_envelope = envelope_result.envelope
            self._persist_testability_evidence_envelope(
                prepared=prepared,
                human_review=self._human_reviewed_risk,
                evidence_envelope=evidence_envelope,
            )
        else:
            evidence_envelope = object()
        self._testability_evidence_envelope = evidence_envelope

        draft, response = generate_testability_assessment(
            human_review=self._human_reviewed_risk,
            context=prepared.context,
            comparison_bindings=comparison_bindings,
            evidence_envelope=evidence_envelope,
            gateway=self._gateway,
        )
        assessment, _report = validate_testability_assessment(
            draft=draft,
            human_review=self._human_reviewed_risk,
            context=prepared.context,
            comparison_bindings=comparison_bindings,
            evidence_envelope=evidence_envelope,
        )
        if assessment is None:
            raise MilestoneTwoTransitionError(
                "Cannot assess testability: local validation rejected the model output."
            )

        if self._is_typed_prepared(prepared):
            self._persist_testability_assessment(
                prepared=prepared,
                human_review=self._human_reviewed_risk,
                draft=draft,
                assessment=assessment,
                response=response,
                freshness=freshness,
            )

        self._testability_draft = draft
        self._testability_response = response
        self._testability_assessment = assessment
        if assessment.decision == "testable_from_frozen_evidence":
            self._state = _State.TESTABILITY_READY
        else:
            self._state = _State.EVIDENCE_REFINEMENT_REQUIRED
        return assessment

    @_with_workflow_lease
    def refine_frozen_evidence(self) -> EvidenceRefinementResult:
        """Replace the context only with bounded code from saved snapshots."""
        if self._state is not _State.EVIDENCE_REFINEMENT_REQUIRED:
            raise MilestoneTwoTransitionError(
                "Cannot refine frozen evidence: a model stage must request "
                "structured frozen evidence before continuing."
            )
        if self._evidence_refiner is None:
            raise MilestoneTwoTransitionError(
                "Cannot refine frozen evidence: no frozen-evidence refiner "
                "is configured."
            )
        if self._context_refinements and self._context_refinements[-1].exhausted:
            raise MilestoneTwoTransitionError(
                "Cannot refine frozen evidence: the bounded search is already "
                "exhausted."
            )

        prepared = self._require_prepared("refine frozen evidence")
        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed before frozen "
                "evidence refinement."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: frozen evidence refinement "
                "requires a current snapshot."
            )

        needs = self._active_refinement_needs()
        resolution = self._evidence_refiner.resolve(
            snapshot=prepared.snapshot,
            context=prepared.context,
            needs=needs,
            store=self._store,
            limits=ContextLimits.from_settings(self._settings),
            completed_rounds=len(self._context_refinements),
            max_rounds=self._settings.max_model_evidence_rounds,
            created_at=datetime.now(UTC),
        )
        refined_context = resolution.context
        refinement = resolution.refinement
        if self._is_typed_prepared(prepared):
            self._persist_evidence_refinement(
                prepared=prepared,
                needs=needs,
                successor_context=refined_context,
                refinement=refinement,
                freshness=freshness,
            )
        self._context_refinements.append(refinement)
        if refinement.exhausted:
            return refinement

        self._prepared = PreparedPullRequest(
            snapshot=prepared.snapshot,
            diffs=prepared.diffs,
            context=refined_context,
        )
        self._refinement_priority_anchor_ids = tuple(
            dict.fromkeys(
                (*refinement.priority_anchor_ids, *refinement.added_anchor_ids)
            )
        )
        self._analysis_round += 1
        self._risk_evidence_envelope = None
        self._risk_draft = None
        self._risk_response = None
        self._risk_assessment = None
        self._risk_grounding_report = None
        self._risk_failure = None
        self._human_reviewed_risk = None
        self._testability_evidence_envelope = None
        self._testability_draft = None
        self._testability_response = None
        self._testability_assessment = None
        self._gherkin_evidence_envelope = None
        self._gherkin_candidate = None
        self._gherkin_response = None
        self._gherkin_validation_report = None
        self._validated_gherkin_text = None
        self._gherkin_approval = None
        self._state = _State.PREPARED
        return refinement

    def _active_refinement_needs(self) -> tuple[FrozenEvidenceNeed, ...]:
        """Return the validated needs from the exact stage requesting refinement."""
        if (
            self._risk_assessment is not None
            and getattr(self._risk_assessment, "outcome", None)
            == "insufficient_context_to_assess"
        ):
            needs = tuple(self._risk_assessment.evidence_needs)
        elif (
            self._testability_assessment is not None
            and self._testability_assessment.decision == "needs_more_frozen_evidence"
        ):
            needs = tuple(self._testability_assessment.evidence_needs)
        else:
            raise MilestoneTwoTransitionError(
                "Cannot refine frozen evidence: no structured evidence need is active."
            )
        if not needs:
            raise MilestoneTwoTransitionError(
                "Cannot refine frozen evidence: the active stage supplied no needs."
            )
        return needs

    @_with_workflow_lease
    def finish_with_insufficient_frozen_evidence(self) -> MilestoneTwoRunRecord:
        """Seal an exhausted frozen-evidence search without treating it as safe."""
        risk_exhausted = (
            self._risk_assessment is not None
            and getattr(self._risk_assessment, "outcome", None)
            == "insufficient_context_to_assess"
            and self._human_reviewed_risk is None
            and self._testability_assessment is None
        )
        testability_exhausted = (
            self._risk_assessment is not None
            and self._human_reviewed_risk is not None
            and self._testability_assessment is not None
            and self._testability_assessment.decision == "needs_more_frozen_evidence"
        )
        if (
            self._state is not _State.EVIDENCE_REFINEMENT_REQUIRED
            or not self._context_refinements
            or not self._context_refinements[-1].exhausted
            or not (risk_exhausted or testability_exhausted)
        ):
            raise MilestoneTwoTransitionError(
                "Cannot finish with insufficient frozen evidence: an exhausted "
                "frozen-evidence refinement is required before continuing."
            )

        prepared = self._require_prepared("finish with insufficient frozen evidence")
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

        if risk_exhausted:
            status = MilestoneTwoStatus.INSUFFICIENT_CONTEXT_TO_ASSESS
            reason_code = self._risk_assessment.reason_code
            explanation = (
                "Insufficient bounded frozen code evidence to assess this change."
            )
        else:
            status = MilestoneTwoStatus.INSUFFICIENT_FROZEN_EVIDENCE_FOR_SCENARIO
            reason_code = "insufficient_frozen_evidence_for_scenario"
            explanation = (
                "Insufficient frozen code evidence to design an executable scenario."
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
            risk_assessment=self._risk_assessment,
            human_reviewed_risk=self._human_reviewed_risk,
            testability_assessment=self._testability_assessment,
            context_refinements=tuple(self._context_refinements),
            gherkin_candidate=None,
            gherkin_approval=None,
        )
        self._write_final_measurements(record)
        self._recorder.finalize_run(self._run_handle, record)

        self._terminal_record = record
        self._state = _State.FINALIZED
        return record

    @_with_workflow_lease
    def generate_gherkin(self) -> GherkinCandidate:
        """Recheck freshness and generate one candidate for the approved risk."""
        prepared = self._require_prepared("generate Gherkin")
        if (
            self._state is not _State.TESTABILITY_READY
            or self._human_reviewed_risk is None
            or self._testability_assessment is None
        ):
            raise MilestoneTwoTransitionError(
                "Cannot generate Gherkin: assess testability after approving "
                "a risk before continuing."
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

        comparison_bindings = self._comparison_bindings(prepared)
        if self._is_typed_prepared(prepared):
            envelope_result = build_gherkin_evidence(
                human_review=self._human_reviewed_risk,
                testability_assessment=self._testability_assessment,
                context=prepared.context,
                comparison_bindings=comparison_bindings,
                budget=ProviderRequestBudget.from_settings(self._settings),
            )
            evidence_envelope = envelope_result.envelope
            self._persist_gherkin_evidence_envelope(
                prepared=prepared,
                human_review=self._human_reviewed_risk,
                testability_assessment=self._testability_assessment,
                evidence_envelope=evidence_envelope,
            )
        else:
            evidence_envelope = object()
        self._gherkin_evidence_envelope = evidence_envelope

        candidate, response = request_gherkin_candidate(
            human_review=self._human_reviewed_risk,
            testability_assessment=self._testability_assessment,
            context=prepared.context,
            comparison_bindings=comparison_bindings,
            evidence_envelope=evidence_envelope,
            gateway=self._gateway,
        )
        generated_report = validate_gherkin_candidate(
            candidate=candidate,
            human_review=self._human_reviewed_risk,
            testability_assessment=self._testability_assessment,
            context=prepared.context,
            comparison_bindings=comparison_bindings,
            evidence_envelope=evidence_envelope,
        )
        if not generated_report.approved:
            raise MilestoneTwoTransitionError(
                "Cannot generate Gherkin: local validation rejected the "
                "generated scenario."
            )

        if self._is_typed_prepared(prepared):
            self._persist_gherkin_generation(
                prepared=prepared,
                human_review=self._human_reviewed_risk,
                testability_assessment=self._testability_assessment,
                candidate=candidate,
                response=response,
                freshness=freshness,
            )
        self._gherkin_candidate = candidate
        self._gherkin_response = response
        self._gherkin_validation_report = generated_report
        self._validated_gherkin_text = candidate.gherkin_text
        self._state = _State.GHERKIN_READY
        return candidate

    @_with_workflow_lease
    def validate_edited_gherkin(self, text: str) -> GherkinValidationReport:
        """Classify a reviewer edit before it may become final evidence."""
        prepared = self._require_prepared("validate edited Gherkin")
        if (
            self._state is not _State.GHERKIN_READY
            or self._human_reviewed_risk is None
            or self._testability_assessment is None
            or self._gherkin_evidence_envelope is None
            or self._gherkin_candidate is None
        ):
            raise MilestoneTwoTransitionError(
                "Cannot validate Gherkin: generate Gherkin before continuing."
            )
        if text == self._gherkin_candidate.gherkin_text:
            raise MilestoneTwoTransitionError(
                "Cannot validate Gherkin: change the generated Gherkin "
                "before continuing."
            )

        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
            raise MilestoneTwoTransitionError(
                "snapshot_stale: the pull request changed before Gherkin validation."
            )
        if freshness.status != "current":
            raise MilestoneTwoTransitionError(
                "snapshot_currentness_unknown: Gherkin validation requires "
                "a current snapshot."
            )

        report = validate_edited_gherkin(
            candidate=self._gherkin_candidate,
            text=text,
            human_review=self._human_reviewed_risk,
            testability_assessment=self._testability_assessment,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=self._gherkin_evidence_envelope,
        )
        self._gherkin_validation_report = report

        if report.decision == "valid_evidence_bound_gherkin":
            source_candidate = self._gherkin_candidate
            edited_candidate = apply_gherkin_text_edit(
                candidate=source_candidate,
                text=text,
                human_review=self._human_reviewed_risk,
                testability_assessment=self._testability_assessment,
                context=prepared.context,
                comparison_bindings=self._comparison_bindings(prepared),
                evidence_envelope=self._gherkin_evidence_envelope,
            )
            if self._is_typed_prepared(prepared):
                self._persist_gherkin_validation(
                    prepared=prepared,
                    human_review=self._human_reviewed_risk,
                    source_candidate=source_candidate,
                    candidate=edited_candidate,
                    validation=report,
                    freshness=freshness,
                )
            self._gherkin_candidate = edited_candidate
            self._validated_gherkin_text = text
            self._state = _State.GHERKIN_VALIDATED
        elif report.decision == "needs_more_frozen_evidence":
            self._testability_draft = None
            self._testability_response = None
            evidence_assessment = _gherkin_evidence_assessment(
                candidate=self._gherkin_candidate,
                human_review=self._human_reviewed_risk,
                context=prepared.context,
                generated_at=datetime.now(UTC),
            )
            if self._is_typed_prepared(prepared):
                self._persist_gherkin_evidence_gap(
                    prepared=prepared,
                    human_review=self._human_reviewed_risk,
                    candidate=self._gherkin_candidate,
                    text=text,
                    validation=report,
                    assessment=evidence_assessment,
                    freshness=freshness,
                )
            self._testability_assessment = evidence_assessment
            self._state = _State.EVIDENCE_REFINEMENT_REQUIRED
        elif report.decision == "hypothesis_changed":
            self._human_reviewed_risk = None
            self._testability_evidence_envelope = None
            self._testability_draft = None
            self._testability_response = None
            self._testability_assessment = None
            self._gherkin_evidence_envelope = None
            self._gherkin_candidate = None
            self._gherkin_response = None
            self._validated_gherkin_text = None
            self._state = _State.RISKS_READY

        return report

    @_with_workflow_lease
    def approve_gherkin(self, text: str) -> MilestoneTwoRunRecord:
        """Seal only the exact Gherkin text that passed local validation."""
        prepared = self._require_prepared("approve Gherkin")
        if (
            self._state not in {_State.GHERKIN_READY, _State.GHERKIN_VALIDATED}
            or self._human_reviewed_risk is None
            or self._testability_assessment is None
            or self._gherkin_evidence_envelope is None
            or self._gherkin_candidate is None
            or self._gherkin_validation_report is None
            or not self._gherkin_validation_report.approved
            or self._validated_gherkin_text != text
        ):
            raise MilestoneTwoTransitionError(
                "Cannot approve Gherkin: validate changed Gherkin before continuing."
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

        approval = approve_gherkin_candidate(
            candidate=self._gherkin_candidate,
            human_review=self._human_reviewed_risk,
            testability_assessment=self._testability_assessment,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=self._gherkin_evidence_envelope,
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
            testability_assessment=self._testability_assessment,
            context_refinements=tuple(self._context_refinements),
            gherkin_candidate=self._gherkin_candidate,
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
        self._write_final_measurements(record)
        self._recorder.finalize_run(self._run_handle, record)

        self._gherkin_approval = approval
        self._terminal_record = record
        self._state = _State.FINALIZED
        return record

    @_with_workflow_lease
    def finish_without_risk(self) -> MilestoneTwoRunRecord:
        """Acknowledge a supported non-risk outcome and seal its terminal record."""
        prepared = self._require_prepared("finish without risk")
        if self._state is not _State.RISKS_READY or self._risk_assessment is None:
            raise MilestoneTwoTransitionError(
                "Cannot finish without risk: propose risks before continuing."
            )

        assessment = self._risk_assessment
        if assessment.outcome != "no_meaningful_security_risk_found":
            raise MilestoneTwoTransitionError(
                "Cannot finish without risk: only the bounded non-risk assessment "
                "may use this exit. Insufficient context requires exhausted "
                "frozen-evidence refinement."
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

        status = MilestoneTwoStatus.NO_MEANINGFUL_SECURITY_RISK_FOUND
        reason_code = "no_meaningful_security_risk_found"
        explanation = assessment.rationale

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
            context_refinements=tuple(self._context_refinements),
            gherkin_candidate=None,
            gherkin_approval=None,
        )
        self._write_final_measurements(record)
        self._recorder.finalize_run(self._run_handle, record)

        self._terminal_record = record
        self._state = _State.FINALIZED
        return record

    @_with_workflow_lease
    def freshness(self) -> SnapshotFreshness:
        """Recheck and return the frozen PR's currentness without model activity."""
        if self._state is _State.FINALIZED:
            raise MilestoneTwoTransitionError(
                "Cannot check freshness: the workflow is already finalized."
            )
        if self._state is _State.STALE:
            raise MilestoneTwoTransitionError(
                "Cannot check freshness: the workflow is already stale."
            )
        prepared = self._require_prepared("check freshness")

        freshness = self._snapshot_acquirer.recheck(prepared.snapshot)
        self._freshness = freshness
        if freshness.status == "stale":
            self._state = _State.STALE
        return freshness

    def _write_final_measurements(self, record: object) -> None:
        """Persist only observed facts for the terminal research measurement file."""
        prepared = self._prepared
        assessment = self._risk_assessment
        response = self._risk_response
        freshness = self._freshness
        if (
            prepared is None
            or assessment is None
            or response is None
            or freshness is None
            or not self._is_typed_prepared(prepared)
            or not isinstance(assessment, RiskAssessment)
            or not isinstance(response, ModelResponse)
            or not isinstance(freshness, SnapshotFreshness)
            or not hasattr(record, "model_dump")
        ):
            return

        record_dump = record.model_dump(mode="json")
        record_sha256 = hashlib.sha256(
            canonical_json(record_dump).encode("utf-8")
        ).hexdigest()
        human_review = self._human_reviewed_risk
        candidate = self._gherkin_candidate
        measurements = {
            "acquisition": {
                "snapshot_key": prepared.snapshot.snapshot_key,
                "repository": prepared.snapshot.repository,
                "pull_number": prepared.snapshot.pull_number,
                "acquired_at": prepared.snapshot.acquired_at.isoformat(),
            },
            "diffs": {
                "comparison_count": len(prepared.diffs),
                "file_count": sum(len(diff.files) for diff in prepared.diffs),
                "artifact_sha256s": [diff.artifact_sha256 for diff in prepared.diffs],
            },
            "context": {
                "context_sha256": prepared.context.context_sha256,
                "selected_file_count": prepared.context.selected_file_count,
                "selected_anchor_count": prepared.context.selected_anchor_count,
                "selected_bytes": prepared.context.selected_bytes,
                "max_files": prepared.context.max_files,
                "max_anchors": prepared.context.max_anchors,
                "max_bytes": prepared.context.max_bytes,
            },
            "risk_generation": {
                "model_call_count": 1,
                "attempt_count": len(response.attempts),
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "response_sha256": response.response_sha256,
                "assessment_outcome": assessment.outcome,
            },
            "human_review": (
                {
                    "status": "completed",
                    "edited_field_count": len(human_review.field_changes),
                    "approved_at": human_review.approved_at.isoformat(),
                }
                if human_review is not None
                else {"status": "not_applicable"}
            ),
            "gherkin_generation": (
                {
                    "status": "completed",
                    "candidate_id": candidate.candidate_id,
                    "candidate_generated_at": candidate.generated_at.isoformat(),
                }
                if candidate is not None
                else {"status": "not_applicable"}
            ),
            "staleness": {
                "status": freshness.status,
                "reason_code": freshness.reason_code,
                "checked_at": freshness.checked_at.isoformat(),
            },
            "end_to_end": {
                "terminal_status": record.status.value,
                "reason_code": record.reason_code,
                "duration_ms": max(
                    0,
                    int((record.finished_at - self._started_at).total_seconds() * 1000),
                ),
            },
        }
        self._persist_transition(
            artifact_name="artifacts/measurements/final.json",
            event_type="workflow_final_measurements",
            payload=measurements,
            input_hashes={"terminal_record": record_sha256},
            reason_code=record.reason_code,
        )

    @staticmethod
    def _is_typed_prepared(prepared: PreparedPullRequest) -> bool:
        """Keep lightweight unit-test doubles outside the durable evidence path."""
        return (
            isinstance(prepared.snapshot, PullRequestSnapshot)
            and all(isinstance(diff, DiffArtifact) for diff in prepared.diffs)
            and isinstance(prepared.context, ContextBundle)
        )

    @staticmethod
    def _comparison_bindings(
        prepared: PreparedPullRequest,
    ) -> tuple[EvidenceArtifactBinding, ...]:
        """Bind later model stages to the same three frozen comparisons."""
        if not MilestoneTwoWorkflow._is_typed_prepared(prepared):
            return ()
        return tuple(
            EvidenceArtifactBinding(name=diff.kind, sha256=diff.artifact_sha256)
            for diff in prepared.diffs
        )

    def _persist_prepared(self, prepared: PreparedPullRequest) -> None:
        """Save the exact frozen evidence required to resume later model work."""
        payload = {
            "started_at": self._started_at.isoformat(),
            "snapshot": prepared.snapshot.model_dump(mode="json"),
            "diffs": [diff.model_dump(mode="json") for diff in prepared.diffs],
            "context": prepared.context.model_dump(mode="json"),
        }
        diffs_sha256 = hashlib.sha256(
            canonical_json(payload["diffs"]).encode("utf-8")
        ).hexdigest()
        self._persist_transition(
            artifact_name="artifacts/workflow/prepared.json",
            event_type="workflow_prepared",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "diffs": diffs_sha256,
                "context": prepared.context.context_sha256,
            },
            reason_code="prepared_evidence_recorded",
        )

    def _persist_risk_evidence_envelope(
        self,
        *,
        prepared: PreparedPullRequest,
        evidence_envelope: ModelEvidenceEnvelope,
    ) -> None:
        """Save the exact visibility boundary before invoking the risk model."""
        self._persist_transition(
            artifact_name="artifacts/model_evidence/risk_hypothesis.json",
            event_type="workflow_risk_evidence_selected",
            payload=evidence_envelope.model_dump(mode="json"),
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "evidence_envelope": evidence_envelope.envelope_sha256,
            },
            reason_code="risk_evidence_envelope_recorded",
        )

    def _persist_risk_generation(
        self,
        *,
        prepared: PreparedPullRequest,
        draft: RiskAssessmentDraft,
        response: ModelResponse,
    ) -> None:
        """Save the provider response before local grounding can advance state."""
        if self._risk_evidence_envelope is None:
            evidence_envelope = build_risk_evidence(
                snapshot=prepared.snapshot,
                diffs=prepared.diffs,
                context=prepared.context,
                budget=ProviderRequestBudget.from_settings(self._settings),
                priority_anchor_ids=self._refinement_priority_anchor_ids,
            ).envelope
            if draft.evidence_envelope_sha256 != evidence_envelope.envelope_sha256:
                raise MilestoneTwoTransitionError(
                    "Risk response does not bind the selected evidence envelope."
                )
            self._persist_risk_evidence_envelope(
                prepared=prepared,
                evidence_envelope=evidence_envelope,
            )
            self._risk_evidence_envelope = evidence_envelope
        elif (
            draft.evidence_envelope_sha256
            != self._risk_evidence_envelope.envelope_sha256
        ):
            raise MilestoneTwoTransitionError(
                "Risk response does not bind the selected evidence envelope."
            )
        payload = {
            "draft": draft.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/model_responses/risk_hypothesis.json",
            event_type="workflow_risk_generation",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "evidence_envelope": draft.evidence_envelope_sha256,
            },
            reason_code="risk_response_recorded",
        )

    def _record_risk_failure(
        self,
        *,
        prepared: PreparedPullRequest,
        failure: ModelFailureProvenance,
    ) -> None:
        """Persist the first safe failure for this retryable model stage."""
        if self._risk_failure is not None:
            return
        if self._is_typed_prepared(prepared):
            self._persist_risk_failure(prepared=prepared, failure=failure)
        self._risk_failure = failure

    def _persist_risk_failure(
        self,
        *,
        prepared: PreparedPullRequest,
        failure: ModelFailureProvenance,
    ) -> None:
        """Save safe provider metadata without retaining request or error content."""
        if failure.purpose != "risk_hypothesis":
            raise MilestoneTwoTransitionError(
                "Risk-failure provenance does not belong to the risk-proposal stage."
            )
        payload = {
            "snapshot_key": prepared.snapshot.snapshot_key,
            "context_sha256": prepared.context.context_sha256,
            "evidence_envelope_sha256": (
                self._risk_evidence_envelope.envelope_sha256
                if self._risk_evidence_envelope is not None
                else None
            ),
            "failure": failure.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/workflow/risk_generation_failure.json",
            event_type="workflow_risk_generation_failure",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "evidence_envelope": (
                    self._risk_evidence_envelope.envelope_sha256
                    if self._risk_evidence_envelope is not None
                    else "0" * 64
                ),
                "request": failure.request_sha256,
            },
            reason_code=failure.reason_code,
        )

    def _persist_human_review(
        self,
        *,
        prepared: PreparedPullRequest,
        assessment: RiskAssessment,
        review: HumanReviewedRisk,
        freshness: SnapshotFreshness,
    ) -> None:
        """Save one locally grounded reviewer selection before model generation."""
        selected = next(
            (
                hypothesis
                for hypothesis in assessment.hypotheses
                if hypothesis.hypothesis_id == review.selected_hypothesis_id
            ),
            None,
        )
        if (
            selected is None
            or review.snapshot_key != prepared.snapshot.snapshot_key
            or review.assessment_sha256 != assessment.assessment_sha256
            or review.selected_hypothesis_sha256
            != canonical_sha256(selected.model_dump(mode="json"))
            or freshness.snapshot_key != prepared.snapshot.snapshot_key
        ):
            raise MilestoneTwoTransitionError(
                "Human review is not bound to the current grounded evidence."
            )

        payload = {
            "review": review.model_dump(mode="json"),
            "freshness": freshness.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/workflow/human_review.json",
            event_type="workflow_human_review",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "assessment": assessment.assessment_sha256,
                "reviewed_risk": review.reviewed_content_sha256,
            },
            reason_code="human_review_recorded",
        )

    def _persist_testability_assessment(
        self,
        *,
        prepared: PreparedPullRequest,
        human_review: HumanReviewedRisk,
        draft: TestabilityAssessmentDraft,
        assessment: TestabilityAssessment,
        response: ModelResponse,
        freshness: SnapshotFreshness,
    ) -> None:
        """Save a locally validated testability decision before Gherkin work."""
        evidence_envelope = self._testability_evidence_envelope
        if evidence_envelope is None:
            evidence_envelope = build_testability_evidence(
                human_review=human_review,
                context=prepared.context,
                comparison_bindings=self._comparison_bindings(prepared),
                budget=ProviderRequestBudget.from_settings(self._settings),
            ).envelope
            if draft.evidence_envelope_sha256 != evidence_envelope.envelope_sha256:
                raise MilestoneTwoTransitionError(
                    "Testability response does not bind the selected evidence envelope."
                )
            self._persist_testability_evidence_envelope(
                prepared=prepared,
                human_review=human_review,
                evidence_envelope=evidence_envelope,
            )
            self._testability_evidence_envelope = evidence_envelope
        if (
            not isinstance(evidence_envelope, ModelEvidenceEnvelope)
            or draft.snapshot_key != prepared.snapshot.snapshot_key
            or draft.context_sha256 != prepared.context.context_sha256
            or draft.reviewed_risk_sha256 != human_review.reviewed_content_sha256
            or assessment.snapshot_key != prepared.snapshot.snapshot_key
            or assessment.context_sha256 != prepared.context.context_sha256
            or assessment.reviewed_risk_sha256 != human_review.reviewed_content_sha256
            or draft.evidence_envelope_sha256 != evidence_envelope.envelope_sha256
            or assessment.evidence_envelope_sha256 != evidence_envelope.envelope_sha256
            or freshness.snapshot_key != prepared.snapshot.snapshot_key
        ):
            raise MilestoneTwoTransitionError(
                "Testability assessment is not bound to the current reviewed "
                "frozen evidence."
            )

        payload = {
            "draft": draft.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "freshness": freshness.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/workflow/testability_assessment.json",
            event_type="workflow_testability_assessment",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "reviewed_risk": human_review.reviewed_content_sha256,
                "testability": assessment.assessment_sha256,
                "evidence_envelope": evidence_envelope.envelope_sha256,
            },
            reason_code="testability_assessment_recorded",
        )

    def _persist_testability_evidence_envelope(
        self,
        *,
        prepared: PreparedPullRequest,
        human_review: HumanReviewedRisk,
        evidence_envelope: ModelEvidenceEnvelope,
    ) -> None:
        """Save the exact visibility boundary before the testability model call."""
        build_testability_request(
            human_review=human_review,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=evidence_envelope,
        )
        self._persist_transition(
            artifact_name="artifacts/model_evidence/testability_assessment.json",
            event_type="workflow_testability_evidence_selected",
            payload=evidence_envelope.model_dump(mode="json"),
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "reviewed_risk": human_review.reviewed_content_sha256,
                "evidence_envelope": evidence_envelope.envelope_sha256,
            },
            reason_code="testability_evidence_envelope_recorded",
        )

    def _persist_evidence_refinement(
        self,
        *,
        prepared: PreparedPullRequest,
        needs: tuple[FrozenEvidenceNeed, ...],
        successor_context: ContextBundle,
        refinement: EvidenceRefinementResult,
        freshness: SnapshotFreshness,
    ) -> None:
        """Save one refinement link before mutating any in-memory workflow state."""
        requested_need_sha256 = canonical_sha256(
            [need.model_dump(mode="json") for need in needs]
        )
        parent_anchor_ids = {anchor.anchor_id for anchor in prepared.context.anchors}
        successor_anchor_ids = {
            anchor.anchor_id for anchor in successor_context.anchors
        }
        if (
            refinement.round_number != len(self._context_refinements) + 1
            or refinement.parent_context_sha256 != prepared.context.context_sha256
            or refinement.successor_context_sha256 != successor_context.context_sha256
            or refinement.requested_need_sha256 != requested_need_sha256
            or successor_context.snapshot_key != prepared.snapshot.snapshot_key
            or not parent_anchor_ids.issubset(successor_anchor_ids)
            or set(refinement.added_anchor_ids)
            != successor_anchor_ids - parent_anchor_ids
            or any(
                anchor_id not in parent_anchor_ids
                for anchor_id in refinement.priority_anchor_ids
            )
            or any(
                anchor_id not in parent_anchor_ids
                for need in needs
                for anchor_id in need.supporting_anchor_ids
            )
            or freshness.snapshot_key != prepared.snapshot.snapshot_key
            or freshness.status != "current"
        ):
            raise MilestoneTwoTransitionError(
                "Frozen-evidence refinement is not bound to the current snapshot."
            )

        payload = {
            "refinement": refinement.model_dump(mode="json"),
            "needs": [need.model_dump(mode="json") for need in needs],
            "successor_context": successor_context.model_dump(mode="json"),
            "freshness": freshness.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name=(
                "artifacts/workflow/evidence_refinements/"
                f"{refinement.round_number}.json"
            ),
            event_type=f"workflow_frozen_evidence_refinement_{refinement.round_number}",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "parent_context": prepared.context.context_sha256,
                "successor_context": successor_context.context_sha256,
                "requested_need": requested_need_sha256,
                "refinement": refinement.refinement_sha256,
            },
            reason_code=refinement.reason_code,
        )

    def _persist_gherkin_generation(
        self,
        *,
        prepared: PreparedPullRequest,
        human_review: HumanReviewedRisk,
        testability_assessment: TestabilityAssessment,
        candidate: GherkinCandidate,
        response: ModelResponse,
        freshness: SnapshotFreshness,
    ) -> None:
        """Save one validated Gherkin response before a reviewer can edit it."""
        evidence_envelope = self._gherkin_evidence_envelope
        if evidence_envelope is None:
            evidence_envelope = build_gherkin_evidence(
                human_review=human_review,
                testability_assessment=testability_assessment,
                context=prepared.context,
                comparison_bindings=self._comparison_bindings(prepared),
                budget=ProviderRequestBudget.from_settings(self._settings),
            ).envelope
            if candidate.evidence_envelope_sha256 != evidence_envelope.envelope_sha256:
                raise MilestoneTwoTransitionError(
                    "Gherkin response does not bind the selected evidence envelope."
                )
            self._persist_gherkin_evidence_envelope(
                prepared=prepared,
                human_review=human_review,
                testability_assessment=testability_assessment,
                evidence_envelope=evidence_envelope,
            )
            self._gherkin_evidence_envelope = evidence_envelope
        if (
            not isinstance(evidence_envelope, ModelEvidenceEnvelope)
            or candidate.snapshot_key != prepared.snapshot.snapshot_key
            or candidate.reviewed_risk_sha256 != human_review.reviewed_content_sha256
            or candidate.approved_risk != human_review.reviewed_risk
            or candidate.evidence_envelope_sha256 != evidence_envelope.envelope_sha256
            or freshness.snapshot_key != prepared.snapshot.snapshot_key
        ):
            raise MilestoneTwoTransitionError(
                "Generated Gherkin is not bound to the current reviewed evidence."
            )

        payload = {
            "candidate": candidate.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "freshness": freshness.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/workflow/gherkin_generation.json",
            event_type="workflow_gherkin_generation",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "reviewed_risk": human_review.reviewed_content_sha256,
                "evidence_envelope": evidence_envelope.envelope_sha256,
            },
            reason_code="gherkin_response_recorded",
        )

    def _persist_gherkin_evidence_envelope(
        self,
        *,
        prepared: PreparedPullRequest,
        human_review: HumanReviewedRisk,
        testability_assessment: TestabilityAssessment,
        evidence_envelope: ModelEvidenceEnvelope,
    ) -> None:
        """Save the exact visibility boundary before the Gherkin model call."""
        build_gherkin_request(
            human_review=human_review,
            testability_assessment=testability_assessment,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=evidence_envelope,
        )
        self._persist_transition(
            artifact_name="artifacts/model_evidence/gherkin_generation.json",
            event_type="workflow_gherkin_evidence_selected",
            payload=evidence_envelope.model_dump(mode="json"),
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "reviewed_risk": human_review.reviewed_content_sha256,
                "testability": testability_assessment.assessment_sha256,
                "evidence_envelope": evidence_envelope.envelope_sha256,
            },
            reason_code="gherkin_evidence_envelope_recorded",
        )

    def _persist_gherkin_validation(
        self,
        *,
        prepared: PreparedPullRequest,
        human_review: HumanReviewedRisk,
        source_candidate: GherkinCandidate,
        candidate: GherkinCandidate,
        validation: GherkinValidationReport,
        freshness: SnapshotFreshness,
    ) -> None:
        """Save one accepted edited successor before it may be approved."""
        source_candidate_sha256 = canonical_sha256(
            source_candidate.model_dump(mode="json")
        )
        if (
            not validation.approved
            or source_candidate.snapshot_key != prepared.snapshot.snapshot_key
            or source_candidate.context_sha256 != prepared.context.context_sha256
            or source_candidate.reviewed_risk_sha256
            != human_review.reviewed_content_sha256
            or candidate.snapshot_key != prepared.snapshot.snapshot_key
            or candidate.context_sha256 != prepared.context.context_sha256
            or candidate.reviewed_risk_sha256 != human_review.reviewed_content_sha256
            or candidate.approved_risk != human_review.reviewed_risk
            or freshness.snapshot_key != prepared.snapshot.snapshot_key
        ):
            raise MilestoneTwoTransitionError(
                "Validated Gherkin is not bound to the current reviewed "
                "frozen evidence."
            )

        payload = {
            "source_candidate_sha256": source_candidate_sha256,
            "candidate": candidate.model_dump(mode="json"),
            "validation": {
                "decision": validation.decision,
                "approved": validation.approved,
                "reason_codes": list(validation.reason_codes),
            },
            "freshness": freshness.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/workflow/gherkin_validation.json",
            event_type="workflow_gherkin_validation",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "reviewed_risk": human_review.reviewed_content_sha256,
                "source_candidate": source_candidate_sha256,
            },
            reason_code="gherkin_edit_validated",
        )

    def _persist_gherkin_evidence_gap(
        self,
        *,
        prepared: PreparedPullRequest,
        human_review: HumanReviewedRisk,
        candidate: GherkinCandidate,
        text: str,
        validation: GherkinValidationReport,
        assessment: TestabilityAssessment,
        freshness: SnapshotFreshness,
    ) -> None:
        """Save an edited-scenario evidence gap before frozen-code refinement."""
        source_candidate_sha256 = canonical_sha256(candidate.model_dump(mode="json"))
        expected_assessment = _gherkin_evidence_assessment(
            candidate=candidate,
            human_review=human_review,
            context=prepared.context,
            generated_at=assessment.generated_at,
        )
        if (
            text == candidate.gherkin_text
            or validation.decision != "needs_more_frozen_evidence"
            or validation.approved
            or assessment != expected_assessment
            or candidate.snapshot_key != prepared.snapshot.snapshot_key
            or candidate.context_sha256 != prepared.context.context_sha256
            or candidate.reviewed_risk_sha256 != human_review.reviewed_content_sha256
            or candidate.approved_risk != human_review.reviewed_risk
            or freshness.snapshot_key != prepared.snapshot.snapshot_key
            or freshness.status != "current"
        ):
            raise MilestoneTwoTransitionError(
                "The edited Gherkin evidence gap is not bound to the current "
                "reviewed frozen evidence."
            )

        payload = {
            "source_candidate_sha256": source_candidate_sha256,
            "text": text,
            "validation": {
                "decision": validation.decision,
                "approved": validation.approved,
                "reason_codes": list(validation.reason_codes),
            },
            "assessment": assessment.model_dump(mode="json"),
            "freshness": freshness.model_dump(mode="json"),
        }
        self._persist_transition(
            artifact_name="artifacts/workflow/gherkin_evidence_gap.json",
            event_type="workflow_gherkin_evidence_gap",
            payload=payload,
            input_hashes={
                "snapshot": prepared.snapshot.snapshot_key,
                "context": prepared.context.context_sha256,
                "reviewed_risk": human_review.reviewed_content_sha256,
                "source_candidate": source_candidate_sha256,
                "testability": assessment.assessment_sha256,
            },
            reason_code="gherkin_edit_needs_frozen_evidence",
        )

    def _persist_transition(
        self,
        *,
        artifact_name: str,
        event_type: str,
        payload: Mapping[str, object],
        input_hashes: Mapping[str, str],
        reason_code: str,
    ) -> None:
        """Write one hash-bound artifact, then append its exact transformation."""
        artifact_name = self._round_scoped_artifact_name(artifact_name)
        if self._analysis_round and not artifact_name.startswith(
            "artifacts/workflow/evidence_refinements/"
        ):
            event_type = f"{event_type}_round_{self._analysis_round}"
        content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        started_at = datetime.now(UTC)
        event = TransformationEvent(
            event_type=event_type,
            inputs={name: name for name in input_hashes},
            outputs={artifact_name: artifact_name},
            input_hashes=dict(input_hashes),
            output_hashes={artifact_name: digest},
            versions={
                "triageguard": "2.0.0",
                "workflow": "milestone_two",
            },
            started_at=started_at,
            finished_at=started_at + timedelta(microseconds=1),
            reason_code=reason_code,
        )
        self._recorder.write_artifact(
            self._run_handle,
            artifact_name,
            content,
            event,
        )
        self._recorder.record_transformation(self._run_handle, event)

    def _load_durable_artifact(
        self,
        artifact_name: str,
    ) -> tuple[dict[str, object], TransformationEvent] | None:
        """Read an artifact only when its journal and hash agree exactly."""
        artifact_name = self._round_scoped_artifact_name(artifact_name)
        try:
            content = self._recorder.read_artifact(self._run_handle, artifact_name)
        except FileNotFoundError:
            return None

        digest = hashlib.sha256(content).hexdigest()
        events = self._recorder.read_events(self._run_handle)
        started_payloads = [
            event.payload
            for event in events
            if event.event_type == "artifact_write_started"
            and event.payload.get("artifact_name") == artifact_name
        ]
        completed_payloads = [
            event.payload
            for event in events
            if event.event_type == "artifact_write_completed"
            and event.payload.get("artifact_name") == artifact_name
        ]
        if len(started_payloads) != 1 or len(completed_payloads) != 1:
            raise MilestoneTwoTransitionError(
                "A durable workflow artifact lacks one exact journal pair."
            )

        try:
            started = ArtifactWriteJournal.model_validate(started_payloads[0])
            completed = ArtifactWriteJournal.model_validate(completed_payloads[0])
        except (TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "A durable workflow artifact journal is invalid."
            ) from error

        if (
            started != completed
            or started.artifact_sha256 != digest
            or started.artifact_byte_count != len(content)
            or started.provenance.outputs.get(artifact_name) != artifact_name
            or started.provenance.output_hashes.get(artifact_name) != digest
        ):
            raise MilestoneTwoTransitionError(
                "A durable workflow artifact does not match its journal."
            )

        matching_transformations = [
            event
            for event in events
            if event.event_type == started.provenance.event_type
        ]
        if len(matching_transformations) > 1:
            raise MilestoneTwoTransitionError(
                "A durable workflow transformation is duplicated."
            )
        if matching_transformations:
            try:
                transformation = TransformationEvent.model_validate(
                    matching_transformations[0].payload
                )
            except (TypeError, ValueError) as error:
                raise MilestoneTwoTransitionError(
                    "A durable workflow transformation is invalid."
                ) from error
            if transformation != started.provenance:
                raise MilestoneTwoTransitionError(
                    "A durable workflow transformation contradicts its artifact."
                )
        else:
            transformation = started.provenance
            self._recorder.record_transformation(self._run_handle, transformation)

        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise MilestoneTwoTransitionError(
                "A durable workflow artifact is not valid JSON."
            ) from error
        if not isinstance(payload, dict):
            raise MilestoneTwoTransitionError(
                "A durable workflow artifact must contain a JSON object."
            )
        return payload, transformation

    def _round_scoped_artifact_name(self, artifact_name: str) -> str:
        """Keep each post-refinement analysis round in immutable distinct paths."""
        if (
            self._analysis_round == 0
            or artifact_name == "artifacts/workflow/prepared.json"
            or artifact_name.startswith("artifacts/workflow/evidence_refinements/")
        ):
            return artifact_name
        if not artifact_name.startswith("artifacts/"):
            raise ValueError("workflow artifact names must live under artifacts/")
        suffix = artifact_name.removeprefix("artifacts/")
        return f"artifacts/refinement_rounds/{self._analysis_round}/{suffix}"

    def _hydrate_refinement_chain(
        self,
        prepared: PreparedPullRequest,
    ) -> PreparedPullRequest:
        """Replay every contiguous hash-bound refinement before model artifacts."""
        current = prepared
        prefix = "artifacts/workflow/evidence_refinements/"
        durable_rounds: set[int] = set()
        for event in self._recorder.read_events(self._run_handle):
            if event.event_type != "artifact_write_completed":
                continue
            artifact_name = event.payload.get("artifact_name")
            if not isinstance(artifact_name, str) or not artifact_name.startswith(
                prefix
            ):
                continue
            suffix = artifact_name.removeprefix(prefix)
            if not suffix.endswith(".json") or not suffix[:-5].isdigit():
                raise MilestoneTwoTransitionError(
                    "The saved refinement chain contains an invalid round path."
                )
            parsed_round = int(suffix[:-5])
            if parsed_round <= 0:
                raise MilestoneTwoTransitionError(
                    "The saved refinement chain contains an invalid round path."
                )
            durable_rounds.add(parsed_round)
        if durable_rounds and max(durable_rounds) > (
            self._settings.max_model_evidence_rounds + 1
        ):
            raise MilestoneTwoTransitionError(
                "The saved refinement chain exceeds the configured round limit."
            )
        saw_gap = False
        saw_exhaustion = False
        for round_number in range(1, self._settings.max_model_evidence_rounds + 2):
            item = self._load_durable_artifact(
                f"artifacts/workflow/evidence_refinements/{round_number}.json"
            )
            if item is None:
                saw_gap = True
                continue
            if saw_gap:
                raise MilestoneTwoTransitionError(
                    "The saved refinement chain has a missing parent round."
                )
            if saw_exhaustion:
                raise MilestoneTwoTransitionError(
                    "The saved refinement chain continues after exhaustion."
                )

            payload, _event = item
            try:
                raw_refinement = payload["refinement"]
                raw_needs = payload["needs"]
                raw_context = payload["successor_context"]
                raw_freshness = payload["freshness"]
                if (
                    not isinstance(raw_refinement, dict)
                    or not isinstance(raw_needs, list)
                    or not isinstance(raw_context, dict)
                    or not isinstance(raw_freshness, dict)
                ):
                    raise TypeError("refinement payload structure is invalid")
                refinement = EvidenceRefinementResult.model_validate(raw_refinement)
                needs = tuple(
                    FrozenEvidenceNeed.model_validate(need) for need in raw_needs
                )
                successor = ContextBundle.model_validate(raw_context)
                freshness = SnapshotFreshness.model_validate(raw_freshness)
            except (KeyError, TypeError, ValueError) as error:
                raise MilestoneTwoTransitionError(
                    "The saved frozen-evidence refinement is invalid."
                ) from error

            current_anchor_ids = {
                anchor.anchor_id for anchor in current.context.anchors
            }
            successor_anchor_ids = {anchor.anchor_id for anchor in successor.anchors}
            requested_need_sha256 = canonical_sha256(
                [need.model_dump(mode="json") for need in needs]
            )
            if (
                refinement.round_number != round_number
                or refinement.parent_context_sha256 != current.context.context_sha256
                or refinement.successor_context_sha256 != successor.context_sha256
                or refinement.requested_need_sha256 != requested_need_sha256
                or successor.snapshot_key != current.snapshot.snapshot_key
                or not current_anchor_ids.issubset(successor_anchor_ids)
                or set(refinement.added_anchor_ids)
                != successor_anchor_ids - current_anchor_ids
                or any(
                    anchor_id not in current_anchor_ids
                    for anchor_id in refinement.priority_anchor_ids
                )
                or freshness.snapshot_key != current.snapshot.snapshot_key
                or freshness.status != "current"
            ):
                raise MilestoneTwoTransitionError(
                    "The saved refinement chain is not bound to its parent context."
                )

            self._context_refinements.append(refinement)
            self._freshness = freshness
            if refinement.exhausted:
                saw_exhaustion = True
                continue
            current = PreparedPullRequest(
                snapshot=current.snapshot,
                diffs=current.diffs,
                context=successor,
            )
            self._refinement_priority_anchor_ids = tuple(
                dict.fromkeys(
                    (*refinement.priority_anchor_ids, *refinement.added_anchor_ids)
                )
            )
            self._analysis_round += 1
        return current

    def _load_terminal_record(self) -> MilestoneTwoRunRecord | None:
        """Reload only a sealed terminal record with a matching finalization journal."""
        try:
            content = self._recorder.read_artifact(
                self._run_handle,
                "run_record.json",
            )
        except FileNotFoundError:
            return None

        digest = hashlib.sha256(content).hexdigest()
        events = self._recorder.read_events(self._run_handle)
        started = [
            event.payload
            for event in events
            if event.event_type == LifecycleEventType.FINALIZATION_STARTED.value
        ]
        completed = [
            event.payload
            for event in events
            if event.event_type == LifecycleEventType.FINALIZATION_COMPLETED.value
        ]
        expected_journal = {"record_sha256": digest}
        if (
            len(started) != 1
            or len(completed) != 1
            or started[0] != expected_journal
            or completed[0] != expected_journal
        ):
            raise MilestoneTwoTransitionError(
                "The sealed terminal record has an invalid finalization journal."
            )

        try:
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise TypeError("terminal record must be a JSON object")
            record = MilestoneTwoRunRecord.from_persisted(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise MilestoneTwoTransitionError(
                "The sealed terminal record is invalid."
            ) from error

        expected_content = (
            canonical_json(record.model_dump(mode="json")) + "\n"
        ).encode("utf-8")
        if expected_content != content or record.run_id != self._run_handle.run_id:
            raise MilestoneTwoTransitionError(
                "The sealed terminal record does not match this workflow run."
            )
        return record

    def _hydrate_durable_state(self) -> None:
        """Rebuild only verified completed or recoverable durable workflow stages."""
        terminal_record = self._load_terminal_record()
        if terminal_record is not None:
            self._context_refinements = list(terminal_record.context_refinements)
            successful_refinements = tuple(
                refinement
                for refinement in terminal_record.context_refinements
                if not refinement.exhausted
            )
            self._analysis_round = len(successful_refinements)
            if successful_refinements:
                last_refinement = successful_refinements[-1]
                self._refinement_priority_anchor_ids = tuple(
                    dict.fromkeys(
                        (
                            *last_refinement.priority_anchor_ids,
                            *last_refinement.added_anchor_ids,
                        )
                    )
                )
            terminal_assessment = terminal_record.risk_assessment
            if terminal_assessment is not None:
                envelope_item = self._load_durable_artifact(
                    "artifacts/model_evidence/risk_hypothesis.json"
                )
                if envelope_item is None:
                    raise MilestoneTwoTransitionError(
                        "The terminal risk assessment is missing its evidence envelope."
                    )
                envelope_payload, _envelope_event = envelope_item
                try:
                    evidence_envelope = ModelEvidenceEnvelope.model_validate(
                        envelope_payload
                    )
                    evidence_envelope = validate_envelope_binding(
                        envelope=evidence_envelope,
                        stage="risk_hypothesis",
                        context=terminal_assessment.context_bundle,
                        comparison_bindings=evidence_envelope.comparison_bindings,
                        input_bindings=(),
                        output_schema_sha256=canonical_sha256(RISK_OUTPUT_SCHEMA),
                        max_request_body_bytes=self._settings.max_model_request_bytes,
                    )
                    if (
                        evidence_envelope.selection_policy_version != "risk-evidence-v1"
                        or evidence_envelope.envelope_sha256
                        != terminal_assessment.evidence_envelope_sha256
                        or not set(self._refinement_priority_anchor_ids).issubset(
                            {
                                anchor.anchor_id
                                for anchor in evidence_envelope.visible_anchors
                            }
                        )
                    ):
                        raise ValueError(
                            "terminal assessment does not bind the risk envelope"
                        )
                except (TypeError, ValueError) as error:
                    raise MilestoneTwoTransitionError(
                        "The terminal risk evidence envelope is invalid."
                    ) from error
                self._risk_evidence_envelope = evidence_envelope
            terminal_review = terminal_record.human_reviewed_risk
            terminal_testability = terminal_record.testability_assessment
            if terminal_testability is not None:
                if terminal_assessment is None or terminal_review is None:
                    raise MilestoneTwoTransitionError(
                        "The terminal testability assessment lacks its dependencies."
                    )
                item = self._load_durable_artifact(
                    "artifacts/model_evidence/testability_assessment.json"
                )
                if item is None:
                    raise MilestoneTwoTransitionError(
                        "The terminal testability assessment is missing its evidence envelope."
                    )
                payload, _event = item
                try:
                    envelope = ModelEvidenceEnvelope.model_validate(payload)
                    envelope = validate_envelope_binding(
                        envelope=envelope,
                        stage="testability_assessment",
                        context=terminal_assessment.context_bundle,
                        comparison_bindings=envelope.comparison_bindings,
                        input_bindings=(
                            EvidenceArtifactBinding(
                                name="human_reviewed_risk",
                                sha256=terminal_review.reviewed_content_sha256,
                            ),
                        ),
                        output_schema_sha256=canonical_sha256(
                            TESTABILITY_OUTPUT_SCHEMA
                        ),
                        max_request_body_bytes=self._settings.max_model_request_bytes,
                    )
                    if (
                        terminal_testability.evidence_envelope_sha256
                        != envelope.envelope_sha256
                    ):
                        raise ValueError(
                            "terminal testability does not bind its envelope"
                        )
                except (TypeError, ValueError) as error:
                    raise MilestoneTwoTransitionError(
                        "The terminal testability evidence envelope is invalid."
                    ) from error
                self._testability_evidence_envelope = envelope

            terminal_candidate = terminal_record.gherkin_candidate
            if terminal_candidate is not None:
                if (
                    terminal_assessment is None
                    or terminal_review is None
                    or terminal_testability is None
                ):
                    raise MilestoneTwoTransitionError(
                        "The terminal Gherkin candidate lacks its dependencies."
                    )
                item = self._load_durable_artifact(
                    "artifacts/model_evidence/gherkin_generation.json"
                )
                if item is None:
                    raise MilestoneTwoTransitionError(
                        "The terminal Gherkin candidate is missing its evidence envelope."
                    )
                payload, _event = item
                try:
                    envelope = ModelEvidenceEnvelope.model_validate(payload)
                    envelope = validate_envelope_binding(
                        envelope=envelope,
                        stage="gherkin_generation",
                        context=terminal_assessment.context_bundle,
                        comparison_bindings=envelope.comparison_bindings,
                        input_bindings=(
                            EvidenceArtifactBinding(
                                name="human_reviewed_risk",
                                sha256=terminal_review.reviewed_content_sha256,
                            ),
                            EvidenceArtifactBinding(
                                name="testability_assessment",
                                sha256=terminal_testability.assessment_sha256,
                            ),
                        ),
                        output_schema_sha256=canonical_sha256(GHERKIN_OUTPUT_SCHEMA),
                        max_request_body_bytes=self._settings.max_model_request_bytes,
                    )
                    if (
                        terminal_candidate.evidence_envelope_sha256
                        != envelope.envelope_sha256
                    ):
                        raise ValueError("terminal Gherkin does not bind its envelope")
                except (TypeError, ValueError) as error:
                    raise MilestoneTwoTransitionError(
                        "The terminal Gherkin evidence envelope is invalid."
                    ) from error
                self._gherkin_evidence_envelope = envelope
            self._started_at = terminal_record.started_at
            self._terminal_record = terminal_record
            self._freshness = terminal_record.freshness
            self._risk_assessment = terminal_record.risk_assessment
            self._human_reviewed_risk = terminal_record.human_reviewed_risk
            self._testability_assessment = terminal_record.testability_assessment
            self._gherkin_candidate = terminal_record.gherkin_candidate
            self._gherkin_approval = terminal_record.gherkin_approval
            self._state = _State.FINALIZED
            return

        prepared_item = self._load_durable_artifact("artifacts/workflow/prepared.json")
        if prepared_item is None:
            return

        prepared_payload, _prepared_event = prepared_item
        try:
            snapshot_payload = prepared_payload["snapshot"]
            diffs_payload = prepared_payload["diffs"]
            context_payload = prepared_payload["context"]
            started_at_text = prepared_payload["started_at"]
            if (
                not isinstance(snapshot_payload, dict)
                or not isinstance(diffs_payload, list)
                or not isinstance(context_payload, dict)
                or not isinstance(started_at_text, str)
                or len(diffs_payload) != 3
            ):
                raise ValueError("prepared payload structure is invalid")
            snapshot = PullRequestSnapshot.model_validate(snapshot_payload)
            loaded_diffs = tuple(
                DiffArtifact.model_validate(item) for item in diffs_payload
            )
            context = ContextBundle.model_validate(context_payload)
            started_at = datetime.fromisoformat(started_at_text)
            if started_at.tzinfo is None:
                raise ValueError("prepared timestamp has no timezone")
        except (KeyError, TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved prepared evidence is invalid."
            ) from error

        prepared = PreparedPullRequest(
            snapshot=snapshot,
            diffs=(loaded_diffs[0], loaded_diffs[1], loaded_diffs[2]),
            context=context,
        )
        if not self._is_typed_prepared(prepared):
            raise MilestoneTwoTransitionError(
                "The saved prepared evidence has invalid runtime types."
            )
        prepared = self._hydrate_refinement_chain(prepared)
        self._started_at = started_at
        self._prepared = prepared
        self._state = _State.PREPARED

        envelope_item = self._load_durable_artifact(
            "artifacts/model_evidence/risk_hypothesis.json"
        )
        risk_item = self._load_durable_artifact(
            "artifacts/model_responses/risk_hypothesis.json"
        )
        failure_item = self._load_durable_artifact(
            "artifacts/workflow/risk_generation_failure.json"
        )
        if envelope_item is None:
            if risk_item is not None or failure_item is not None:
                raise MilestoneTwoTransitionError(
                    "The saved risk stage is missing its evidence envelope."
                )
            return

        envelope_payload, _envelope_event = envelope_item
        try:
            evidence_envelope = ModelEvidenceEnvelope.model_validate(envelope_payload)
            build_risk_request(
                snapshot=prepared.snapshot,
                diffs=prepared.diffs,
                context=prepared.context,
                evidence_envelope=evidence_envelope,
            )
            if (
                evidence_envelope.max_request_body_bytes
                != self._settings.max_model_request_bytes
                or not set(self._refinement_priority_anchor_ids).issubset(
                    {anchor.anchor_id for anchor in evidence_envelope.visible_anchors}
                )
            ):
                raise ValueError("saved envelope uses a different request budget")
        except (TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved risk evidence envelope is invalid."
            ) from error
        self._risk_evidence_envelope = evidence_envelope

        if risk_item is None:
            if failure_item is None:
                return
            failure_payload, _failure_event = failure_item
            try:
                saved_snapshot_key = failure_payload["snapshot_key"]
                saved_context_sha256 = failure_payload["context_sha256"]
                saved_envelope_sha256 = failure_payload["evidence_envelope_sha256"]
                saved_failure = failure_payload["failure"]
                if (
                    not isinstance(saved_snapshot_key, str)
                    or not isinstance(saved_context_sha256, str)
                    or not isinstance(saved_envelope_sha256, str)
                    or not isinstance(saved_failure, dict)
                ):
                    raise TypeError("risk-failure payload structure is invalid")
                failure = ModelFailureProvenance.model_validate(saved_failure)
            except (KeyError, TypeError, ValueError) as error:
                raise MilestoneTwoTransitionError(
                    "The saved risk-failure provenance is invalid."
                ) from error

            if (
                saved_snapshot_key != prepared.snapshot.snapshot_key
                or saved_context_sha256 != prepared.context.context_sha256
                or saved_envelope_sha256 != evidence_envelope.envelope_sha256
                or failure.purpose != "risk_hypothesis"
            ):
                raise MilestoneTwoTransitionError(
                    "The saved risk-failure provenance is not bound to prepared evidence."
                )
            self._risk_failure = failure
            return

        risk_payload, _risk_event = risk_item
        try:
            draft_payload = risk_payload["draft"]
            response_payload = risk_payload["response"]
            if not isinstance(draft_payload, dict) or not isinstance(
                response_payload, dict
            ):
                raise TypeError("risk payload structure is invalid")
            draft = RiskAssessmentDraft.model_validate(draft_payload)
            response = ModelResponse.model_validate(response_payload)
        except (KeyError, TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved risk response is invalid."
            ) from error

        if (
            draft.snapshot_key != prepared.snapshot.snapshot_key
            or draft.context_sha256 != prepared.context.context_sha256
            or draft.evidence_envelope_sha256 != evidence_envelope.envelope_sha256
            or response.data != draft.model_dump(mode="json")
        ):
            raise MilestoneTwoTransitionError(
                "The saved risk response is not bound to prepared evidence."
            )
        assessment, grounding_report = validate_risk_assessment(
            draft=draft,
            snapshot=prepared.snapshot,
            context=prepared.context,
            evidence_envelope=evidence_envelope,
        )
        if assessment is None:
            raise MilestoneTwoTransitionError(
                "The saved risk response no longer passes local grounding."
            )

        self._risk_draft = draft
        self._risk_response = response
        self._risk_assessment = assessment
        self._risk_grounding_report = grounding_report
        self._state = _State.RISKS_READY

        review_item = self._load_durable_artifact(
            "artifacts/workflow/human_review.json"
        )
        if review_item is None:
            return

        review_payload, _review_event = review_item
        try:
            saved_review = review_payload["review"]
            saved_freshness = review_payload["freshness"]
            if not isinstance(saved_review, dict) or not isinstance(
                saved_freshness,
                dict,
            ):
                raise TypeError("human-review payload structure is invalid")
            review = HumanReviewedRisk.model_validate(saved_review)
            review_freshness = SnapshotFreshness.model_validate(saved_freshness)
            selected = next(
                (
                    hypothesis
                    for hypothesis in assessment.hypotheses
                    if hypothesis.hypothesis_id == review.selected_hypothesis_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("review selects an unknown risk hypothesis")

            edits: dict[str, object] = {}
            for change in review.field_changes:
                original_value = getattr(selected, change.field_name)
                edits[change.field_name] = (
                    change.after
                    if isinstance(original_value, str)
                    else json.loads(change.after)
                )

            expected_review = create_human_review(
                assessment=assessment,
                hypothesis_id=selected.hypothesis_id,
                edits=edits,
                selected_anchor_ids=review.reviewed_risk.citation_anchor_ids,
                reviewed_at=review.approved_at,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MilestoneTwoTransitionError(
                "The saved human review is invalid."
            ) from error

        if (
            review != expected_review
            or review.snapshot_key != prepared.snapshot.snapshot_key
            or review_freshness.snapshot_key != prepared.snapshot.snapshot_key
            or review_freshness.status != "current"
        ):
            raise MilestoneTwoTransitionError(
                "The saved human review is not bound to current grounded evidence."
            )

        self._human_reviewed_risk = review
        self._freshness = review_freshness
        self._state = _State.RISK_APPROVED

        testability_envelope_item = self._load_durable_artifact(
            "artifacts/model_evidence/testability_assessment.json"
        )
        testability_item = self._load_durable_artifact(
            "artifacts/workflow/testability_assessment.json"
        )
        if testability_envelope_item is None:
            if testability_item is not None:
                raise MilestoneTwoTransitionError(
                    "The saved testability stage is missing its evidence envelope."
                )
            return
        if testability_item is None:
            return

        testability_envelope_payload, _testability_envelope_event = (
            testability_envelope_item
        )
        try:
            testability_envelope = ModelEvidenceEnvelope.model_validate(
                testability_envelope_payload
            )
            build_testability_request(
                human_review=review,
                context=prepared.context,
                comparison_bindings=self._comparison_bindings(prepared),
                evidence_envelope=testability_envelope,
            )
            if (
                testability_envelope.max_request_body_bytes
                != self._settings.max_model_request_bytes
            ):
                raise ValueError("saved envelope uses a different request budget")
        except (TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved testability evidence envelope is invalid."
            ) from error
        self._testability_evidence_envelope = testability_envelope

        testability_payload, _testability_event = testability_item
        try:
            saved_draft = testability_payload["draft"]
            saved_assessment = testability_payload["assessment"]
            saved_response = testability_payload["response"]
            saved_freshness = testability_payload["freshness"]
            if (
                not isinstance(saved_draft, dict)
                or not isinstance(saved_assessment, dict)
                or not isinstance(saved_response, dict)
                or not isinstance(saved_freshness, dict)
            ):
                raise TypeError("testability payload structure is invalid")
            testability_draft = TestabilityAssessmentDraft.model_validate(saved_draft)
            testability_assessment = TestabilityAssessment.model_validate(
                saved_assessment
            )
            testability_response = ModelResponse.model_validate(saved_response)
            testability_freshness = SnapshotFreshness.model_validate(saved_freshness)
        except (KeyError, TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved testability assessment is invalid."
            ) from error

        revalidated_assessment, testability_report = validate_testability_assessment(
            draft=testability_draft,
            human_review=review,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=testability_envelope,
        )
        if (
            revalidated_assessment is None
            or not testability_report.approved
            or revalidated_assessment != testability_assessment
            or testability_response.data != testability_draft.model_dump(mode="json")
            or testability_draft.evidence_envelope_sha256
            != testability_envelope.envelope_sha256
            or testability_assessment.evidence_envelope_sha256
            != testability_envelope.envelope_sha256
            or testability_freshness.snapshot_key != prepared.snapshot.snapshot_key
            or testability_freshness.status != "current"
        ):
            raise MilestoneTwoTransitionError(
                "The saved testability assessment is not bound to current "
                "reviewed frozen evidence."
            )

        self._testability_draft = testability_draft
        self._testability_response = testability_response
        self._testability_assessment = testability_assessment
        self._freshness = testability_freshness
        if testability_assessment.decision == "testable_from_frozen_evidence":
            self._state = _State.TESTABILITY_READY
        else:
            self._state = _State.EVIDENCE_REFINEMENT_REQUIRED
            return

        gherkin_envelope_item = self._load_durable_artifact(
            "artifacts/model_evidence/gherkin_generation.json"
        )
        gherkin_item = self._load_durable_artifact(
            "artifacts/workflow/gherkin_generation.json"
        )
        if gherkin_envelope_item is None:
            if gherkin_item is not None:
                raise MilestoneTwoTransitionError(
                    "The saved Gherkin stage is missing its evidence envelope."
                )
            return
        if gherkin_item is None:
            return

        gherkin_envelope_payload, _gherkin_envelope_event = gherkin_envelope_item
        try:
            gherkin_envelope = ModelEvidenceEnvelope.model_validate(
                gherkin_envelope_payload
            )
            build_gherkin_request(
                human_review=review,
                testability_assessment=testability_assessment,
                context=prepared.context,
                comparison_bindings=self._comparison_bindings(prepared),
                evidence_envelope=gherkin_envelope,
            )
            if (
                gherkin_envelope.max_request_body_bytes
                != self._settings.max_model_request_bytes
            ):
                raise ValueError("saved envelope uses a different request budget")
        except (TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved Gherkin evidence envelope is invalid."
            ) from error
        self._gherkin_evidence_envelope = gherkin_envelope

        gherkin_payload, _gherkin_event = gherkin_item
        try:
            saved_candidate = gherkin_payload["candidate"]
            saved_response = gherkin_payload["response"]
            saved_freshness = gherkin_payload["freshness"]
            if (
                not isinstance(saved_candidate, dict)
                or not isinstance(saved_response, dict)
                or not isinstance(saved_freshness, dict)
            ):
                raise TypeError("Gherkin payload structure is invalid")
            candidate = GherkinCandidate.from_persisted(saved_candidate)
            gherkin_response = ModelResponse.model_validate(saved_response)
            gherkin_freshness = SnapshotFreshness.model_validate(saved_freshness)
        except (KeyError, TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved Gherkin response is invalid."
            ) from error

        validation = validate_gherkin_candidate(
            candidate=candidate,
            human_review=review,
            testability_assessment=testability_assessment,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=gherkin_envelope,
        )
        if (
            not validation.approved
            or candidate.snapshot_key != prepared.snapshot.snapshot_key
            or candidate.reviewed_risk_sha256 != review.reviewed_content_sha256
            or candidate.approved_risk != review.reviewed_risk
            or candidate.evidence_envelope_sha256 != gherkin_envelope.envelope_sha256
            or gherkin_freshness.snapshot_key != prepared.snapshot.snapshot_key
            or gherkin_freshness.status != "current"
        ):
            raise MilestoneTwoTransitionError(
                "The saved Gherkin response is not bound to current reviewed evidence."
            )

        self._gherkin_candidate = candidate
        self._gherkin_response = gherkin_response
        self._gherkin_validation_report = validation
        self._validated_gherkin_text = candidate.gherkin_text
        self._freshness = gherkin_freshness
        self._state = _State.GHERKIN_READY

        evidence_gap_item = self._load_durable_artifact(
            "artifacts/workflow/gherkin_evidence_gap.json"
        )
        if evidence_gap_item is not None:
            evidence_gap_payload, _evidence_gap_event = evidence_gap_item
            try:
                source_candidate_sha256 = evidence_gap_payload[
                    "source_candidate_sha256"
                ]
                text = evidence_gap_payload["text"]
                saved_validation = evidence_gap_payload["validation"]
                saved_assessment = evidence_gap_payload["assessment"]
                saved_freshness = evidence_gap_payload["freshness"]
                if (
                    not isinstance(source_candidate_sha256, str)
                    or not isinstance(text, str)
                    or not isinstance(saved_validation, dict)
                    or not isinstance(saved_assessment, dict)
                    or not isinstance(saved_freshness, dict)
                ):
                    raise TypeError("Gherkin-evidence-gap payload structure is invalid")

                decision = saved_validation["decision"]
                approved = saved_validation["approved"]
                reason_codes = saved_validation["reason_codes"]
                if (
                    decision != "needs_more_frozen_evidence"
                    or approved is not False
                    or not isinstance(reason_codes, list)
                    or any(
                        not isinstance(reason_code, str) for reason_code in reason_codes
                    )
                ):
                    raise ValueError("Gherkin evidence-gap validation is invalid")

                evidence_gap_validation = GherkinValidationReport(
                    decision=decision,
                    approved=approved,
                    reason_codes=tuple(reason_codes),
                )
                evidence_gap_assessment = TestabilityAssessment.model_validate(
                    saved_assessment
                )
                evidence_gap_freshness = SnapshotFreshness.model_validate(
                    saved_freshness
                )
            except (KeyError, TypeError, ValueError) as error:
                raise MilestoneTwoTransitionError(
                    "The saved Gherkin evidence gap is invalid."
                ) from error

            revalidated_gap = validate_edited_gherkin(
                candidate=candidate,
                text=text,
                human_review=review,
                testability_assessment=testability_assessment,
                context=prepared.context,
                comparison_bindings=self._comparison_bindings(prepared),
                evidence_envelope=gherkin_envelope,
            )
            expected_gap_assessment = _gherkin_evidence_assessment(
                candidate=candidate,
                human_review=review,
                context=prepared.context,
                generated_at=evidence_gap_assessment.generated_at,
            )
            if (
                source_candidate_sha256
                != canonical_sha256(candidate.model_dump(mode="json"))
                or revalidated_gap != evidence_gap_validation
                or evidence_gap_assessment != expected_gap_assessment
                or evidence_gap_freshness.snapshot_key != prepared.snapshot.snapshot_key
                or evidence_gap_freshness.status != "current"
            ):
                raise MilestoneTwoTransitionError(
                    "The saved Gherkin evidence gap is not bound to current "
                    "reviewed frozen evidence."
                )

            self._testability_draft = None
            self._testability_response = None
            self._testability_assessment = evidence_gap_assessment
            self._gherkin_validation_report = evidence_gap_validation
            self._validated_gherkin_text = None
            self._freshness = evidence_gap_freshness
            self._state = _State.EVIDENCE_REFINEMENT_REQUIRED
            return

        validation_item = self._load_durable_artifact(
            "artifacts/workflow/gherkin_validation.json"
        )
        if validation_item is None:
            return

        validation_payload, _validation_event = validation_item
        try:
            source_candidate_sha256 = validation_payload["source_candidate_sha256"]
            saved_candidate = validation_payload["candidate"]
            saved_validation = validation_payload["validation"]
            saved_freshness = validation_payload["freshness"]
            if (
                not isinstance(source_candidate_sha256, str)
                or not isinstance(saved_candidate, dict)
                or not isinstance(saved_validation, dict)
                or not isinstance(saved_freshness, dict)
            ):
                raise TypeError("Gherkin-validation payload structure is invalid")

            edited_candidate = GherkinCandidate.from_persisted(saved_candidate)
            decision = saved_validation["decision"]
            approved = saved_validation["approved"]
            reason_codes = saved_validation["reason_codes"]
            if (
                decision != "valid_evidence_bound_gherkin"
                or approved is not True
                or not isinstance(reason_codes, list)
                or any(not isinstance(reason_code, str) for reason_code in reason_codes)
            ):
                raise ValueError("Gherkin validation result is invalid")

            edited_validation = GherkinValidationReport(
                decision=decision,
                approved=approved,
                reason_codes=tuple(reason_codes),
            )
            validation_freshness = SnapshotFreshness.model_validate(saved_freshness)
        except (KeyError, TypeError, ValueError) as error:
            raise MilestoneTwoTransitionError(
                "The saved Gherkin validation is invalid."
            ) from error

        revalidated_edit = validate_gherkin_candidate(
            candidate=edited_candidate,
            human_review=review,
            testability_assessment=testability_assessment,
            context=prepared.context,
            comparison_bindings=self._comparison_bindings(prepared),
            evidence_envelope=gherkin_envelope,
        )
        if (
            source_candidate_sha256
            != canonical_sha256(candidate.model_dump(mode="json"))
            or not revalidated_edit.approved
            or revalidated_edit != edited_validation
            or edited_candidate.snapshot_key != prepared.snapshot.snapshot_key
            or edited_candidate.context_sha256 != prepared.context.context_sha256
            or edited_candidate.reviewed_risk_sha256 != review.reviewed_content_sha256
            or edited_candidate.approved_risk != review.reviewed_risk
            or validation_freshness.snapshot_key != prepared.snapshot.snapshot_key
            or validation_freshness.status != "current"
        ):
            raise MilestoneTwoTransitionError(
                "The saved Gherkin validation is not bound to current reviewed "
                "frozen evidence."
            )

        self._gherkin_candidate = edited_candidate
        self._gherkin_validation_report = edited_validation
        self._validated_gherkin_text = edited_candidate.gherkin_text
        self._freshness = validation_freshness
        self._state = _State.GHERKIN_VALIDATED

    def _require_prepared(self, action: str) -> PreparedPullRequest:
        """Require the frozen snapshot/evidence stage before downstream actions."""
        if self._state is _State.NEW or self._prepared is None:
            raise MilestoneTwoTransitionError(
                f"Cannot {action}: prepare a pull request before continuing."
            )
        return self._prepared


def resume_milestone_two_workflow(
    *,
    run_handle: RunHandle,
    dependencies: MilestoneTwoDependencies,
) -> MilestoneTwoWorkflow:
    """Attach to one authenticated existing run without repeating external work."""
    if not isinstance(run_handle, RunHandle):
        raise TypeError("resume_milestone_two_workflow requires a RunHandle")
    if not isinstance(dependencies, MilestoneTwoDependencies):
        raise TypeError(
            "resume_milestone_two_workflow requires MilestoneTwoDependencies"
        )

    try:
        dependencies.recorder.verify_run_handle(run_handle)
    except (OSError, ValueError) as error:
        raise MilestoneTwoTransitionError(
            "The requested run handle could not be authenticated for recovery."
        ) from error

    workflow = MilestoneTwoWorkflow(
        run_id=run_handle.run_id,
        settings=dependencies.settings,
        recorder=dependencies.recorder,
        snapshot_acquirer=dependencies.snapshot_acquirer,
        diff_builder=dependencies.diff_builder,
        context_builder=dependencies.context_builder,
        store=dependencies.store,
        gateway=dependencies.gateway,
        evidence_refiner=dependencies.evidence_refiner,
        _run_handle=run_handle,
    )
    with dependencies.recorder.workflow_lease(run_handle):
        workflow._hydrate_durable_state()
    return workflow
