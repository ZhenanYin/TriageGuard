"""Testable state for the guided five-page Milestone 2 review UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from triageguard.config import PublicSettings
from triageguard.contracts import GherkinValidationReport
from triageguard.domain import (
    ContextRefinement,
    GherkinCandidate,
    HumanReviewedRisk,
    MilestoneTwoRunRecord,
    RiskAssessment,
    TestabilityAssessment,
)
from triageguard.workflow import MilestoneTwoWorkflow, PreparedPullRequest

_COMPARISON_LABELS = {
    "author_change": "Author change (M → H)",
    "integration_change": "Merge impact (B → C)",
    "base_drift_change": "Main-branch drift (M → B)",
    "repository_context": "Frozen repository context",
}


class PresentationTransitionError(RuntimeError):
    """The user attempted to move past a required workflow step."""


@dataclass
class MilestoneTwoAppState:
    """One human-guided Milestone 2 pull-request review session."""

    settings: PublicSettings
    workflow: MilestoneTwoWorkflow = field(repr=False)
    current_page: int = 1
    prepared: PreparedPullRequest | None = None
    risk_assessment: RiskAssessment | None = None
    selected_hypothesis_id: str | None = None
    human_reviewed_risk: HumanReviewedRisk | None = None
    testability_assessment: TestabilityAssessment | None = None
    gherkin_candidate: GherkinCandidate | None = None
    gherkin_validation_report: GherkinValidationReport | None = None
    latest_context_refinement: ContextRefinement | None = None
    edited_gherkin: str = ""
    terminal_record: MilestoneTwoRunRecord | None = None
    workflow_factory: Callable[[], MilestoneTwoWorkflow] | None = field(
        default=None,
        repr=False,
    )

    def go_next(self) -> None:
        """Advance only when the current page has completed its required action."""
        if self.current_page == 1:
            if self.prepared is None:
                raise PresentationTransitionError(
                    "analyze a pull request before continuing."
                )
            self.current_page = 2
            return
        if self.current_page == 2:
            if self.risk_assessment is None:
                raise PresentationTransitionError("propose risks before continuing.")
            self.current_page = 3
            return
        if self.current_page == 3:
            if self.selected_hypothesis_id is None:
                raise PresentationTransitionError("choose one risk before continuing.")
            self.current_page = 4
            return
        if self.current_page == 4:
            if self.human_reviewed_risk is None:
                raise PresentationTransitionError(
                    "save the reviewed risk before continuing."
                )
            self.current_page = 5
            return
        raise PresentationTransitionError("the last page has no next step.")

    def go_back(self) -> None:
        """Move back one page without discarding the frozen evidence."""
        if self.current_page == 1:
            raise PresentationTransitionError("the first page has no previous step.")
        self.current_page -= 1

    def analyze_pr(self, pr_url: str) -> PreparedPullRequest:
        """Freeze the submitted OpenMRS Core pull request for this session."""
        if self.prepared is not None:
            raise PresentationTransitionError(
                "a pull request has already been analyzed for this session."
            )
        self.prepared = self.workflow.prepare_pr(pr_url)
        return self.prepared

    def propose_risks(self) -> RiskAssessment:
        """Request model proposals and keep only locally grounded outcomes."""
        if self.prepared is None:
            raise PresentationTransitionError(
                "analyze a pull request before proposing risks."
            )
        if self.risk_assessment is not None:
            raise PresentationTransitionError(
                "risk proposals have already been created for this session."
            )
        self.risk_assessment = self.workflow.propose_risks()
        return self.risk_assessment

    def risk_review_view(self) -> dict[str, object]:
        """Return plain-language risk data without claiming a vulnerability exists."""
        assessment = self._require_assessment("review risks")
        anchors_by_id = {
            anchor.anchor_id: anchor for anchor in assessment.context_bundle.anchors
        }
        hypotheses = [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "paragraph": hypothesis.explanation,
                "actor": hypothesis.actor,
                "action": hypothesis.action,
                "protected_asset": hypothesis.protected_asset,
                "security_property": hypothesis.security_property,
                "expected_secure_behavior": hypothesis.expected_secure_behavior,
                "possible_failure": hypothesis.possible_failure,
                "observables": list(hypothesis.observables),
                "limitations": list(hypothesis.limitations),
                "missing_evidence": list(hypothesis.missing_evidence),
                "citation_anchor_ids": list(hypothesis.citation_anchor_ids),
                "comparison_labels": list(
                    dict.fromkeys(
                        _COMPARISON_LABELS[anchors_by_id[anchor_id].change_relation]
                        for anchor_id in hypothesis.citation_anchor_ids
                    )
                ),
            }
            for hypothesis in assessment.hypotheses
        ]
        return {
            "outcome": assessment.outcome,
            "hypotheses": hypotheses,
            "rationale": assessment.rationale,
            "coverage_limitations": list(assessment.coverage_limitations),
            "missing_evidence": list(assessment.missing_evidence),
            "needed_evidence": list(assessment.needed_evidence),
            "reason_code": assessment.reason_code,
        }

    def select_risk(self, hypothesis_id: str) -> None:
        """Select one locally grounded hypothesis for human review."""
        assessment = self._require_assessment("select a risk")
        if assessment.outcome != "risks_proposed":
            raise PresentationTransitionError(
                "a risk can be selected only when risks were proposed."
            )

        if not any(
            hypothesis.hypothesis_id == hypothesis_id
            for hypothesis in assessment.hypotheses
        ):
            raise ValueError("selected risk is not part of this assessment.")

        self.selected_hypothesis_id = hypothesis_id

    def save_risk_edits(
        self,
        edits: Mapping[str, object],
    ) -> HumanReviewedRisk:
        """Apply permitted human edits and record one reviewed risk."""
        assessment = self._require_assessment("save a reviewed risk")
        if self.selected_hypothesis_id is None:
            raise PresentationTransitionError(
                "choose one risk before saving the reviewed risk."
            )
        if self.human_reviewed_risk is not None:
            raise PresentationTransitionError(
                "a reviewed risk has already been saved for this session."
            )

        selected = next(
            hypothesis
            for hypothesis in assessment.hypotheses
            if hypothesis.hypothesis_id == self.selected_hypothesis_id
        )
        self.human_reviewed_risk = self.workflow.approve_risk(
            selected.hypothesis_id,
            dict(edits),
            tuple(selected.citation_anchor_ids),
        )
        return self.human_reviewed_risk

    def save_reviewed_hypothesis(self, paragraph: str) -> HumanReviewedRisk:
        """Save one readable review paragraph while preserving hidden evidence fields."""
        selected = self._selected_hypothesis_for_review()
        edited_paragraph = str(paragraph).strip()
        if not edited_paragraph:
            raise ValueError("reviewed risk hypothesis must not be empty.")
        edits: dict[str, object] = {}
        if edited_paragraph != selected.explanation:
            edits["explanation"] = edited_paragraph
        return self.save_risk_edits(edits)

    def assess_testability(self) -> TestabilityAssessment:
        """Check that the saved code can support a scenario before generating it."""
        if self.human_reviewed_risk is None:
            raise PresentationTransitionError(
                "save the reviewed risk before assessing testability."
            )
        if self.testability_assessment is not None:
            raise PresentationTransitionError(
                "testability has already been assessed for this reviewed risk."
            )
        self.testability_assessment = self.workflow.assess_testability()
        return self.testability_assessment

    def generate_gherkin(self) -> GherkinCandidate:
        """Generate one locally validated scenario for the reviewed risk."""
        if self.human_reviewed_risk is None:
            raise PresentationTransitionError(
                "save the reviewed risk before generating a scenario."
            )
        if (
            self.testability_assessment is None
            or self.testability_assessment.decision != "testable_from_frozen_evidence"
        ):
            raise PresentationTransitionError(
                "assess testability from frozen evidence before generating a scenario."
            )
        if self.gherkin_candidate is not None:
            raise PresentationTransitionError(
                "a scenario has already been generated for this session."
            )

        self.gherkin_candidate = self.workflow.generate_gherkin()
        self.edited_gherkin = self.gherkin_candidate.gherkin_text
        return self.gherkin_candidate

    def refine_frozen_evidence(self) -> ContextRefinement:
        """Search only the already frozen snapshots for a named evidence gap."""
        if (
            self.testability_assessment is None
            or self.testability_assessment.decision != "needs_more_frozen_evidence"
        ):
            raise PresentationTransitionError(
                "a testability decision needing frozen evidence is required first."
            )
        refinement = self.workflow.refine_frozen_evidence()
        self.latest_context_refinement = refinement
        if not refinement.exhausted:
            self.prepared = self.workflow.prepared_pull_request
            self.risk_assessment = None
            self.selected_hypothesis_id = None
            self.human_reviewed_risk = None
            self.testability_assessment = None
            self.gherkin_candidate = None
            self.gherkin_validation_report = None
            self.edited_gherkin = ""
            self.current_page = 2
        return refinement

    def finish_with_insufficient_frozen_evidence(self) -> MilestoneTwoRunRecord:
        """Seal an exhausted frozen-code search without calling the PR safe."""
        if (
            self.latest_context_refinement is None
            or not self.latest_context_refinement.exhausted
        ):
            raise PresentationTransitionError(
                "exhaust the frozen-evidence search before finishing this review."
            )
        if self.terminal_record is not None:
            raise PresentationTransitionError(
                "the workflow has already reached a terminal result."
            )
        self.terminal_record = self.workflow.finish_with_insufficient_frozen_evidence()
        return self.terminal_record

    def set_gherkin(self, text: str) -> None:
        """Keep a human-editable scenario draft until its final approval."""
        if self.gherkin_candidate is None:
            raise PresentationTransitionError("generate a scenario before editing it.")
        self.edited_gherkin = str(text)

    def validate_edited_gherkin(self) -> GherkinValidationReport:
        """Validate only a changed scenario before it can be approved."""
        if self.gherkin_candidate is None:
            raise PresentationTransitionError(
                "generate a scenario before validating an edit."
            )
        if self.edited_gherkin == self.gherkin_candidate.gherkin_text:
            raise PresentationTransitionError(
                "change the generated scenario before validating an edit."
            )

        report = self.workflow.validate_edited_gherkin(self.edited_gherkin)
        self.gherkin_validation_report = report
        if report.approved:
            self.gherkin_candidate = self.workflow.gherkin_candidate
        elif report.decision == "needs_more_frozen_evidence":
            self.testability_assessment = self.workflow.testability_assessment
        elif report.decision == "hypothesis_changed":
            self.human_reviewed_risk = None
            self.testability_assessment = None
            self.gherkin_candidate = None
            self.edited_gherkin = ""
            self.selected_hypothesis_id = None
            self.current_page = 3
        return report

    def approve_gherkin(self) -> MilestoneTwoRunRecord:
        """Approve the current editable scenario and seal the research record."""
        if self.gherkin_candidate is None:
            raise PresentationTransitionError(
                "generate a scenario before approving it."
            )
        if self.terminal_record is not None:
            raise PresentationTransitionError(
                "the workflow has already reached a terminal result."
            )
        if self.edited_gherkin != self.gherkin_candidate.gherkin_text and (
            self.gherkin_validation_report is None
            or not self.gherkin_validation_report.approved
        ):
            raise PresentationTransitionError(
                "validate the changed scenario before approving it."
            )

        self.terminal_record = self.workflow.approve_gherkin(self.edited_gherkin)
        self.gherkin_candidate = self.terminal_record.gherkin_candidate
        return self.terminal_record

    def finish_without_risk(self) -> MilestoneTwoRunRecord:
        """Seal only the two model outcomes that explicitly stop before Gherkin."""
        assessment = self._require_assessment("finish without risk")
        if assessment.outcome == "risks_proposed":
            raise PresentationTransitionError(
                "select and review a proposed risk instead of finishing without risk."
            )
        if self.terminal_record is not None:
            raise PresentationTransitionError(
                "the workflow has already reached a terminal result."
            )

        self.terminal_record = self.workflow.finish_without_risk()
        return self.terminal_record

    def scenario_view(self) -> dict[str, object]:
        """Return only safe, human-readable scenario data for page five."""
        if self.gherkin_candidate is None:
            return {
                "available": False,
                "gherkin_text": "",
                "feature_title": None,
                "scenario_title": None,
            }
        return {
            "available": True,
            "gherkin_text": self.edited_gherkin,
            "feature_title": self.gherkin_candidate.feature_title,
            "scenario_title": self.gherkin_candidate.scenario_title,
        }

    def testability_view(self) -> dict[str, object]:
        """Return the testability decision in reviewer-facing language."""
        assessment = self.testability_assessment
        if assessment is None:
            return {
                "available": False,
                "decision": None,
                "explanation": None,
                "evidence_needs": [],
            }
        return {
            "available": True,
            "decision": assessment.decision,
            "explanation": assessment.explanation,
            "evidence_needs": [
                {
                    "category": need.category,
                    "search_terms": list(need.search_terms),
                    "explanation": need.explanation,
                }
                for need in assessment.evidence_needs
            ],
        }

    def terminal_view(self) -> dict[str, object]:
        """Return the terminal outcome without inventing severity or execution facts."""
        if self.terminal_record is None:
            return {
                "available": False,
                "status": None,
                "reason_code": None,
                "explanation": None,
            }
        return {
            "available": True,
            "status": self.terminal_record.status.value,
            "reason_code": self.terminal_record.reason_code,
            "explanation": self.terminal_record.explanation,
        }

    def provider_view(self) -> dict[str, str]:
        """Expose model provenance without exposing secrets."""
        if self.settings.llm_mode == "replay":
            return {
                "mode": "Replay",
                "provider": "replay",
                "model": "replay/openai-gpt-oss-120b",
            }
        return {
            "mode": "Live",
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
        }

    def start_new_analysis(self) -> MilestoneTwoAppState:
        """Create a fresh workflow without reusing this run's frozen evidence."""
        if self.workflow_factory is None:
            raise PresentationTransitionError(
                "this UI state cannot create a new analysis workflow."
            )
        return MilestoneTwoAppState(
            settings=self.settings,
            workflow=self.workflow_factory(),
            workflow_factory=self.workflow_factory,
        )

    def _require_assessment(self, action: str) -> RiskAssessment:
        """Return the one assessment required by a later human action."""
        if self.risk_assessment is None:
            raise PresentationTransitionError(
                f"propose risks before attempting to {action}."
            )
        return self.risk_assessment

    def _selected_hypothesis_for_review(self):
        """Return the exact selected hypothesis whose readable text is editable."""
        assessment = self._require_assessment("save a reviewed risk")
        if self.selected_hypothesis_id is None:
            raise PresentationTransitionError(
                "choose one risk before saving the reviewed risk."
            )
        return next(
            hypothesis
            for hypothesis in assessment.hypotheses
            if hypothesis.hypothesis_id == self.selected_hypothesis_id
        )
