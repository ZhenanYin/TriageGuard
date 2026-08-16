"""Testable Streamlit composition for the real Milestone 1 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triageguard.config import (
    MAX_REPEAT_COUNT,
    MIN_REPEAT_COUNT,
    PublicSettings,
    Settings,
)
from triageguard.contracts import AlignmentReport, validate_gherkin_alignment
from triageguard.domain import RunRecord
from triageguard.generation import CodeValidationReport
from triageguard.llm import GroqStructuredGateway
from triageguard.research import ArtifactRecorder
from triageguard.ui.presentation import (
    cvss_source_label,
    guided_progress,
    observation_row,
    result_message,
    severity_card_data,
)
from triageguard.workflow import (
    GeneratedWorkflow,
    MilestoneOneWorkflow,
    PreparedWorkflow,
    UnsafeGeneratedCodeError,
    WorkflowTransitionError,
    build_replay_workflow,
)

CONTROLLED_FIXTURE_WARNING = (
    "Controlled OpenMRS-shaped fixture only — not a real OpenMRS revision and "
    "not publication evidence."
)
ANALYSIS_EXPLANATION = (
    "TriageGuard evaluates whether a proposed change weakens patient-deletion "
    "authorization by converting the selected risk into an executable security "
    "test."
)
ANALYSIS_SOURCE_LABEL = "Analysis source: controlled example"
_SESSION_STATE_KEY = "triageguard_milestone_one_state"
_FIXTURE_RELATIVE_PATH = Path("fixtures") / "patient_delete_authorization"


class ContractEditError(ValueError):
    """The edited Gherkin no longer expresses the frozen risk contract."""


class PresentationTransitionError(RuntimeError):
    """The wizard cannot move without its presentation and workflow gate."""


class TerminalRunError(RuntimeError):
    """A finalized run cannot accept another state-changing operation."""


@dataclass
class AppState:
    """Long-lived workflow objects retained across Streamlit reruns."""

    settings: PublicSettings
    fixture_directory: Path
    workflow: MilestoneOneWorkflow
    prepared: PreparedWorkflow
    current_page: int = 1
    edited_gherkin: str = ""
    risk_accepted: bool = False
    approved: bool = False
    generated: GeneratedWorkflow | None = None
    result: RunRecord | None = None
    validation_report: CodeValidationReport | None = None
    generation_error: dict[str, Any] | None = None
    execution_error: dict[str, Any] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "current_page" and (
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4
        ):
            raise PresentationTransitionError(
                "wizard page must be an integer from 1 through 4"
            )
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        if isinstance(self.settings, Settings):
            self.settings = self.settings.public_view()
        if not self.edited_gherkin:
            self.edited_gherkin = self.prepared.gherkin

    @property
    def mode_label(self) -> str:
        if self.settings.llm_mode == "replay":
            return "Replay (prerecorded model outputs)"
        return "Live (Groq structured generation)"

    @property
    def provider(self) -> str:
        return self.workflow.model_gateway_provider

    @property
    def model(self) -> str:
        return self.workflow.model_gateway_model

    @property
    def warning(self) -> str:
        return CONTROLLED_FIXTURE_WARNING

    @property
    def terminal(self) -> bool:
        return self.result is not None

    def go_next(self) -> None:
        """Advance only when the corresponding security workflow gate is met."""
        if self.current_page == 1:
            self.current_page = 2
            return
        if self.current_page == 2:
            if not self.risk_accepted:
                raise PresentationTransitionError(
                    "risk selection is required before reviewing the test"
                )
            self.current_page = 3
            return
        if self.current_page == 3:
            if self.generated is None or not self.generated.validation.approved:
                raise PresentationTransitionError(
                    "a validated test is required before running the comparison"
                )
            self.current_page = 4
            return
        raise PresentationTransitionError("the last page has no next step")

    def go_back(self) -> None:
        """Move one page back without mutating durable workflow state."""
        if self.current_page == 1:
            raise PresentationTransitionError("the first page has no previous step")
        self.current_page -= 1

    def impact_view_data(self) -> dict[str, Any]:
        """Return fixture provenance and the curated whole-path impact story."""
        report = self.prepared.impact_report
        return {
            "report_id": report.report_id,
            "classification": report.classification,
            "purpose": report.purpose,
            "affected_workflow": report.affected_workflow,
            "publication_status": report.publication_status,
            "evidence_scope": report.evidence_scope,
            "impact_path": [
                "controlled authorization mutation",
                "patient deletion workflow",
                "clerk authorization boundary",
                "base-versus-candidate execution",
            ],
        }

    def risks_view_data(self) -> list[dict[str, Any]]:
        """Expose the one evidence-supported fixture risk without inventing findings."""
        contract = self.prepared.contract
        report = self.prepared.impact_report
        return [
            {
                "contract_id": contract.contract_id,
                "hypothesis": (
                    f"If the candidate changes the {contract.actor} authorization "
                    "boundary, a patient deletion may succeed without the required "
                    "privilege."
                ),
                "actor": contract.actor,
                "actor_privileges": list(contract.actor_privileges),
                "missing_privileges": list(contract.missing_privileges),
                "action": contract.action,
                "secure_expectation": contract.secure_expectation,
                "observable_evidence": list(contract.observable_evidence),
                "provenance": {
                    "report_id": report.report_id,
                    "classification": report.classification,
                    "affected_workflow": report.affected_workflow,
                },
                "limitations": [report.evidence_scope, report.publication_status],
            }
        ]

    def set_gherkin(self, text: str) -> AlignmentReport:
        """Retain a proposed edit and return deterministic semantic alignment."""
        self.edited_gherkin = str(text)
        return validate_gherkin_alignment(self.prepared.contract, self.edited_gherkin)

    def accept_risk(self) -> None:
        """Record the user's selection without bypassing contract approval."""
        if self.terminal:
            raise TerminalRunError("Run is finalized. Start New Run to continue.")
        self.risk_accepted = True

    def contract_view_data(self) -> dict[str, Any]:
        """Return the frozen contract plus the current human-editable Gherkin view."""
        contract = self.prepared.contract
        alignment = validate_gherkin_alignment(contract, self.edited_gherkin)
        return {
            "contract_id": contract.contract_id,
            "actor": contract.actor,
            "actor_privileges": list(contract.actor_privileges),
            "missing_privileges": list(contract.missing_privileges),
            "preconditions": list(contract.preconditions),
            "action": contract.action,
            "secure_expectation": contract.secure_expectation,
            "observable_evidence": list(contract.observable_evidence),
            "base_expectation": contract.base_expectation,
            "candidate_expectation": contract.candidate_expectation,
            "cleanup": list(contract.cleanup),
            "original_gherkin": self.prepared.gherkin,
            "edited_gherkin": self.edited_gherkin,
            "alignment": {
                "approved": alignment.approved,
                "reason_codes": list(alignment.reason_codes),
                "matched_steps": dict(alignment.matched_steps),
            },
            "approval_enabled": alignment.approved and not self.approved,
            "approved": self.approved,
        }

    def approve_contract(self) -> None:
        """Approve only after the local alignment gate succeeds."""
        alignment = validate_gherkin_alignment(
            self.prepared.contract, self.edited_gherkin
        )
        if not alignment.approved:
            raise ContractEditError("edited Gherkin is not aligned with the contract")
        self.workflow.approve_contract(self.prepared.contract, self.edited_gherkin)
        self.approved = True

    def generate_test(self) -> GeneratedWorkflow:
        """Generate through the real workflow and retain a safe failure summary."""
        if self.terminal:
            raise TerminalRunError("Run is finalized. Start New Run to continue.")
        if not self.approved:
            raise RuntimeError("contract approval is required before generation")
        try:
            generated = self.workflow.generate()
        except Exception as error:
            if isinstance(error, UnsafeGeneratedCodeError):
                self.validation_report = CodeValidationReport.model_validate(
                    error.report.model_dump(mode="json")
                )
            self._capture_finalized_result()
            self.generation_error = _safe_stage_error("generation", error)
            raise
        self.generated = generated
        self.validation_report = generated.validation
        self.generation_error = None
        return generated

    def approve_and_generate(self) -> GeneratedWorkflow:
        """Run the existing approval and generation gates for the guided action."""
        if not self.risk_accepted:
            raise RuntimeError("risk selection is required before generation")
        if not self.approved:
            self.approve_contract()
        return self.generate_test()

    def test_view_data(self) -> dict[str, Any]:
        """Return provider metadata and complete generated/validated artifacts."""
        generated = self.generated
        validation = (
            generated.validation if generated is not None else self.validation_report
        )
        return {
            "available": generated is not None,
            "mode": self.mode_label,
            "provider": self.provider,
            "model": self.model,
            "plan": (
                generated.plan.model_dump(mode="json")
                if generated is not None
                else None
            ),
            "generated_source": (
                generated.generated.code if generated is not None else None
            ),
            "validation": (
                validation.model_dump(mode="json") if validation is not None else None
            ),
            "generation_error": self.generation_error,
            "generation_enabled": (
                self.approved and generated is None and not self.terminal
            ),
            "terminal_result": _terminal_result_data(self),
            "provider_provenance": self.provider_provenance(),
        }

    def execute(self, *, repeat_count: int) -> RunRecord:
        """Run bounded paired experiments through the actual workflow."""
        if self.terminal:
            raise TerminalRunError("Run is finalized. Start New Run to continue.")
        if self.generated is None:
            raise RuntimeError("validated generated test is required before execution")
        try:
            result = self.workflow.execute(repeat_count=repeat_count)
        except Exception as error:
            self._capture_finalized_result()
            self.execution_error = _safe_stage_error("execution", error)
            raise
        self.result = result
        self.execution_error = None
        return result

    def evidence_view_data(self) -> dict[str, Any]:
        """Return raw observations, classification, stability, and local paths."""
        if self.result is None:
            return {
                "available": False,
                "differential_available": False,
                "execution_enabled": self.generated is not None,
                "execution_error": self.execution_error,
                "severity_assessment": None,
            }
        common = {
            "available": True,
            "status": self.result.status.value,
            "reason_code": self.result.reason_code,
            "explanation": self.result.explanation,
            "artifact_paths": _artifact_paths(self),
            "execution_error": self.execution_error,
            "execution_enabled": False,
            "severity_assessment": (
                self.result.severity_assessment.model_dump(mode="json")
                if self.result.severity_assessment is not None
                else None
            ),
        }
        if self.result.differential_evidence is None:
            return {**common, "differential_available": False}
        evidence = self.result.differential_evidence
        return {
            **common,
            "differential_available": True,
            "base": evidence.base.model_dump(mode="json"),
            "candidate": evidence.candidate.model_dump(mode="json"),
            "repetitions": evidence.repetitions,
            "stable": evidence.stable,
            "base_differing_run_indexes": list(evidence.base_differing_run_indexes),
            "candidate_differing_run_indexes": list(
                evidence.candidate_differing_run_indexes
            ),
        }

    def provider_provenance(self) -> dict[str, str | None]:
        """Separate configured provider, active gateway, and recorded response."""
        recorded_provider = (
            self.generated.generated.model_response.provider
            if self.generated is not None
            else None
        )
        recorded_model = (
            self.generated.generated.model_response.model
            if self.generated is not None
            else None
        )
        return {
            "configured_live_provider": self.settings.llm_provider,
            "configured_live_model": self.settings.llm_model,
            "active_gateway_provider": self.provider,
            "active_gateway_model": self.model,
            "recorded_response_provider": recorded_provider,
            "recorded_response_model": recorded_model,
        }

    def start_new_run(self) -> AppState:
        """Create a fresh prepared workflow without mutating this run."""
        if self.settings.llm_mode == "live":
            runtime_settings = Settings.from_env()
        else:
            runtime_settings = Settings(
                llm_mode="replay",
                llm_provider=self.settings.llm_provider,
                llm_model=self.settings.llm_model,
                artifacts_dir=self.settings.artifacts_dir,
                repeat_count=self.settings.repeat_count,
                environment_kind=self.settings.environment_kind,
            )
        return create_app_state(runtime_settings, self.fixture_directory)

    def _capture_finalized_result(self) -> None:
        """Retain only a recorder-finalized result; leave recoverable errors open."""
        try:
            result = self.workflow.result()
        except WorkflowTransitionError:
            return
        self.result = result


def create_app_state(settings: Settings, fixture_dir: str | Path) -> AppState:
    """Construct and prepare one real workflow without starting Streamlit."""
    fixture_directory = Path(fixture_dir)
    if settings.llm_mode == "replay":
        workflow = build_replay_workflow(
            artifact_root=settings.artifacts_dir,
            fixture_directory=fixture_directory,
            settings=settings,
        )
    else:
        workflow = MilestoneOneWorkflow(
            fixture_directory=fixture_directory,
            settings=settings,
            gateway=GroqStructuredGateway(settings),
            recorder=ArtifactRecorder(settings.artifacts_dir),
        )
    prepared = workflow.prepare()
    return AppState(
        settings=settings.public_view(),
        fixture_directory=fixture_directory,
        workflow=workflow,
        prepared=prepared,
    )


def _safe_stage_error(stage: str, error: Exception) -> dict[str, Any]:
    """Describe a stage failure without exposing provider text or credentials."""
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": f"{stage.capitalize()} did not complete. No fallback was used.",
        "fallback_used": False,
    }


def _terminal_result_data(state: AppState) -> dict[str, Any] | None:
    if state.result is None:
        return None
    return {
        "status": state.result.status.value,
        "reason_code": state.result.reason_code,
        "explanation": state.result.explanation,
        "artifact_paths": _artifact_paths(state),
    }


def _artifact_paths(state: AppState) -> dict[str, Any]:
    run_directory = (state.settings.artifacts_dir / state.prepared.run_id).resolve()
    execution_directory = run_directory / "executions"
    execution_files = (
        sorted(
            str(path.resolve())
            for path in execution_directory.rglob("*")
            if path.is_file()
        )
        if execution_directory.is_dir()
        else []
    )
    return {
        "run_directory": str(run_directory),
        "run_record": str((run_directory / "run_record.json").resolve()),
        "event_log": str((run_directory / "events.jsonl").resolve()),
        "execution_files": execution_files,
    }


def _configuration_error_message(error: ValueError) -> str:
    """Map owned settings errors to fixed UI text without echoing input."""
    reason = str(error)
    if "TRIAGEGUARD_LLM_PROVIDER" in reason:
        return (
            "Configuration is invalid. Only the Groq provider is supported; "
            "no gateway or run was created."
        )
    if "repeat_count" in reason or "TRIAGEGUARD_REPEAT_COUNT" in reason:
        return (
            "Configuration is invalid. Repeat count must be an integer from 1 to "
            "20; no run was created."
        )
    if "GROQ_API_KEY" in reason:
        return (
            "Configuration is invalid. Live mode requires GROQ_API_KEY in the "
            "local environment; no provider call was made."
        )
    return "Configuration is invalid; no gateway or run was created."


def render_app(streamlit_module: Any | None = None) -> None:
    """Render the app; importing this module performs no Streamlit work."""
    if streamlit_module is None:
        import streamlit as streamlit_module

    st = streamlit_module
    st.set_page_config(page_title="TriageGuard", layout="wide")
    st.title("TriageGuard")
    try:
        settings = Settings.from_env()
    except ValueError as error:
        st.error(_configuration_error_message(error))
        return
    fixture_directory = Path(__file__).parents[3] / _FIXTURE_RELATIVE_PATH
    if _SESSION_STATE_KEY not in st.session_state:
        try:
            st.session_state[_SESSION_STATE_KEY] = create_app_state(
                settings, fixture_directory
            )
        except Exception:  # noqa: BLE001 - UI boundary must redact all preparation errors
            st.error(
                "The local workflow could not be prepared. Check the fixture and "
                "artifact directory configuration."
            )
            return
    state: AppState = st.session_state[_SESSION_STATE_KEY]
    _render_progress(st, state)
    st.caption(f"Step {state.current_page} of 4")
    page_renderers = {
        1: _render_demonstration_step,
        2: _render_risk_step,
        3: _render_test_step,
        4: _render_evidence_step,
    }
    page_renderers[state.current_page](st, state)
    if state.terminal:
        _render_start_new_run(st, state)


def _render_progress(st: Any, state: AppState) -> None:
    """Show read-only progress rather than a page-selection control."""
    st.sidebar.markdown("### Progress")
    prefixes = {"complete": "✅", "current": "➡️", "locked": "🔒"}
    for step in guided_progress(
        current_page=state.current_page,
        risk_accepted=state.risk_accepted,
        test_ready=state.generated is not None,
        terminal=state.terminal,
    ):
        st.sidebar.write(f"{prefixes[step.state]} {step.label}")


def _render_demonstration_step(st: Any, state: AppState) -> None:
    """Explain the controlled input without presenting internal fixture JSON."""
    st.header("1. Understand the change")
    st.markdown(ANALYSIS_EXPLANATION)
    st.caption(ANALYSIS_SOURCE_LABEL)
    base, candidate = st.columns(2)
    with base:
        st.markdown("#### Secure base")
        st.write("Expected to enforce the patient-deletion permission boundary.")
    with candidate:
        st.markdown("#### Candidate")
        st.write(
            "Intentionally changed behavior used to check whether TriageGuard "
            "can detect a regression."
        )
    report = state.impact_view_data()
    with st.expander("Technical details"):
        st.write(f"Fixture classification: {report['classification']}")
        st.write(f"Affected workflow: {report['affected_workflow']}")
        st.write(f"Evidence scope: {report['evidence_scope']}")
        st.write(f"Report ID: {report['report_id']}")
    if st.button("Next: Review risk", type="primary"):
        state.go_next()
        st.rerun()


def _render_risk_step(st: Any, state: AppState) -> None:
    """Present one prepared risk in language a non-programmer can review."""
    st.header("2. Review the security risk")
    risk = state.risks_view_data()[0]
    st.markdown("#### Possible authorization risk")
    st.markdown(
        "A clerk without **Delete Patients** permission might be able to delete "
        "a patient."
    )
    who, action, expected = st.columns(3)
    with who:
        st.markdown("**Who**")
        st.write("A clerk who can view patients but cannot delete them.")
    with action:
        st.markdown("**Action**")
        st.write("The clerk attempts to delete a patient.")
    with expected:
        st.markdown("**Secure behavior**")
        st.write("The request is denied and the patient still exists.")
    st.caption("Prepared demonstration scenario")
    if st.button(
        "Use this risk",
        disabled=state.risk_accepted or state.terminal,
        type="primary",
    ):
        try:
            state.accept_risk()
        except TerminalRunError:
            st.error("This run is finalized. Start a new run to continue.")
        else:
            st.success("Risk selected. Review the security scenario below.")
    if state.risk_accepted:
        st.success("Selected for this run.")
    with st.expander("Research details"):
        st.write(f"Contract ID: {risk['contract_id']}")
        st.write(f"Internal hypothesis: {risk['hypothesis']}")
        st.write(f"Missing privilege: {', '.join(risk['missing_privileges'])}")
        st.write(f"Evidence limitations: {', '.join(risk['limitations'])}")
    back, next_step = st.columns(2)
    with back:
        if st.button("Back"):
            state.go_back()
            st.rerun()
    with next_step:
        if st.button(
            "Next: Review test",
            disabled=not state.risk_accepted,
            type="primary",
        ):
            state.go_next()
            st.rerun()


def _render_test_step(st: Any, state: AppState) -> None:
    """Guide Gherkin review and keep generated internals optional."""
    st.header("3. Review and generate the test")
    st.write(
        "Given/When/Then describes the setup, action, and expected secure "
        "behavior in plain language."
    )
    st.markdown("#### What TriageGuard will test")
    edited = st.text_area(
        "Given / When / Then scenario",
        value=state.edited_gherkin,
        height=360,
        disabled=not state.risk_accepted or state.approved,
    )
    alignment = state.set_gherkin(edited)
    if state.risk_accepted and not alignment.approved:
        st.error(
            "The scenario changed the approved security meaning. Restore the "
            "actor, action, denial, and patient-state checks before continuing."
        )
    if st.button(
        "Approve scenario and generate test",
        disabled=(
            not state.risk_accepted
            or state.generated is not None
            or state.terminal
            or not alignment.approved
        ),
        type="primary",
    ):
        try:
            with st.spinner("Creating and checking the executable test..."):
                state.approve_and_generate()
        except ContractEditError:
            st.error(
                "The scenario changed the approved security meaning. Restore the "
                "required security checks before continuing."
            )
        except Exception:  # noqa: BLE001 - UI boundary redacts provider errors
            st.error("Test generation stopped. No fallback test was used.")
        else:
            st.success("Test generated and passed deterministic safety checks.")
    test_data = state.test_view_data()
    if test_data["generation_error"] is not None:
        st.error("Test generation stopped. No fallback test was used.")
    if test_data["validation"] is not None and not test_data["validation"]["approved"]:
        st.error("The generated test was rejected and will not be executed.")
    if test_data["available"]:
        st.success("Test generated and passed deterministic safety checks.")
    with st.expander("Generated test and model details"):
        st.write(f"Mode: {test_data['mode']}")
        st.write(f"Active provider: {test_data['provider']}")
        st.write(f"Active model: {test_data['model']}")
        if test_data["plan"] is not None:
            st.markdown("**Structured test plan**")
            st.json(test_data["plan"])
        if test_data["generated_source"] is not None:
            st.markdown("**Generated pytest-bdd source**")
            st.code(test_data["generated_source"], language="python", line_numbers=True)
        if test_data["validation"] is not None:
            st.markdown("**Deterministic validation report**")
            st.json(test_data["validation"])
        st.markdown("**Provider provenance**")
        st.json(test_data["provider_provenance"])
        if test_data["generation_error"] is not None:
            st.markdown("**Safe generation failure**")
            st.json(test_data["generation_error"])
    back, next_step = st.columns(2)
    with back:
        if st.button("Back"):
            state.go_back()
            st.rerun()
    with next_step:
        if st.button(
            "Next: Run comparison",
            disabled=(
                state.generated is None or not state.generated.validation.approved
            ),
            type="primary",
        ):
            state.go_next()
            st.rerun()


def _render_evidence_step(st: Any, state: AppState) -> None:
    """Explain paired evidence first and retain raw research details on demand."""
    st.header("4. Run the comparison")
    if st.button("Back"):
        state.go_back()
        st.rerun()
    with st.expander("Experiment settings"):
        repeat_count = st.number_input(
            "Paired repetitions",
            min_value=MIN_REPEAT_COUNT,
            max_value=MAX_REPEAT_COUNT,
            value=state.settings.repeat_count,
            step=1,
            disabled=state.terminal,
        )
    data = state.evidence_view_data()
    if state.generated is None:
        st.info("Generate and validate the test before running the comparison.")
    if st.button(
        "Run base vs candidate comparison",
        disabled=not data["execution_enabled"] or state.terminal,
        type="primary",
    ):
        try:
            with st.spinner("Running the approved test against both versions..."):
                state.execute(repeat_count=int(repeat_count))
        except Exception:  # noqa: BLE001 - UI boundary must redact all execution errors
            st.error(
                "Comparison did not complete. No security conclusion was inferred."
            )
        data = state.evidence_view_data()
    if data.get("execution_error") is not None:
        st.error("Comparison did not complete. No security conclusion was inferred.")
    if not data["available"]:
        return
    message = result_message(data["status"])
    st.subheader(message.title)
    getattr(st, message.level)(message.body)
    if data["differential_available"]:
        rows = [
            observation_row("Secure base", data["base"]),
            observation_row("Candidate", data["candidate"]),
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No complete base-versus-candidate table is available for this run.")
    _render_severity_assessment(st, data["severity_assessment"])

    with st.expander("Research evidence"):
        st.write(f"Status: {data['status']}")
        st.write(f"Reason code: {data['reason_code']}")
        st.write(f"Recorded explanation: {data['explanation']}")
        if data["differential_available"]:
            st.markdown("**Raw base observation**")
            st.json(data["base"])
            st.markdown("**Raw candidate observation**")
            st.json(data["candidate"])
            st.write(f"Repetitions: {data['repetitions']}")
            st.write(f"Stable: {data['stable']}")
            st.write(
                f"Base differing run indexes: {data['base_differing_run_indexes']}"
            )
            st.write(
                "Candidate differing run indexes: "
                f"{data['candidate_differing_run_indexes']}"
            )
        if data.get("execution_error") is not None:
            st.markdown("**Safe execution failure**")
            st.json(data["execution_error"])

    with st.expander("Local artifacts"):
        paths = data["artifact_paths"]
        st.write(f"Run directory: {paths['run_directory']}")
        st.write(f"Final run record: {paths['run_record']}")
        st.write(f"Event log: {paths['event_log']}")
        if paths["execution_files"]:
            st.write("Execution files:")
            for path in paths["execution_files"]:
                st.code(path, language=None)


def _render_severity_assessment(
    st: Any,
    assessment: dict[str, Any] | None,
) -> None:
    """Show evidence-gated CVSS claims without implying a secure-side zero."""
    st.markdown("### CVSS 4.0 severity")
    if assessment is None:
        st.info(
            "CVSS 4.0 was not calculated because this run did not produce "
            "complete differential evidence."
        )
        return

    base_card = severity_card_data("Secure base", assessment["base"])
    candidate_card = severity_card_data("Candidate", assessment["candidate"])
    for column, card in zip(
        st.columns(2),
        (base_card, candidate_card),
        strict=True,
    ):
        with column:
            st.markdown(f"#### {card['version']}")
            st.markdown(f"**{card['headline']}**")
            st.caption(card["label"])
            st.write(card["reason"])
            if card["vector"]:
                st.code(card["vector"], language=None)

    with st.expander("CVSS assessment details"):
        st.write(
            "This is a provisional, expert-authored CVSS 4.0 Base assessment. "
            "The test determines whether the vulnerability was observed; "
            "deployment assumptions and expert judgment were not measured by pytest."
        )
        for version_label, version_assessment in (
            ("Secure base", assessment["base"]),
            ("Candidate", assessment["candidate"]),
        ):
            st.markdown(f"#### {version_label}")
            if version_assessment["status"] != "provisional":
                st.write(
                    "No metric profile was applied because the tested "
                    "vulnerability was not observed."
                )
                continue
            st.caption(
                f"Calculator: {version_assessment['calculator']} · "
                f"Review status: {version_assessment['review_status']}"
            )
            for metric in version_assessment["metrics"]:
                source = cvss_source_label(metric["source_category"])
                st.markdown(
                    f"**{metric['metric']}: {metric['value']} — {source}** "
                    f"(`{metric['source_category']}`)"
                )
                st.write(metric["rationale"])
                st.caption("Sources: " + ", ".join(metric["source_references"]))


def _render_start_new_run(st: Any, state: AppState) -> None:
    if st.button("Start New Run", type="primary"):
        try:
            replacement = state.start_new_run()
        except Exception:  # noqa: BLE001 - UI boundary must redact preparation errors
            st.error("A new local run could not be prepared.")
            return
        st.session_state[_SESSION_STATE_KEY] = replacement
        st.rerun()


def main() -> None:
    """Streamlit script entry point."""
    render_app()


if __name__ == "__main__":
    main()
