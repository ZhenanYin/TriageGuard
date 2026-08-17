"""Five-page Streamlit interface for the human-guided Milestone 2 workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from triageguard.analysis import (
    ContextBuilder,
    ContextBuildError,
    DiffBuilder,
    DiffBuildError,
    FrozenContextRefiner,
    SnapshotAcquirer,
)
from triageguard.analysis.snapshot import SnapshotAcquisitionError
from triageguard.config import PublicSettings, Settings
from triageguard.llm import GroqStructuredGateway
from triageguard.research import ArtifactRecorder
from triageguard.sources.git import GitObjectStore
from triageguard.sources.github import GitHubClient
from triageguard.ui.milestone_two import (
    MilestoneTwoAppState,
    PresentationTransitionError,
)
from triageguard.ui.milestone_two_presentation import (
    COMMIT_ID_EXPLANATION,
    comparison_cards,
    freshness_label,
    guided_progress,
)
from triageguard.workflow import MilestoneTwoWorkflow
from triageguard.workflow.milestone_two_replay import (
    build_milestone_two_replay_workflow,
)

_SESSION_STATE_KEY = "triageguard_milestone_two_state"
_SUPPORTED_PR_URL = "https://github.com/openmrs/openmrs-core/pull/900000001"


def _preparation_error_message(error: Exception) -> str:
    """Return a typed preparation reason without exposing implementation details."""
    if isinstance(error, (ContextBuildError, DiffBuildError, SnapshotAcquisitionError)):
        return f"Preparation stopped ({error.reason_code}): {error.safe_message}"
    return "The pull request could not be prepared for review."


def _initial_pull_request_url(state: MilestoneTwoAppState) -> str:
    """Return the one safe initial URL for the selected analysis mode."""
    if state.prepared is not None:
        return state.prepared.snapshot.pull_url
    if state.settings.llm_mode == "replay":
        return _SUPPORTED_PR_URL
    return ""


def create_milestone_two_app_state(settings: Settings) -> MilestoneTwoAppState:
    """Create replay or live workflow state while retaining public settings only."""
    public_settings = settings.public_view()

    if settings.llm_mode == "replay":
        workflow = build_milestone_two_replay_workflow(settings)

        def workflow_factory() -> MilestoneTwoWorkflow:
            return build_milestone_two_replay_workflow(
                _replay_settings_from_public(public_settings)
            )

    else:
        workflow = _build_live_workflow(settings)

        def workflow_factory() -> MilestoneTwoWorkflow:
            runtime_settings = Settings.from_env()
            if runtime_settings.llm_mode != "live":
                raise ValueError(
                    "live analysis cannot restart while replay mode is configured."
                )
            return _build_live_workflow(runtime_settings)

    return MilestoneTwoAppState(
        settings=public_settings,
        workflow=workflow,
        workflow_factory=workflow_factory,
    )


def _replay_settings_from_public(public: PublicSettings) -> Settings:
    """Recreate replay-only runtime settings without any credential field."""
    return Settings(
        llm_mode="replay",
        llm_provider=public.llm_provider,
        llm_model=public.llm_model,
        artifacts_dir=public.artifacts_dir,
        github_api_version=public.github_api_version,
        analysis_cache_dir=public.analysis_cache_dir,
        max_diff_files=public.max_diff_files,
        max_diff_bytes=public.max_diff_bytes,
        max_context_files=public.max_context_files,
        max_context_anchors=public.max_context_anchors,
        max_context_bytes=public.max_context_bytes,
        max_context_anchor_lines=public.max_context_anchor_lines,
        max_context_blob_bytes=public.max_context_blob_bytes,
        max_context_search_identifiers=public.max_context_search_identifiers,
        max_context_hits_per_identifier=public.max_context_hits_per_identifier,
        repeat_count=public.repeat_count,
        environment_kind=public.environment_kind,
    )


def _build_live_workflow(settings: Settings) -> MilestoneTwoWorkflow:
    """Create the real read-only GitHub/Git/Groq dependency graph for live mode."""
    run_id = f"m2-live-{uuid4().hex}"
    store = GitObjectStore(
        settings.analysis_cache_dir / "milestone-two-live" / f"{run_id}.git"
    )
    store.initialize()

    return MilestoneTwoWorkflow(
        run_id=run_id,
        settings=settings,
        recorder=ArtifactRecorder(settings.artifacts_dir),
        snapshot_acquirer=SnapshotAcquirer(
            github=GitHubClient(
                token=settings.github_token,
                api_version=settings.github_api_version,
            ),
            store=store,
            settings=settings,
        ),
        diff_builder=DiffBuilder(
            store,
            max_files=settings.max_diff_files,
            max_bytes=settings.max_diff_bytes,
        ),
        context_builder=ContextBuilder(),
        store=store,
        gateway=GroqStructuredGateway(settings),
        evidence_refiner=FrozenContextRefiner(),
    )


def render_app() -> None:
    """Render one current page of the guided review, never all pages at once."""
    import streamlit as st

    st.set_page_config(
        page_title="TriageGuard",
        page_icon="🛡️",
        layout="wide",
    )
    state = _load_state(st)
    if state is None:
        return

    st.title("TriageGuard")
    st.caption("Guided security-review evidence for OpenMRS Core pull-request changes.")
    _render_progress(st, state)

    if state.current_page == 1:
        _render_choose_pull_request(st, state)
    elif state.current_page == 2:
        _render_understand_change(st, state)
    elif state.current_page == 3:
        _render_review_risks(st, state)
    elif state.current_page == 4:
        _render_edit_risk(st, state)
    else:
        _render_scenario(st, state)


def _load_state(st: Any) -> MilestoneTwoAppState | None:
    """Load the existing safe UI state or create a replay state for this session."""
    try:
        settings = Settings.from_env()
    except ValueError as error:
        st.error(str(error))
        return None

    existing = st.session_state.get(_SESSION_STATE_KEY)
    if isinstance(existing, MilestoneTwoAppState):
        return existing

    try:
        state = create_milestone_two_app_state(settings)
    except Exception:  # noqa: BLE001 - UI must not expose implementation details
        st.error("The local replay workflow could not be started.")
        return None

    st.session_state[_SESSION_STATE_KEY] = state
    return state


def _render_progress(st: Any, state: MilestoneTwoAppState) -> None:
    """Show five short steps without rendering their later page bodies."""
    with st.sidebar:
        st.markdown("### Review progress")
        for index, step in enumerate(
            guided_progress(state.current_page),
            start=1,
        ):
            marker = {
                "complete": "✓",
                "current": "→",
                "locked": "•",
            }[step.state]
            st.caption(f"{marker} {index}. {step.label}")

        provider = state.provider_view()
        st.markdown("### Analysis mode")
        st.caption(f"{provider['mode']} · {provider['provider']}")
        st.caption(provider["model"])
        st.caption("Milestone 2 stops after approved Gherkin; it does not run pytest.")


def _render_choose_pull_request(st: Any, state: MilestoneTwoAppState) -> None:
    """Render page one: freeze the exact pull-request evidence."""
    st.header("1. Choose a pull request")
    st.write(
        "Paste an OpenMRS Core pull-request URL. TriageGuard will save four "
        "exact code photographs before asking the model for possible risks."
    )
    if state.settings.llm_mode == "replay":
        st.info(
            "This default replay demo uses a synthetic OpenMRS-shaped pull "
            "request. It does not contact GitHub or an LLM."
        )

    pr_url = st.text_input(
        "OpenMRS Core pull-request URL",
        value=_initial_pull_request_url(state),
    )
    if st.button(
        "Analyze pull request",
        disabled=state.prepared is not None,
        type="primary",
    ):
        try:
            with st.spinner("Freezing the pull request and its code context..."):
                state.analyze_pr(pr_url)
        except Exception as error:  # noqa: BLE001 - safe boundary for URL/Git/model errors
            st.error(_preparation_error_message(error))
        else:
            st.success("Pull request frozen. You can now inspect the change.")

    if state.prepared is not None:
        _render_freshness(st, state)
        st.success(
            "Four code photographs are ready: shared starting point, current "
            "main branch, pull-request branch, and merge preview."
        )

    _render_navigation(
        st,
        state,
        next_disabled=state.prepared is None,
    )


def _render_understand_change(st: Any, state: MilestoneTwoAppState) -> None:
    """Render page two: explain M, B, H, and C without raw Git terminology."""
    prepared = _require_prepared(st, state)
    if prepared is None:
        return

    st.header("2. Understand the change")
    st.write(
        "TriageGuard compares four saved code photographs so it can examine the "
        "author's change, the current main branch, and what a merge would add."
    )
    _render_freshness(st, state)

    snapshot = prepared.snapshot
    st.dataframe(
        [
            {
                "Photo": "M — shared starting point",
                "Meaning": "Where the pull request and main branch last matched",
                "Commit": snapshot.merge_base_sha[:12],
            },
            {
                "Photo": "B — current main",
                "Meaning": "What OpenMRS Core looks like now",
                "Commit": snapshot.base_sha[:12],
            },
            {
                "Photo": "H — pull request",
                "Meaning": "What the author proposes",
                "Commit": snapshot.head_sha[:12],
            },
            {
                "Photo": "C — merge preview",
                "Meaning": "What main would look like if merged now",
                "Commit": snapshot.candidate_sha[:12],
            },
        ],
        hide_index=True,
        width="stretch",
    )

    st.caption(COMMIT_ID_EXPLANATION)
    st.markdown("#### The three comparisons")
    columns = st.columns(3)
    for column, card in zip(columns, comparison_cards(), strict=True):
        with column, st.container(border=True):
            st.subheader(f"{card.title} — {card.comparison}")
            st.write(card.explanation)

    if st.button(
        "Propose possible risks",
        disabled=state.risk_assessment is not None,
        type="primary",
    ):
        try:
            with st.spinner("Generating and locally checking risk proposals..."):
                state.propose_risks()
        except Exception:  # noqa: BLE001 - safe boundary, never invent a fallback
            st.error("Risk proposals could not be created from this evidence.")
        else:
            st.success("Risk outcome is ready for your review.")

    if state.risk_assessment is not None:
        outcome = state.risk_assessment.outcome
        if outcome == "no_meaningful_security_risk_found":
            st.info(
                "No specific testable risk was proposed from the bounded evidence. "
                "This is not proof that the change is safe."
            )
        elif outcome == "insufficient_context_to_assess":
            st.warning(
                "The bounded evidence was insufficient to assess this change. "
                "Review the listed evidence gap before ending the run."
            )

    with st.expander("Technical evidence and model provenance"):
        st.json(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "diffs": [diff.model_dump(mode="json") for diff in prepared.diffs],
                "context": prepared.context.model_dump(mode="json"),
                "provider": state.provider_view(),
            }
        )

    _render_navigation(
        st,
        state,
        next_disabled=state.risk_assessment is None,
    )


def _render_review_risks(st: Any, state: MilestoneTwoAppState) -> None:
    """Render page three: show unconfirmed proposals and require one selection."""
    st.header("3. Review possible risks")
    st.write(
        "These are unconfirmed risk hypotheses, not confirmed vulnerabilities. "
        "Choose one only if its explanation and evidence make sense to you."
    )
    freshness = _render_freshness(st, state)
    view = state.risk_review_view()

    if view["outcome"] != "risks_proposed":
        _render_nonrisk_outcome(st, state, view, freshness)
        _render_navigation(st, state, next_disabled=True)
        return

    for hypothesis in view["hypotheses"]:
        with st.container(border=True):
            st.subheader(str(hypothesis["title"]))
            st.write(str(hypothesis["paragraph"]))
            st.caption(
                "This is a hypothesis. It still needs a human decision and "
                "an executable scenario."
            )
            if st.button(
                "Choose this risk",
                key=str(hypothesis["hypothesis_id"]),
                disabled=freshness != "current",
                type="primary",
            ):
                try:
                    state.select_risk(str(hypothesis["hypothesis_id"]))
                except Exception:  # noqa: BLE001 - safe UI boundary
                    st.error("That risk could not be selected.")
                else:
                    st.success("Risk selected. Continue to review and edit it.")

            with st.expander("Why this was suggested"):
                st.write(
                    "The proposal is grounded in saved comparison evidence and "
                    "is not a confirmed vulnerability."
                )
                st.write("Relevant comparisons:")
                for comparison_label in hypothesis["comparison_labels"]:
                    st.write(f"• {comparison_label}")
                st.write("Frozen evidence anchors:")
                for anchor_id in hypothesis["citation_anchor_ids"]:
                    st.write(f"• {anchor_id}")
                st.json(hypothesis)

    _render_navigation(
        st,
        state,
        next_disabled=state.selected_hypothesis_id is None,
    )


def _render_edit_risk(st: Any, state: MilestoneTwoAppState) -> None:
    """Render page four: preserve a human-reviewed risk before Gherkin work."""
    st.header("4. Choose and edit one risk")
    st.write(
        "You may clarify the approved risk before creating a scenario. The "
        "citation anchors stay fixed so the reviewed version remains tied to "
        "the frozen pull-request evidence."
    )
    freshness = _render_freshness(st, state)
    view = state.risk_review_view()
    selected = next(
        (
            hypothesis
            for hypothesis in view["hypotheses"]
            if hypothesis["hypothesis_id"] == state.selected_hypothesis_id
        ),
        None,
    )
    if selected is None:
        st.error("Return to the previous page and choose a risk.")
        _render_navigation(st, state, next_disabled=True)
        return

    paragraph = st.text_area(
        "Reviewed risk hypothesis",
        value=str(selected["paragraph"]),
        height=210,
        disabled=state.human_reviewed_risk is not None,
        help=(
            "Edit this readable, unconfirmed hypothesis. Its frozen evidence "
            "anchors remain fixed and are shown below."
        ),
    )

    if st.button(
        "Save reviewed risk",
        disabled=state.human_reviewed_risk is not None or freshness != "current",
        type="primary",
    ):
        try:
            state.save_reviewed_hypothesis(paragraph)
        except Exception:  # noqa: BLE001 - safe boundary for invalid grounded edits
            st.error(
                "The edited risk could not be saved because it no longer met "
                "the evidence requirements."
            )
        else:
            st.success(
                "Reviewed risk saved. Check whether frozen code supports a scenario."
            )

    testability = state.testability_view()
    if state.human_reviewed_risk is not None and not testability["available"]:
        if st.button(
            "Assess scenario testability",
            disabled=freshness != "current",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Checking whether saved code can support a scenario..."
                ):
                    state.assess_testability()
            except Exception:  # noqa: BLE001 - safe boundary for model or evidence errors
                st.error("Testability could not be assessed from the frozen evidence.")
            else:
                st.success("Testability result is ready for review.")
        testability = state.testability_view()

    if testability["available"]:
        _render_testability_result(st, state, testability, freshness)

    with st.expander("Frozen evidence references"):
        st.json(
            {
                "selected_hypothesis_id": state.selected_hypothesis_id,
                "citation_anchor_ids": selected["citation_anchor_ids"],
            }
        )

    _render_navigation(
        st,
        state,
        next_disabled=(
            state.human_reviewed_risk is None
            or state.testability_assessment is None
            or state.testability_assessment.decision != "testable_from_frozen_evidence"
        ),
    )


def _render_testability_result(
    st: Any,
    state: MilestoneTwoAppState,
    view: dict[str, object],
    freshness: str,
) -> None:
    """Explain whether the current frozen code can support a scenario."""
    decision = view["decision"]
    explanation = str(view["explanation"])
    if decision == "testable_from_frozen_evidence":
        st.success(f"Scenario can be designed from saved code evidence. {explanation}")
        return
    if decision == "not_grounded_in_frozen_evidence":
        st.error(
            "The reviewed hypothesis is no longer grounded in the saved evidence. "
            "Return to risk review rather than creating a scenario."
        )
        return

    st.warning(
        "More frozen code evidence is needed before a scenario can be designed. "
        f"{explanation}"
    )
    for need in view["evidence_needs"]:
        st.write(f"• {need['explanation']}")

    refinement = state.latest_context_refinement
    if refinement is not None and refinement.exhausted:
        st.error(
            "No further relevant frozen code was found. This does not mean the "
            "pull request is safe."
        )
        if st.button(
            "Finish with insufficient frozen code evidence",
            disabled=freshness != "current" or state.terminal_record is not None,
            type="primary",
        ):
            try:
                state.finish_with_insufficient_frozen_evidence()
            except Exception:  # noqa: BLE001 - safe terminal boundary
                st.error("The evidence-insufficient result could not be recorded.")
            else:
                st.warning(
                    "Review finished: insufficient frozen code evidence to design "
                    "an executable scenario."
                )
        return

    if st.button(
        "Find more frozen code evidence",
        disabled=freshness != "current",
        type="primary",
    ):
        try:
            with st.spinner("Searching only the saved code photographs..."):
                refinement = state.refine_frozen_evidence()
        except Exception:  # noqa: BLE001 - no broad or fresh evidence fallback
            st.error("More frozen code evidence could not be checked.")
        else:
            if refinement.exhausted:
                st.warning(
                    "No further relevant frozen code was found. You may finish with "
                    "the evidence-insufficient result."
                )
            else:
                st.success(
                    "Additional frozen code evidence was added. Risk proposals will "
                    "be regenerated for the new context."
                )
                st.rerun()


def _render_scenario(st: Any, state: MilestoneTwoAppState) -> None:
    """Render page five: generate, edit, and approve a single Gherkin scenario."""
    st.header("5. Create and approve the scenario")
    st.write(
        "The scenario is plain text. Read it as a short description of what "
        "will eventually be tested. Milestone 2 stops after you approve it."
    )
    freshness = _render_freshness(st, state)
    testability = state.testability_view()
    if (
        not testability["available"]
        or testability["decision"] != "testable_from_frozen_evidence"
    ):
        st.warning(
            "Return to the reviewed risk and confirm that frozen code evidence can "
            "support a scenario before continuing."
        )
        if testability["decision"] == "needs_more_frozen_evidence":
            _render_testability_result(st, state, testability, freshness)
        if state.current_page > 1 and st.button("Back"):
            state.go_back()
            st.rerun()
        return

    scenario = state.scenario_view()

    if st.button(
        "Generate scenario",
        disabled=(
            scenario["available"]
            or state.terminal_record is not None
            or freshness != "current"
            or testability["decision"] != "testable_from_frozen_evidence"
        ),
        type="primary",
    ):
        try:
            with st.spinner("Generating and validating the Gherkin scenario..."):
                state.generate_gherkin()
        except Exception:  # noqa: BLE001 - no fallback scenario is permitted
            st.error("A valid scenario could not be generated.")
        else:
            st.success("Scenario generated. Review or edit the text below.")
        scenario = state.scenario_view()

    if scenario["available"]:
        edited = st.text_area(
            "Gherkin scenario",
            value=str(scenario["gherkin_text"]),
            height=340,
            disabled=state.terminal_record is not None,
        )
        state.set_gherkin(edited)

        changed = edited != state.gherkin_candidate.gherkin_text
        if changed:
            if st.button(
                "Validate edited scenario",
                disabled=state.terminal_record is not None or freshness != "current",
                type="primary",
            ):
                try:
                    state.validate_edited_gherkin()
                except Exception:  # noqa: BLE001 - safe validation boundary
                    st.error("The edited scenario could not be validated.")
                else:
                    report = state.gherkin_validation_report
                    if report is not None and report.approved:
                        st.success(
                            "Edited scenario is evidence-bound and ready to approve."
                        )
                    elif report is not None:
                        st.warning(
                            "The edited scenario is not ready to approve: "
                            + ", ".join(report.reason_codes)
                        )
        else:
            st.info(
                "The generated scenario already passed local validation and may be "
                "approved unchanged."
            )

        if st.button(
            "Approve scenario",
            disabled=(
                state.terminal_record is not None
                or freshness != "current"
                or (
                    changed
                    and (
                        state.gherkin_validation_report is None
                        or not state.gherkin_validation_report.approved
                    )
                )
            ),
            type="primary",
        ):
            try:
                state.approve_gherkin()
            except Exception:  # noqa: BLE001 - safety gate error remains private
                st.error(
                    "The scenario could not be approved because it no longer "
                    "matches the reviewed risk."
                )
            else:
                st.success("Scenario approved and recorded as research evidence.")

    terminal = state.terminal_view()
    if terminal["available"]:
        st.success("Scenario approved. Milestone 2 is complete for this run.")
        st.write(str(terminal["explanation"]))
        with st.expander("Recorded terminal evidence"):
            st.json(terminal)

        if st.button("Start new analysis"):
            del st.session_state[_SESSION_STATE_KEY]
            st.rerun()

    with st.expander("Model and scenario provenance"):
        st.json(state.provider_view())

    if state.current_page > 1 and not terminal["available"] and st.button("Back"):
        state.go_back()
        st.rerun()


def _render_nonrisk_outcome(
    st: Any,
    state: MilestoneTwoAppState,
    view: dict[str, object],
    freshness: str,
) -> None:
    """Let a human finalize only the two explicit non-risk workflow outcomes."""
    if view["outcome"] == "no_meaningful_security_risk_found":
        st.info(str(view["rationale"]))
        st.write("Limitations:")
        for limitation in view["coverage_limitations"]:
            st.write(f"• {limitation}")
    else:
        st.warning("The bounded evidence was insufficient to assess this change.")
        st.write("Needed evidence:")
        for item in view["needed_evidence"]:
            st.write(f"• {item}")

    if st.button(
        "Finish without a scenario",
        disabled=freshness != "current",
        type="primary",
    ):
        try:
            state.finish_without_risk()
        except Exception:  # noqa: BLE001 - terminal error details stay private
            st.error("This review could not be finalized.")
        else:
            st.success("Review finalized without a proposed scenario.")


def _render_navigation(
    st: Any,
    state: MilestoneTwoAppState,
    *,
    next_disabled: bool,
) -> None:
    """Render only the available Back/Next controls for the current page."""
    back, next_step = st.columns(2)
    with back:
        if state.current_page > 1 and st.button("Back"):
            state.go_back()
            st.rerun()
    with next_step:
        if state.current_page < 5 and st.button(
            "Next",
            disabled=next_disabled,
            type="primary",
        ):
            try:
                state.go_next()
            except PresentationTransitionError:
                st.error("Complete the current step before continuing.")
            else:
                st.rerun()


def _render_freshness(st: Any, state: MilestoneTwoAppState) -> str:
    """Show the currentness of the frozen snapshot without exposing raw failures."""
    if state.prepared is None:
        return "unknown"

    if state.terminal_record is not None:
        freshness = state.terminal_record.freshness
        status = freshness.status if freshness is not None else "unknown"
    else:
        try:
            status = state.workflow.freshness().status
        except Exception:  # noqa: BLE001 - a recheck must fail safely in the UI
            status = "unknown"

    label = freshness_label(status)
    if status == "current":
        st.caption(f"Snapshot status: {label}")
    elif status == "stale":
        st.error(f"Snapshot status: {label}. Approval is disabled.")
    else:
        st.warning(f"Snapshot status: {label}. Approval is disabled.")
    return status


def _require_prepared(
    st: Any,
    state: MilestoneTwoAppState,
) -> object | None:
    """Protect direct page access if a Streamlit session was manually altered."""
    if state.prepared is None:
        st.error("Analyze a pull request before viewing this page.")
        return None
    return state.prepared


def main() -> None:
    """Streamlit entry point."""
    render_app()


if __name__ == "__main__":
    main()
