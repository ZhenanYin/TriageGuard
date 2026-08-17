"""Tests for the guided five-page Milestone 2 application state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from triageguard.analysis.context import ContextBuildError
from triageguard.analysis.diffs import DiffBuildError
from triageguard.analysis.snapshot import SnapshotAcquisitionError
from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.evidence import (
    ModelEvidenceBudgetError,
    ModelEvidencePreflightStop,
)
from triageguard.llm import (
    ModelAttempt,
    ModelFailureProvenance,
    ModelOutputInvalid,
)
from triageguard.ui import app as milestone_two_app
from triageguard.ui.milestone_two import (
    MilestoneTwoAppState,
    PresentationTransitionError,
)
from triageguard.workflow.milestone_two_replay import (
    build_milestone_two_replay_workflow,
)

SUPPORTED_PR_URL = "https://github.com/openmrs/openmrs-core/pull/900000001"


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            SnapshotAcquisitionError(
                "merge_conflict",
                "GitHub reported that the pull request cannot be merged.",
            ),
            (
                "Preparation stopped (merge_conflict): GitHub reported that the "
                "pull request cannot be merged."
            ),
        ),
        (
            DiffBuildError(
                "diff_parse_failed",
                "The frozen Git diff could not be parsed.",
            ),
            (
                "Preparation stopped (diff_parse_failed): The frozen Git diff "
                "could not be parsed."
            ),
        ),
        (
            ContextBuildError(
                "java_parse_failed",
                "A related Java source could not be parsed.",
            ),
            (
                "Preparation stopped (java_parse_failed): A related Java source "
                "could not be parsed."
            ),
        ),
    ),
)
def test_preparation_error_message_exposes_only_safe_typed_details(
    error: Exception,
    expected: str,
) -> None:
    """The live UI explains a typed preparation failure without a traceback."""
    assert milestone_two_app._preparation_error_message(error) == expected
    assert milestone_two_app._preparation_error_message(RuntimeError("internal")) == (
        "The pull request could not be prepared for review."
    )


def test_initial_pull_request_url_is_blank_only_in_live_mode() -> None:
    """Replay keeps its fixture, while live analysis requires an explicit URL."""
    replay_state = SimpleNamespace(
        prepared=None,
        settings=SimpleNamespace(llm_mode="replay"),
    )
    live_state = SimpleNamespace(
        prepared=None,
        settings=SimpleNamespace(llm_mode="live"),
    )

    assert milestone_two_app._initial_pull_request_url(replay_state) == (
        SUPPORTED_PR_URL
    )
    assert milestone_two_app._initial_pull_request_url(live_state) == ""


@pytest.mark.parametrize(
    ("reason_code", "expected"),
    (
        (
            "groq_transient_retries_exhausted",
            "Groq was temporarily unavailable after 1 attempt. Try again shortly.",
        ),
        (
            "groq_non_retryable_error",
            (
                "Groq rejected the risk-proposal request. Check the live model "
                "configuration and API access, then try again."
            ),
        ),
        (
            "risk_assessment_invalid",
            (
                "Groq returned a response, but it did not meet TriageGuard's "
                "required evidence format. Try again."
            ),
        ),
    ),
)
def test_risk_proposal_error_message_uses_only_safe_failure_details(
    reason_code: str,
    expected: str,
) -> None:
    """The UI must explain a failed live call without leaking provider error text."""
    now = datetime(2026, 8, 16, tzinfo=UTC)
    attempt = ModelAttempt(
        number=1,
        started_at=now,
        finished_at=now,
        latency_ms=0,
        outcome="invalid_output",
        error_type="ProviderError",
    )
    failure = ModelFailureProvenance(
        provider="groq",
        model="openai/gpt-oss-120b",
        purpose="risk_hypothesis",
        prompt_sha256="a" * 64,
        request_sha256="b" * 64,
        response_sha256="c" * 64,
        error_sha256="d" * 64,
        input_tokens=10,
        output_tokens=2,
        latency_ms=0,
        attempts=(attempt,),
        final_outcome="invalid_output",
        reason_code=reason_code,
    )

    error = ModelOutputInvalid(
        "secret provider response must not reach the UI",
        [attempt],
        provenance=failure,
    )

    assert milestone_two_app._risk_proposal_error_message(error) == expected
    assert milestone_two_app._risk_proposal_error_message(RuntimeError("internal")) == (
        "Risk proposals could not be created from this evidence."
    )


def test_risk_proposal_error_message_identifies_an_invalid_groq_api_key() -> None:
    """A 401 must lead to an actionable configuration message, never raw text."""
    now = datetime(2026, 8, 16, tzinfo=UTC)
    attempt = ModelAttempt(
        number=1,
        started_at=now,
        finished_at=now,
        latency_ms=0,
        outcome="failed",
        error_type="APIStatusError",
        status_code=401,
    )
    failure = ModelFailureProvenance(
        provider="groq",
        model="openai/gpt-oss-120b",
        purpose="risk_hypothesis",
        prompt_sha256="a" * 64,
        request_sha256="b" * 64,
        response_sha256=None,
        error_sha256="d" * 64,
        input_tokens=None,
        output_tokens=None,
        latency_ms=0,
        attempts=(attempt,),
        final_outcome="failed",
        reason_code="groq_non_retryable_error",
    )

    error = ModelOutputInvalid("secret", [attempt], provenance=failure)

    assert milestone_two_app._risk_proposal_error_message(error) == (
        "Groq rejected the configured API key (HTTP 401). Set a valid "
        "GROQ_API_KEY in the terminal that starts Streamlit, then restart the app."
    )


def test_local_evidence_budget_stop_reports_exact_bytes_without_raw_error() -> None:
    """A pre-provider size stop must not be mislabeled as model validation."""
    error = ModelEvidenceBudgetError(
        "secret anchor identity must not be rendered",
        stage="testability_assessment",
        request_body_bytes=15_488,
        limit_bytes=12_000,
    )
    state = SimpleNamespace(
        model_failure_view=lambda stage: None,
        provider_view=lambda: {"provider": "groq"},
    )

    message = milestone_two_app._model_stage_error_message(
        "testability_assessment",
        error,
        state,
    )

    assert message == (
        "Scenario testability assessment stopped before contacting groq: the "
        "exact request was 15,488 bytes, above the declared 12,000-byte policy. "
        "No conclusion was produced."
    )
    assert "secret anchor identity" not in message


def test_restored_preflight_stop_is_presented_as_zero_provider_attempts() -> None:
    """Restart diagnostics must distinguish local policy from a Groq failure."""
    stop = ModelEvidencePreflightStop(
        stage="risk_hypothesis",
        snapshot_key="a" * 64,
        context_sha256="b" * 64,
        reason_code="model_request_too_large",
        request_body_bytes=12_589,
        max_request_body_bytes=7_000,
        catalog_anchor_count=54,
        observed_at=datetime(2026, 8, 17, tzinfo=UTC),
    )
    workflow = SimpleNamespace(
        model_failure=lambda stage: None,
        model_preflight_stop=lambda stage: stop,
    )
    settings = Settings(
        llm_mode="live",
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        groq_api_key="test-only-key",
    ).public_view()
    state = MilestoneTwoAppState(settings=settings, workflow=workflow)

    assert state.model_failure_view("risk_hypothesis") == {
        "stage": "risk_hypothesis",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "purpose": "risk_hypothesis",
        "reason_code": "model_request_too_large",
        "final_outcome": "stopped_before_provider",
        "attempt_count": 0,
        "latency_ms": 0,
        "last_http_status": None,
        "last_error_type": "ModelEvidenceBudgetError",
        "last_request_body_bytes": 12_589,
        "provider_body_limit_bytes": None,
        "declared_request_limit_bytes": 7_000,
        "catalog_anchor_count": 54,
    }


def _app_state(
    tmp_path,
    *,
    outcome: str = "risks_proposed",
) -> MilestoneTwoAppState:
    settings = Settings(
        llm_mode="replay",
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
        artifacts_dir=tmp_path / "artifacts",
        analysis_cache_dir=tmp_path / "analysis-cache",
    )
    workflow = build_milestone_two_replay_workflow(
        settings,
        outcome=outcome,
    )
    return MilestoneTwoAppState(
        settings=settings.public_view(),
        workflow=workflow,
        workflow_factory=lambda: build_milestone_two_replay_workflow(
            settings,
            outcome=outcome,
        ),
    )


def test_next_is_locked_until_the_pull_request_is_analyzed(tmp_path) -> None:
    """A user cannot skip the frozen-evidence preparation step."""
    state = _app_state(tmp_path)

    assert state.current_page == 1
    with pytest.raises(PresentationTransitionError, match="analyze a pull request"):
        state.go_next()

    state.analyze_pr(SUPPORTED_PR_URL)
    state.go_next()

    assert state.current_page == 2


def test_risk_flow_reaches_human_approved_gherkin_without_skipping_steps(
    tmp_path,
) -> None:
    """One selected grounded risk becomes one approved editable scenario."""
    state = _app_state(tmp_path)

    state.analyze_pr(SUPPORTED_PR_URL)
    state.go_next()
    assessment = state.propose_risks()
    state.go_next()

    risk_view = state.risk_review_view()
    hypothesis = risk_view["hypotheses"][0]
    state.select_risk(hypothesis["hypothesis_id"])
    state.go_next()

    review = state.save_risk_edits({})
    state.go_next()
    testability = state.assess_testability()
    candidate = state.generate_gherkin()
    state.set_gherkin(candidate.gherkin_text)
    record = state.approve_gherkin()

    assert assessment.outcome == "risks_proposed"
    assert review.selected_hypothesis_id == hypothesis["hypothesis_id"]
    assert testability.decision == "testable_from_frozen_evidence"
    assert record.status.value == "approved_gherkin"
    assert state.terminal_view()["status"] == "approved_gherkin"
    assert state.current_page == 5


def test_reviewed_hypothesis_is_edited_as_one_readable_paragraph(tmp_path) -> None:
    """The user edits the displayed risk narrative, not internal component fields."""
    state = _app_state(tmp_path)

    state.analyze_pr(SUPPORTED_PR_URL)
    assessment = state.propose_risks()
    selected = assessment.hypotheses[0]
    state.select_risk(selected.hypothesis_id)
    edited_paragraph = (
        "The merge-preview deletion path may let an authenticated user request "
        "patient deletion without the expected authorization, even though the "
        "secure result should reject the request and preserve the patient record. "
        "This remains an unconfirmed hypothesis based on the saved merge impact."
    )

    review = state.save_reviewed_hypothesis(edited_paragraph)

    assert review.reviewed_risk.explanation == edited_paragraph
    assert state.risk_review_view()["hypotheses"][0]["paragraph"] == (
        selected.explanation
    )


def test_edited_risk_can_reach_an_approved_unchanged_scenario(tmp_path) -> None:
    """A reviewed risk paragraph must bind the generated scenario and approval."""
    state = _app_state(tmp_path)

    state.analyze_pr(SUPPORTED_PR_URL)
    assessment = state.propose_risks()
    selected = assessment.hypotheses[0]
    state.select_risk(selected.hypothesis_id)
    state.save_reviewed_hypothesis(
        selected.explanation.replace(
            "may be able to request deletion",
            "could request deletion",
        )
    )
    state.assess_testability()
    candidate = state.generate_gherkin()

    record = state.approve_gherkin()

    assert record.gherkin_candidate == candidate
    assert record.human_reviewed_risk == state.human_reviewed_risk


def test_risk_review_identifies_the_comparisons_behind_its_evidence(tmp_path) -> None:
    """A readable risk still names the saved comparison that supports it."""
    state = _app_state(tmp_path)

    state.analyze_pr(SUPPORTED_PR_URL)
    state.propose_risks()

    hypothesis = state.risk_review_view()["hypotheses"][0]

    assert hypothesis["comparison_labels"] == ["Merge impact (B → C)"]
    assert state.risk_review_view()["validation_note"] == (
        "Citation validation confirms that references resolve to frozen code "
        "shown to the model. It does not prove that a vulnerability exists."
    )


def test_state_reports_exact_model_visible_evidence_coverage(tmp_path) -> None:
    """Presentation derives coverage from the immutable stage envelope."""
    state = _app_state(tmp_path)
    state.analyze_pr(SUPPORTED_PR_URL)
    state.propose_risks()

    view = state.model_evidence_view("risk_hypothesis")

    assert view["available"] is True
    assert view["stage"] == "risk_hypothesis"
    assert view["visible_anchor_count"] == 2
    assert view["total_anchor_count"] == 2
    assert view["coverage"] == ("2 of 2 frozen anchors visible to this model call.")
    assert view["omitted_anchors"] == []
    assert view["max_request_body_bytes"] == 7_000


def test_refinement_is_available_only_for_structured_frozen_evidence_needs(
    tmp_path,
) -> None:
    """A generic error or ordinary risk proposal must not expose retrieval."""
    proposed = _app_state(tmp_path / "proposed")
    proposed.analyze_pr(SUPPORTED_PR_URL)
    proposed.propose_risks()

    insufficient = _app_state(
        tmp_path / "insufficient",
        outcome="insufficient_context_to_assess",
    )
    insufficient.analyze_pr(SUPPORTED_PR_URL)
    insufficient.propose_risks()

    assert proposed.can_refine_frozen_evidence() is False
    assert insufficient.can_refine_frozen_evidence() is True


def test_model_evidence_summary_lists_each_omitted_anchor_reason() -> None:
    """Removing omission disclosure must fail the reviewer-facing UI contract."""
    rendered: dict[str, list[str]] = {"caption": [], "write": []}
    streamlit = SimpleNamespace(
        caption=lambda value: rendered["caption"].append(value),
        write=lambda value: rendered["write"].append(value),
    )
    state = SimpleNamespace(
        model_evidence_view=lambda stage: {
            "available": True,
            "coverage": "1 of 2 frozen anchors visible to this model call.",
            "omitted_anchors": [
                {
                    "anchor_id": "anchor-hidden",
                    "reason_code": "request_budget",
                    "explanation": (
                        "Omitted because the model request reached its declared "
                        "byte budget."
                    ),
                }
            ],
        }
    )

    milestone_two_app._render_model_evidence_summary(
        streamlit,
        state,
        "risk_hypothesis",
    )

    assert rendered == {
        "caption": ["1 of 2 frozen anchors visible to this model call."],
        "write": [
            "Frozen anchors not visible to this model call:",
            (
                "• anchor-hidden: Omitted because the model request reached its "
                "declared byte budget."
            ),
        ],
    }


def test_changed_gherkin_requires_and_records_local_validation(tmp_path) -> None:
    """Only an edited scenario takes the additional local validation step."""
    state = _app_state(tmp_path)

    state.analyze_pr(SUPPORTED_PR_URL)
    assessment = state.propose_risks()
    hypothesis = assessment.hypotheses[0]
    state.select_risk(hypothesis.hypothesis_id)
    state.save_risk_edits({})
    state.assess_testability()
    candidate = state.generate_gherkin()
    state.set_gherkin(
        candidate.gherkin_text.replace(
            "the user requests deletion",
            "the authenticated user requests deletion",
        )
    )

    report = state.validate_edited_gherkin()
    record = state.approve_gherkin()

    assert report.approved is True
    assert state.gherkin_validation_report is report
    assert record.status.value == "approved_gherkin"


def test_edited_gherkin_needing_code_evidence_can_refine_saved_photos(tmp_path) -> None:
    """An unsupported scenario edit searches only the already saved code photos."""
    state = _app_state(tmp_path)

    state.analyze_pr(SUPPORTED_PR_URL)
    assessment = state.propose_risks()
    state.select_risk(assessment.hypotheses[0].hypothesis_id)
    state.save_risk_edits({})
    state.assess_testability()
    candidate = state.generate_gherkin()
    state.set_gherkin(
        candidate.gherkin_text.replace(
            "using purgePatient and deletePatient",
            "through authorizePatientDeletion",
        )
    )

    report = state.validate_edited_gherkin()
    refinement = state.refine_frozen_evidence()

    assert report.decision == "needs_more_frozen_evidence"
    assert refinement.exhausted is False
    assert state.risk_assessment is None
    assert state.human_reviewed_risk is None
    assert state.current_page == 2


@pytest.mark.parametrize(
    ("outcome", "terminal_status"),
    (
        (
            "no_meaningful_security_risk_found",
            "no_meaningful_security_risk_found",
        ),
        (
            "insufficient_context_to_assess",
            "insufficient_context_to_assess",
        ),
    ),
)
def test_nonrisk_outcomes_finish_without_review_or_gherkin(
    tmp_path,
    outcome: str,
    terminal_status: str,
) -> None:
    """A non-risk outcome ends openly instead of inventing a scenario."""
    state = _app_state(tmp_path, outcome=outcome)

    state.analyze_pr(SUPPORTED_PR_URL)
    state.propose_risks()
    if outcome == "insufficient_context_to_assess":
        while True:
            refinement = state.refine_frozen_evidence()
            if refinement.exhausted:
                break
            state.propose_risks()
    record = state.finish_without_risk()

    assert record.status.value == terminal_status
    assert state.terminal_view()["status"] == terminal_status
    assert state.scenario_view()["available"] is False


def test_state_exposes_only_public_provider_configuration(tmp_path) -> None:
    """A UI state must not retain or display any credential-bearing settings."""
    state = _app_state(tmp_path)

    assert state.settings.groq_api_key is None
    assert state.settings.github_token is None
    assert state.provider_view() == {
        "mode": "Replay",
        "provider": "replay",
        "model": "replay/openai-gpt-oss-120b",
    }
    assert "GROQ_API_KEY" not in repr(state)
    assert "github_token" not in repr(state)


def test_streamlit_replay_app_guides_one_risk_to_an_approved_scenario(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default app shows one page at a time and keeps human gates visible."""
    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv(
        "TRIAGEGUARD_ANALYSIS_CACHE_DIR",
        str(tmp_path / "analysis-cache"),
    )
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "replay")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert [header.value for header in app.header] == ["1. Choose a pull request"]

    buttons = {button.label: button for button in app.button}
    buttons["Analyze pull request"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Next"].click().run()

    assert [header.value for header in app.header] == ["2. Understand the change"]
    assert [subheader.value for subheader in app.subheader] == [
        "Author change — M → H",
        "Merge impact — B → C",
        "Main-branch drift — M → B",
    ]
    assert any(
        "Commit abbreviation" in dataframe.value.columns for dataframe in app.dataframe
    )
    buttons = {button.label: button for button in app.button}
    buttons["Propose possible risks"].click().run()
    assert any(
        caption.value == "2 of 2 frozen anchors visible to this model call."
        for caption in app.caption
    )
    buttons = {button.label: button for button in app.button}
    buttons["Next"].click().run()

    assert [header.value for header in app.header] == ["3. Review possible risks"]
    buttons = {button.label: button for button in app.button}
    buttons["Choose this risk"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Next"].click().run()

    assert [header.value for header in app.header] == ["4. Choose and edit one risk"]
    buttons = {button.label: button for button in app.button}
    buttons["Save reviewed risk"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Assess scenario testability"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Next"].click().run()

    assert [header.value for header in app.header] == [
        "5. Create and approve the scenario"
    ]
    buttons = {button.label: button for button in app.button}
    buttons["Generate scenario"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Approve scenario"].click().run()

    assert any("Scenario approved" in message.value for message in app.success)


def test_start_new_analysis_creates_a_fresh_replay_workflow(tmp_path) -> None:
    """Starting again must create a new run instead of reusing frozen evidence."""
    first = _app_state(tmp_path)

    first.analyze_pr(SUPPORTED_PR_URL)
    second = first.start_new_analysis()

    assert second is not first
    assert second.current_page == 1
    assert second.prepared is None
    assert second.workflow.run_handle.run_id != first.workflow.run_handle.run_id


def test_live_state_uses_public_settings_without_exposing_credentials(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live setup creates real local dependencies but keeps secrets out of UI state."""

    class FakeGroqGateway:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def generate(self, request):  # pragma: no cover - no model call in setup
            raise AssertionError("Live model generation is not part of UI setup.")

    monkeypatch.setattr(
        milestone_two_app,
        "GroqStructuredGateway",
        FakeGroqGateway,
        raising=False,
    )
    api_key = "test-groq-key-that-must-not-appear-in-the-ui-state"
    github_token = "test-github-token-that-must-not-appear-in-the-ui-state"
    settings = Settings(
        llm_mode="live",
        groq_api_key=api_key,
        github_token=github_token,
        artifacts_dir=tmp_path / "artifacts",
        analysis_cache_dir=tmp_path / "analysis-cache",
        environment_kind=EnvironmentKind.REAL_PR_ANALYSIS,
    )

    state = milestone_two_app.create_milestone_two_app_state(settings)

    assert state.provider_view() == {
        "mode": "Live",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
    }
    assert api_key not in repr(state)
    assert github_token not in repr(state)
