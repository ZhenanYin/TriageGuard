import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.execution import (
    ExecutionArtifacts,
    ExecutionTimeoutError,
    MissingObservationError,
)
from triageguard.llm import ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.ui.app import (
    ANALYSIS_EXPLANATION,
    ANALYSIS_SOURCE_LABEL,
    CONTROLLED_FIXTURE_WARNING,
    AppState,
    ContractEditError,
    PresentationTransitionError,
    TerminalRunError,
    create_app_state,
)
from triageguard.workflow import MilestoneOneWorkflow, UnsafeGeneratedCodeError

FIXTURE_ROOT = (
    Path(__file__).parents[2] / "fixtures" / "patient_delete_authorization"
)


def _fixture_payload(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


class _CountingGateway:
    def __init__(self, responses: dict[str, dict]) -> None:
        self._delegate = ReplayGateway(responses)
        self.call_count = 0

    def generate(self, request):
        self.call_count += 1
        return self._delegate.generate(request)


class _NoSocketServer:
    base_url = "http://127.0.0.1:1"

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _no_socket_server_factory(**kwargs: str) -> _NoSocketServer:
    return _NoSocketServer()


def _execution_artifacts(root: Path) -> ExecutionArtifacts:
    run_directory = root / "failed-attempt"
    run_directory.mkdir(parents=True)
    return ExecutionArtifacts(
        run_directory=run_directory,
        pytest_config_path=run_directory / "pytest.ini",
        feature_path=run_directory / "authorization.feature",
        test_path=run_directory / "test_authorization.py",
        observation_path=run_directory / "observation.json",
        pytest_outcome_path=run_directory / "pytest-outcome.json",
        stdout_path=run_directory / "stdout.log",
        stderr_path=run_directory / "stderr.log",
    )


def _failing_runner_factory(error_type, calls: list[str]):
    class FailingRunner:
        def __init__(self, *, artifact_root: Path, **kwargs: object) -> None:
            self.last_artifacts = _execution_artifacts(Path(artifact_root))

        def run(self, target) -> None:
            calls.append(target.revision)
            raise error_type(
                "private-credential-and-raw-runtime-error", self.last_artifacts
            )

    return FailingRunner


def _custom_state(
    tmp_path: Path,
    *,
    gateway,
    runner_factory=None,
    server_factory=None,
) -> AppState:
    settings = Settings(llm_mode="replay", artifacts_dir=tmp_path)
    dependencies = {}
    if runner_factory is not None:
        dependencies["runner_factory"] = runner_factory
    if server_factory is not None:
        dependencies["server_factory"] = server_factory
    workflow = MilestoneOneWorkflow(
        fixture_directory=FIXTURE_ROOT,
        settings=settings,
        gateway=gateway,
        recorder=ArtifactRecorder(tmp_path),
        **dependencies,
    )
    prepared = workflow.prepare()
    return AppState(
        settings=settings,
        fixture_directory=FIXTURE_ROOT,
        workflow=workflow,
        prepared=prepared,
    )


def test_create_app_state_prepares_real_replay_workflow_without_api_key(
    tmp_path: Path,
) -> None:
    settings = Settings(
        llm_mode="replay",
        artifacts_dir=tmp_path,
        environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
    )

    state = create_app_state(settings, FIXTURE_ROOT)

    assert state.mode_label == "Replay (prerecorded model outputs)"
    assert state.provider == "replay"
    assert state.model == "replay/openai-gpt-oss-120b"
    assert state.prepared.contract.contract_id == "patient-delete-authz-001"
    assert state.prepared.environment_kind is EnvironmentKind.CONTROLLED_FIXTURE
    assert state.prepared.run_id
    assert (tmp_path / state.prepared.run_id).is_dir()


def test_impact_and_risk_views_are_fixture_scoped_and_evidence_supported(
    tmp_path: Path,
) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )

    impact = state.impact_view_data()
    risks = state.risks_view_data()

    assert state.warning == CONTROLLED_FIXTURE_WARNING
    assert "not a real OpenMRS revision" in state.warning
    assert "not publication evidence" in state.warning
    assert impact["impact_path"] == [
        "controlled authorization mutation",
        "patient deletion workflow",
        "clerk authorization boundary",
        "base-versus-candidate execution",
    ]
    assert impact["publication_status"] == "not a production security finding"
    assert len(risks) == 1
    assert risks[0]["contract_id"] == "patient-delete-authz-001"
    assert risks[0]["actor"] == "clerk"
    assert risks[0]["missing_privileges"] == ["Delete Patients"]
    assert risks[0]["provenance"]["classification"] == (
        "controlled_development_mutation"
    )
    assert risks[0]["limitations"] == [
        "fixture-only development validation",
        "not a production security finding",
    ]


def test_contract_view_blocks_semantic_edits_before_workflow_approval(
    tmp_path: Path,
) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    changed = state.prepared.gherkin.replace(
        "the clerk attempts to delete the patient",
        "the administrator attempts to delete the patient",
    )

    alignment = state.set_gherkin(changed)

    assert alignment.approved is False
    assert "action_changed" in alignment.reason_codes
    assert state.contract_view_data()["approval_enabled"] is False
    with pytest.raises(ContractEditError, match="not aligned"):
        state.approve_contract()
    assert not (tmp_path / state.prepared.run_id / "run_record.json").exists()


def test_guided_action_requires_risk_selection_before_generation(
    tmp_path: Path,
) -> None:
    """Removing the human selection gate must fail before any model operation."""
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )

    with pytest.raises(RuntimeError, match="risk selection"):
        state.approve_and_generate()

    assert state.approved is False
    assert state.generated is None


def test_guided_action_uses_existing_alignment_approval_and_generation(
    tmp_path: Path,
) -> None:
    """The friendly action must still traverse approval and deterministic gates."""
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    state.accept_risk()

    generated = state.approve_and_generate()

    assert state.risk_accepted is True
    assert state.approved is True
    assert state.generated == generated
    assert generated.validation.approved is True


def test_wizard_navigation_preserves_one_run_and_enforces_security_gates(
    tmp_path: Path,
) -> None:
    """Removing either forward gate or creating a new run must fail this path."""
    gateway = _CountingGateway(
        {
            "test_plan": _fixture_payload("planner_response.json"),
            "pytest_generation": _fixture_payload("generator_response.json"),
        }
    )
    state = _custom_state(tmp_path, gateway=gateway)
    run_id = state.prepared.run_id

    assert state.current_page == 1
    with pytest.raises(PresentationTransitionError, match="first page"):
        state.go_back()
    state.go_next()
    assert state.current_page == 2
    with pytest.raises(PresentationTransitionError, match="risk selection"):
        state.go_next()
    state.accept_risk()
    state.go_next()
    assert state.current_page == 3
    with pytest.raises(PresentationTransitionError, match="validated test"):
        state.go_next()
    state.approve_and_generate()
    assert gateway.call_count == 2
    state.go_next()
    assert state.current_page == 4
    assert state.prepared.run_id == run_id
    with pytest.raises(PresentationTransitionError, match="last page"):
        state.go_next()

    state.go_back()
    state.go_back()
    assert state.current_page == 2
    assert state.prepared.run_id == run_id
    assert state.generated is not None
    assert gateway.call_count == 2


def test_wizard_rejects_invalid_page_assignment_and_resets_to_page_one(
    tmp_path: Path,
) -> None:
    """Invalid or stale page numbers must not escape the guarded range."""
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )

    with pytest.raises(PresentationTransitionError, match="1 through 4"):
        state.current_page = 0
    with pytest.raises(PresentationTransitionError, match="1 through 4"):
        AppState(
            settings=state.settings,
            fixture_directory=state.fixture_directory,
            workflow=state.workflow,
            prepared=state.prepared,
            current_page=5,
        )

    state.go_next()
    new_state = state.start_new_run()
    assert new_state.current_page == 1
    assert new_state.prepared.run_id != state.prepared.run_id


def test_terminal_generation_failure_does_not_unlock_results_page(
    tmp_path: Path,
) -> None:
    """A rejected generated test must remain reviewable on Page 3, not executable."""
    unsafe_generation = _fixture_payload("generator_response.json")
    unsafe_generation["code"] += "\nimport subprocess\n"
    gateway = _CountingGateway(
        {
            "test_plan": _fixture_payload("planner_response.json"),
            "pytest_generation": unsafe_generation,
        }
    )
    state = _custom_state(tmp_path, gateway=gateway)
    state.go_next()
    state.accept_risk()
    state.go_next()

    with pytest.raises(UnsafeGeneratedCodeError):
        state.approve_and_generate()

    assert state.current_page == 3
    assert state.terminal is True
    assert state.generated is None
    with pytest.raises(PresentationTransitionError, match="validated test"):
        state.go_next()
    assert gateway.call_count == 2


def test_aligned_gherkin_edit_is_frozen_with_original_and_edited_values(
    tmp_path: Path,
) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    edited = state.prepared.gherkin.replace(
        "  Scenario:",
        "  # Human note: verify the denial and persistence oracles\n  Scenario:",
    )

    alignment = state.set_gherkin(edited)
    state.approve_contract()
    generated = state.generate_test()

    assert alignment.approved is True
    assert state.approved is True
    assert generated.validation.approved is True
    approved_artifact = next(
        (tmp_path / state.prepared.run_id / "artifacts" / "approved").iterdir()
    )
    payload = json.loads(approved_artifact.read_text(encoding="utf-8"))
    assert payload["prepared"]["contract"] == state.prepared.contract.model_dump(
        mode="json"
    )
    assert payload["prepared"]["gherkin"] == state.prepared.gherkin
    assert payload["approved"]["contract"] == state.prepared.contract.model_dump(
        mode="json"
    )
    assert payload["approved"]["gherkin"] == edited


def test_test_view_exposes_real_replay_generation_and_full_validation(
    tmp_path: Path,
) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    state.approve_contract()
    generated = state.generate_test()

    view = state.test_view_data()

    assert view["available"] is True
    assert view["mode"] == "Replay (prerecorded model outputs)"
    assert view["provider"] == "replay"
    assert view["model"] == "replay/openai-gpt-oss-120b"
    assert view["plan"] == generated.plan.model_dump(mode="json")
    assert view["generated_source"] == generated.generated.code
    assert view["validation"] == generated.validation.model_dump(mode="json")
    assert view["validation"]["approved"] is True
    assert view["validation"]["reason_codes"] == []
    assert view["generation_error"] is None
    assert view["provider_provenance"] == {
        "configured_live_provider": "groq",
        "configured_live_model": "openai/gpt-oss-120b",
        "active_gateway_provider": "replay",
        "active_gateway_model": "replay/openai-gpt-oss-120b",
        "recorded_response_provider": "replay",
        "recorded_response_model": "replay/openai-gpt-oss-120b",
    }
    assert generated.generated.model_response.provider == "replay"


def test_evidence_view_is_derived_from_a_completed_real_replay_run(
    tmp_path: Path,
) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    state.approve_contract()
    state.generate_test()
    result = state.execute(repeat_count=1)

    view = state.evidence_view_data()

    assert view["available"] is True
    assert view["status"] == "candidate_regression_observed"
    assert view["status"] == result.status.value
    assert view["reason_code"] == "candidate_regression_observed"
    assert view["reason_code"] == result.reason_code
    assert view["explanation"] == (
        "The base denied unauthorized deletion and preserved the patient, while "
        "the candidate allowed deletion and removed the patient."
    )
    assert view["base"]["revision"] == "base-revision"
    assert view["base"]["control_succeeded"] is True
    assert view["base"]["control_request_status"] == 204
    assert view["base"]["control_resource_exists_before"] is True
    assert view["base"]["control_resource_exists_after"] is False
    assert view["candidate"]["revision"] == "candidate-revision"
    assert view["candidate"]["control_succeeded"] is True
    assert view["candidate"]["control_request_status"] == 204
    assert view["candidate"]["control_resource_exists_before"] is True
    assert view["candidate"]["control_resource_exists_after"] is False
    assert view["repetitions"] == 1
    assert view["stable"] is True
    assert view["base_differing_run_indexes"] == []
    assert view["candidate_differing_run_indexes"] == []
    assert view["severity_assessment"] == (
        result.severity_assessment.model_dump(mode="json")
    )
    assert view["severity_assessment"]["base"]["status"] == "not_scored"
    assert view["severity_assessment"]["base"]["score"] is None
    assert view["severity_assessment"]["candidate"]["status"] == "provisional"
    assert view["severity_assessment"]["candidate"]["score"] == 7.1
    assert view["severity_assessment"]["candidate"]["severity"] == "High"
    assert str(tmp_path.resolve()) in view["artifact_paths"]["run_directory"]
    assert Path(view["artifact_paths"]["run_record"]).is_file()
    assert Path(view["artifact_paths"]["event_log"]).is_file()


def test_unexecuted_view_models_do_not_invent_scoring_content(tmp_path: Path) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    payload = {
        "impact": state.impact_view_data(),
        "risks": state.risks_view_data(),
        "contract": state.contract_view_data(),
        "test": state.test_view_data(),
        "evidence": state.evidence_view_data(),
    }

    assert payload["evidence"]["available"] is False
    assert payload["evidence"].get("severity_assessment") is None


def test_streamlit_app_renders_only_the_current_wizard_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stacking later page bodies or bypassing Next must fail this flow."""
    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "replay")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert not app.radio
    assert [header.value for header in app.header] == ["1. Understand the change"]
    assert "Step 1 of 4" in [item.value for item in app.caption]
    assert ANALYSIS_EXPLANATION in [item.value for item in app.markdown]
    assert ANALYSIS_SOURCE_LABEL in [item.value for item in app.caption]
    assert all(
        "This prototype uses a prepared patient-deletion example" not in item.value
        for item in app.markdown
    )
    buttons = {button.label: button for button in app.button}
    assert set(buttons) == {"Next: Review risk"}
    assert [expander.label for expander in app.expander] == ["Technical details"]
    original_run_id = app.session_state.filtered_state[
        "triageguard_milestone_one_state"
    ].prepared.run_id

    buttons["Next: Review risk"].click().run()
    assert not app.exception
    assert [header.value for header in app.header] == [
        "2. Review the security risk"
    ]
    buttons = {button.label: button for button in app.button}
    assert set(buttons) == {"Back", "Use this risk", "Next: Review test"}
    assert buttons["Next: Review test"].disabled is True
    buttons["Use this risk"].click().run()
    buttons = {button.label: button for button in app.button}
    assert buttons["Next: Review test"].disabled is False
    buttons["Next: Review test"].click().run()
    assert [header.value for header in app.header] == [
        "3. Review and generate the test"
    ]
    assert (
        app.session_state.filtered_state["triageguard_milestone_one_state"]
        .prepared.run_id
        == original_run_id
    )


def test_guided_replay_run_explains_the_regression_without_requiring_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the plain-language result must fail a real guided replay run."""
    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "replay")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45).run()

    buttons = {button.label: button for button in app.button}
    buttons["Next: Review risk"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Use this risk"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Next: Review test"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Approve scenario and generate test"].click().run()
    buttons = {button.label: button for button in app.button}
    buttons["Next: Run comparison"].click().run()
    app.number_input[0].set_value(1).run()
    buttons = {button.label: button for button in app.button}
    buttons["Run base vs candidate comparison"].click().run()

    assert not app.exception
    assert any(
        item.value == "Potential security regression detected"
        for item in app.subheader
    )
    assert any(
        "without the required permission" in item.value
        for item in app.error
    )
    table = app.dataframe[0].value
    assert list(table["Version"]) == ["Secure base", "Candidate"]
    assert list(table["Meaning"]) == [
        "Protected",
        "Unauthorized deletion observed",
    ]
    rendered_text = "\n".join(
        item.value
        for collection in (
            app.markdown,
            app.caption,
            app.subheader,
            app.info,
            app.warning,
            app.error,
            app.success,
            app.code,
        )
        for item in collection
        if isinstance(item.value, str)
    )
    assert "Not scored" in rendered_text
    assert "CVSS 4.0 not calculated" in rendered_text
    assert "Tested vulnerability not observed in this version." in rendered_text
    assert "7.1 High" in rendered_text
    assert "Provisional CVSS 4.0" in rendered_text
    assert (
        "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:N/VI:H/VA:N/"
        "SC:N/SI:N/SA:N"
    ) in rendered_text
    assert "0.0" not in rendered_text
    assert "score delta" not in rendered_text.lower()
    assert [expander.label for expander in app.expander][-3:] == [
        "CVSS assessment details",
        "Research evidence",
        "Local artifacts",
    ]
    cvss_details = next(
        expander
        for expander in app.expander
        if expander.label == "CVSS assessment details"
    )
    details_text = "\n".join(
        item.value
        for collection in (
            cvss_details.markdown,
            cvss_details.caption,
            cvss_details.code,
        )
        for item in collection
        if isinstance(item.value, str)
    )
    rendered_state = app.session_state.filtered_state[
        "triageguard_milestone_one_state"
    ]
    for metric in rendered_state.result.severity_assessment.candidate.metrics:
        assert f"{metric.metric}: {metric.value}" in details_text
        assert metric.rationale in details_text
        assert metric.source_category in details_text
    assert "Deployment assumption" in details_text
    assert "Expert judgment" in details_text
    assert "not measured by pytest" in details_text
    assert (
        app.session_state.filtered_state["triageguard_milestone_one_state"]
        .result.status.value
        == "candidate_regression_observed"
    )
    completed_state = rendered_state
    completed_run_id = completed_state.prepared.run_id
    completed_result = completed_state.result
    event_log = Path(
        completed_state.evidence_view_data()["artifact_paths"]["event_log"]
    )
    events_before_navigation = event_log.read_text(encoding="utf-8")

    buttons = {button.label: button for button in app.button}
    buttons["Back"].click().run()
    revisited_state = app.session_state.filtered_state[
        "triageguard_milestone_one_state"
    ]
    assert [header.value for header in app.header] == [
        "3. Review and generate the test"
    ]
    assert revisited_state.prepared.run_id == completed_run_id
    assert revisited_state.result == completed_result

    buttons = {button.label: button for button in app.button}
    buttons["Next: Run comparison"].click().run()
    returned_state = app.session_state.filtered_state[
        "triageguard_milestone_one_state"
    ]
    assert [header.value for header in app.header] == ["4. Run the comparison"]
    assert returned_state.result == completed_result
    assert event_log.read_text(encoding="utf-8") == events_before_navigation

    buttons = {button.label: button for button in app.button}
    buttons["Start New Run"].click().run()
    replacement = app.session_state.filtered_state[
        "triageguard_milestone_one_state"
    ]
    assert [header.value for header in app.header] == ["1. Understand the change"]
    assert replacement.current_page == 1
    assert replacement.prepared.run_id != completed_run_id


def test_live_configuration_without_credentials_fails_before_state_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "live")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert len(app.error) == 1
    assert "Live mode requires GROQ_API_KEY" in app.error[0].value
    assert "triageguard_milestone_one_state" not in app.session_state.filtered_state
    assert not list(tmp_path.iterdir())


def test_generation_failure_view_never_contains_raw_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path), FIXTURE_ROOT
    )
    state.approve_contract()

    def fail() -> None:
        raise RuntimeError("secret-token-and-provider-payload")

    monkeypatch.setattr(state.workflow, "generate", fail)
    with pytest.raises(RuntimeError, match="secret-token"):
        state.generate_test()

    rendered = json.dumps(state.test_view_data())
    assert "secret-token-and-provider-payload" not in rendered
    assert state.test_view_data()["generation_error"] == {
        "stage": "generation",
        "error_type": "RuntimeError",
        "message": "Generation did not complete. No fallback was used.",
        "fallback_used": False,
    }


def test_import_is_side_effect_free_and_entrypoint_is_guarded(tmp_path: Path) -> None:
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    guarded_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(child, ast.Constant) and child.value == "__main__"
            for child in ast.walk(node.test)
        )
    ]
    assert len(guarded_calls) == 1

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    environment["TRIAGEGUARD_ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    completed = subprocess.run(
        [sys.executable, "-c", "import triageguard.ui.app"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "artifacts").exists()


def test_unsafe_generation_becomes_one_stable_terminal_ui_result(
    tmp_path: Path,
) -> None:
    unsafe_generation = _fixture_payload("generator_response.json")
    unsafe_generation["code"] += (
        "\nimport subprocess\n# private-credential-and-provider-payload\n"
    )
    gateway = _CountingGateway(
        {
            "test_plan": _fixture_payload("planner_response.json"),
            "pytest_generation": unsafe_generation,
        }
    )
    state = _custom_state(tmp_path, gateway=gateway)
    state.approve_contract()

    with pytest.raises(UnsafeGeneratedCodeError):
        state.generate_test()

    test_view = state.test_view_data()
    evidence_view = state.evidence_view_data()
    assert gateway.call_count == 2
    assert test_view["validation"] == {
        "approved": False,
        "reason_codes": ["forbidden_import"],
        "implemented_steps": test_view["validation"]["implemented_steps"],
        "used_primitives": test_view["validation"]["used_primitives"],
        "code_sha256": test_view["validation"]["code_sha256"],
    }
    assert test_view["terminal_result"] == {
        "status": "validation_failed",
        "reason_code": "unsafe_generated_code",
        "explanation": "Generated code was rejected by deterministic validation.",
        "artifact_paths": test_view["terminal_result"]["artifact_paths"],
    }
    assert test_view["generation_enabled"] is False
    assert evidence_view["available"] is True
    assert evidence_view["differential_available"] is False
    assert evidence_view["status"] == "validation_failed"
    assert "private-credential" not in json.dumps(test_view)

    with pytest.raises(TerminalRunError, match="Start New Run"):
        state.generate_test()
    assert gateway.call_count == 2


@pytest.mark.parametrize(
    ("error_type", "reason_code", "explanation"),
    [
        (
            ExecutionTimeoutError,
            "generated_test_timeout",
            "The approved generated test exceeded its bounded timeout.",
        ),
        (
            MissingObservationError,
            "missing_runtime_observation",
            "Execution ended without all five required raw observation facts.",
        ),
    ],
)
def test_execution_failure_is_terminal_visible_stable_and_resettable(
    tmp_path: Path,
    error_type,
    reason_code: str,
    explanation: str,
) -> None:
    gateway = _CountingGateway(
        {
            "test_plan": _fixture_payload("planner_response.json"),
            "pytest_generation": _fixture_payload("generator_response.json"),
        }
    )
    runner_calls: list[str] = []
    state = _custom_state(
        tmp_path,
        gateway=gateway,
        runner_factory=_failing_runner_factory(error_type, runner_calls),
        server_factory=_no_socket_server_factory,
    )
    state.approve_contract()
    state.generate_test()

    with pytest.raises(error_type):
        state.execute(repeat_count=1)

    view = state.evidence_view_data()
    assert runner_calls == ["base-revision"]
    assert view["available"] is True
    assert view["differential_available"] is False
    assert view["status"] == "execution_inconclusive"
    assert view["reason_code"] == reason_code
    assert view["explanation"] == explanation
    assert view["execution_enabled"] is False
    assert Path(view["artifact_paths"]["run_record"]).is_file()
    assert "private-credential" not in json.dumps(view)

    with pytest.raises(TerminalRunError, match="Start New Run"):
        state.execute(repeat_count=1)
    assert runner_calls == ["base-revision"]

    old_result = state.result
    new_state = state.start_new_run()
    assert new_state.prepared.run_id != state.prepared.run_id
    assert new_state.risk_accepted is False
    assert new_state.approved is False
    assert new_state.generated is None
    assert new_state.result is None
    assert state.result == old_result


def test_invalid_provider_app_configuration_creates_no_gateway_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGEGUARD_LLM_PROVIDER", "openai")
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert len(app.error) == 1
    assert app.error[0].value == (
        "Configuration is invalid. Only the Groq provider is supported; no "
        "gateway or run was created."
    )
    assert "triageguard_milestone_one_state" not in app.session_state.filtered_state
    assert not list(tmp_path.iterdir())


def test_invalid_repeat_configuration_fails_before_evidence_widget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGEGUARD_REPEAT_COUNT", "21")
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert app.error[0].value == (
        "Configuration is invalid. Repeat count must be an integer from 1 to 20; "
        "no run was created."
    )
    assert not app.number_input
    assert "triageguard_milestone_one_state" not in app.session_state.filtered_state
    assert not list(tmp_path.iterdir())


def test_live_and_replay_states_distinguish_configuration_from_active_gateway(
    tmp_path: Path,
) -> None:
    replay = create_app_state(
        Settings(llm_mode="replay", artifacts_dir=tmp_path / "replay"),
        FIXTURE_ROOT,
    )
    live = create_app_state(
        Settings(
            llm_mode="live",
            llm_provider="groq",
            groq_api_key="local-test-key-never-used",
            artifacts_dir=tmp_path / "live",
        ),
        FIXTURE_ROOT,
    )

    assert replay.test_view_data()["provider_provenance"] == {
        "configured_live_provider": "groq",
        "configured_live_model": "openai/gpt-oss-120b",
        "active_gateway_provider": "replay",
        "active_gateway_model": "replay/openai-gpt-oss-120b",
        "recorded_response_provider": None,
        "recorded_response_model": None,
    }
    assert live.test_view_data()["provider_provenance"] == {
        "configured_live_provider": "groq",
        "configured_live_model": "openai/gpt-oss-120b",
        "active_gateway_provider": "groq",
        "active_gateway_model": "openai/gpt-oss-120b",
        "recorded_response_provider": None,
        "recorded_response_model": None,
    }
    assert live.settings.groq_api_key is None
    assert live.workflow._settings.groq_api_key is None


def test_streamlit_terminal_reset_replaces_session_workflow_with_fresh_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe_generation = _fixture_payload("generator_response.json")
    unsafe_generation["code"] += "\nimport subprocess\n"
    terminal_state = _custom_state(
        tmp_path / "terminal",
        gateway=_CountingGateway(
            {
                "test_plan": _fixture_payload("planner_response.json"),
                "pytest_generation": unsafe_generation,
            }
        ),
    )
    terminal_state.approve_contract()
    with pytest.raises(UnsafeGeneratedCodeError):
        terminal_state.generate_test()
    terminal_state.risk_accepted = True
    terminal_state.current_page = 3
    old_run_id = terminal_state.prepared.run_id

    monkeypatch.setenv("TRIAGEGUARD_ARTIFACTS_DIR", str(tmp_path / "app"))
    app_path = Path(__file__).parents[2] / "src" / "triageguard" / "ui" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    app.session_state["triageguard_milestone_one_state"] = terminal_state
    app.run()

    buttons = {button.label: button for button in app.button}
    assert any(
        item.value == "The generated test was rejected and will not be executed."
        for item in app.error
    )
    assert any(
        item.value == "Test generation stopped. No fallback test was used."
        for item in app.error
    )
    assert buttons["Approve scenario and generate test"].disabled is True
    assert "Run base vs candidate comparison" not in buttons
    assert buttons["Start New Run"].disabled is False
    buttons["Start New Run"].click().run()

    replacement = app.session_state.filtered_state[
        "triageguard_milestone_one_state"
    ]
    assert replacement.prepared.run_id != old_run_id
    assert replacement.result is None
    assert replacement.generated is None
    assert terminal_state.result is not None
