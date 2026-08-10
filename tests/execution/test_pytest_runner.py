from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from triageguard.contracts.gherkin import render_gherkin
from triageguard.domain.models import RiskContract, TestPlan
from triageguard.execution import (
    ControlledAuthorizationServer,
    ExecutionTarget,
    ExecutionTimeoutError,
    InvalidObservationError,
    PytestRunner,
    UnexpectedPytestOutcomeError,
)
from triageguard.execution import pytest_runner as pytest_runner_module
from triageguard.runtime import ObservationWriter, RuntimeObservationEnvelope

ADMINISTRATOR = "fixture-administrator"
CLERK = "clerk"
PASSWORD = "fixture-password-not-for-logs"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:8080",
        "http://localhost:8080",
        "http://[::1]:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://user:password@127.0.0.1:8080",
        "http://127.0.0.1:8080/",
        "http://127.0.0.1:8080/openmrs",
        "http://127.0.0.1:8080?mode=secure",
        "http://127.0.0.1:8080#fragment",
        "not-a-url",
    ],
)
def test_execution_target_rejects_non_controlled_loopback_urls(
    base_url: str,
) -> None:
    """Remote, ambiguous, or path-bearing targets must never reach subprocess."""
    with pytest.raises(ValueError, match="HTTP loopback target"):
        ExecutionTarget(
            base_url=base_url,
            username=ADMINISTRATOR,
            password=PASSWORD,
            revision="candidate-revision",
        )


def _approved_inputs() -> tuple[str, RiskContract, TestPlan, str, str]:
    fixture_root = (
        Path(__file__).parents[2]
        / "fixtures"
        / "patient_delete_authorization"
    )
    contract_payload = json.loads(
        (fixture_root / "approved_contract.json").read_text(encoding="utf-8")
    )
    generated_payload = json.loads(
        (fixture_root / "generator_response.json").read_text(encoding="utf-8")
    )
    planner_payload = json.loads(
        (fixture_root / "planner_response.json").read_text(encoding="utf-8")
    )
    canonical_contract = json.dumps(
        contract_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    contract = RiskContract.model_validate(contract_payload)
    return (
        generated_payload["code"],
        contract,
        TestPlan.model_validate(planner_payload),
        render_gherkin(contract),
        hashlib.sha256(canonical_contract.encode("utf-8")).hexdigest(),
    )


def _runner(
    artifact_root: Path, *, generated_code: str | None = None
) -> PytestRunner:
    approved_code, contract, plan, gherkin, _ = _approved_inputs()
    return PytestRunner(
        generated_code=(generated_code if generated_code is not None else approved_code),
        contract=contract,
        plan=plan,
        gherkin=gherkin,
        artifact_root=artifact_root,
    )


def _target(
    server: ControlledAuthorizationServer,
    *,
    revision: str,
    password: str = PASSWORD,
) -> ExecutionTarget:
    return ExecutionTarget(
        base_url=server.base_url,
        username=ADMINISTRATOR,
        password=password,
        revision=revision,
    )


def test_runner_executes_approved_replay_against_secure_base(
    tmp_path: Path,
) -> None:
    """The approved secure scenario must execute, not be simulated in-process."""
    runner = _runner(tmp_path / "artifacts")
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        observation = runner.run(_target(server, revision="base-revision"))

    assert observation.model_dump() == {
        "revision": "base-revision",
        "setup_succeeded": True,
        "action_attempted": True,
        "control_succeeded": True,
        "control_request_status": 204,
        "control_resource_exists_before": True,
        "control_resource_exists_after": False,
        "request_status": 403,
        "resource_exists_after": True,
        "pytest_exit_code": 0,
        "reason_code": "pytest_completed_with_observation",
    }
    artifacts = runner.last_artifacts
    assert artifacts is not None
    assert artifacts.run_directory.parent == tmp_path / "artifacts"
    assert artifacts.test_path.read_text(encoding="utf-8") == _approved_inputs()[0]
    assert artifacts.feature_path.read_text(encoding="utf-8") == _approved_inputs()[3]
    assert "1 passed" in artifacts.stdout_path.read_text(encoding="utf-8")
    assert artifacts.stderr_path.read_text(encoding="utf-8") == ""
    assert PASSWORD not in artifacts.stdout_path.read_text(encoding="utf-8")
    envelope = RuntimeObservationEnvelope.model_validate_json(
        artifacts.observation_path.read_text(encoding="utf-8")
    )
    assert envelope.contract_sha256 == _approved_inputs()[4]
    assert envelope.control_succeeded is True
    assert envelope.control_request_status == 204
    assert envelope.control_resource_exists_before is True
    assert envelope.control_resource_exists_after is False


def test_runner_preserves_complete_vulnerable_facts_when_pytest_fails(
    tmp_path: Path,
) -> None:
    """A candidate assertion failure must not erase its preceding raw facts."""
    runner = _runner(tmp_path / "artifacts")
    with ControlledAuthorizationServer(
        behavior="vulnerable",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        observation = runner.run(_target(server, revision="candidate-revision"))

    assert observation.model_dump() == {
        "revision": "candidate-revision",
        "setup_succeeded": True,
        "action_attempted": True,
        "control_succeeded": True,
        "control_request_status": 204,
        "control_resource_exists_before": True,
        "control_resource_exists_after": False,
        "request_status": 204,
        "resource_exists_after": False,
        "pytest_exit_code": 1,
        "reason_code": "pytest_failed_with_complete_observation",
    }
    artifacts = runner.last_artifacts
    assert artifacts is not None
    assert "1 failed" in artifacts.stdout_path.read_text(encoding="utf-8")
    assert PASSWORD not in artifacts.stdout_path.read_text(encoding="utf-8")
    envelope = RuntimeObservationEnvelope.model_validate_json(
        artifacts.observation_path.read_text(encoding="utf-8")
    )
    assert envelope.request_status == 204
    assert envelope.resource_exists_after is False
    assert envelope.pytest_exit_code == 1
    assert envelope.control_succeeded is True
    assert envelope.control_request_status == 204
    assert envelope.control_resource_exists_before is True
    assert envelope.control_resource_exists_after is False


def test_runner_rejects_a_changed_primary_assertion_before_execution(
    tmp_path: Path,
) -> None:
    """A frame in the right function cannot approve a changed assertion."""
    approved_code = _approved_inputs()[0]
    changed_code = approved_code.replace(
        "assert delete_status == 403",
        "assert False",
        1,
    )

    with pytest.raises(ValueError, match="full generated-code approval"):
        _runner(tmp_path / "artifacts", generated_code=changed_code)


@pytest.mark.parametrize(
    "mutation",
    [
        "import subprocess",
        (
            'Path(os.environ["TRIAGEGUARD_OBSERVATION_PATH"])'
            '.with_name("pytest-outcome.json").write_text("{}")'
        ),
        "def pytest_sessionfinish(session, exitstatus):\n    pass",
        'pytest_plugins = ("forged_outcome_plugin",)',
        (
            "@pytest.fixture(autouse=True)\n"
            "def forged_fixture():\n"
            "    return None"
        ),
        "raise SystemExit(0)",
    ],
)
def test_runner_replays_full_code_validation_before_subprocess(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Canonical assertions cannot bypass the complete Task 6 approval gate."""
    changed_code = f"{_approved_inputs()[0]}\n{mutation}\n"

    with pytest.raises(ValueError, match="full generated-code approval"):
        _runner(tmp_path / "artifacts", generated_code=changed_code)

    assert not (tmp_path / "artifacts").exists()


def test_runner_replays_full_validation_again_at_run_entry(
    tmp_path: Path,
) -> None:
    """A post-construction source change is stopped before artifact creation."""
    runner = _runner(tmp_path / "artifacts")
    runner._generated_code = f"{runner._generated_code}\nimport subprocess\n"
    target = ExecutionTarget(
        base_url="http://127.0.0.1:1",
        username=ADMINISTRATOR,
        password=PASSWORD,
        revision="never-executed",
    )

    with pytest.raises(ValueError, match="full generated-code approval"):
        runner.run(target)

    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "assert control_delete_status == 204",
            "assert control_delete_status == 200",
        ),
        (
            "observation_writer.record_patient_exists(patient_exists)",
            (
                "observation_writer.record_patient_exists(patient_exists)\n"
                "    assert False, 'unrelated failure after complete facts'"
            ),
        ),
        (
            "observation_writer.record_patient_exists(patient_exists)",
            (
                "observation_writer.record_patient_exists(patient_exists)\n"
                "    raise RuntimeError('transport failure after complete facts')"
            ),
        ),
    ],
)
def test_runner_rejects_non_primary_failures_after_complete_facts(
    tmp_path: Path,
    needle: str,
    replacement: str,
) -> None:
    """Unapproved failure paths cannot reach the subprocess boundary."""
    approved_code = _approved_inputs()[0]
    changed_code = approved_code.replace(needle, replacement, 1)
    assert changed_code != approved_code

    with pytest.raises(ValueError, match="full generated-code approval"):
        _runner(tmp_path / "artifacts", generated_code=changed_code)

    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("failed_phase", "exception_type", "expected_reason"),
    [
        ("call", "RuntimeError", "pytest_unapproved_call_failure"),
        ("teardown", "AssertionError", "pytest_teardown_or_control_failed"),
    ],
)
def test_runner_outcome_gate_rejects_non_primary_failures_defensively(
    tmp_path: Path,
    failed_phase: str,
    exception_type: str,
    expected_reason: str,
) -> None:
    """Structured outcome checks remain a second independent defense."""
    runner = _runner(tmp_path / "artifacts")
    artifacts = runner._create_artifacts()
    artifacts.test_path.write_text(_approved_inputs()[0], encoding="utf-8")
    writer = ObservationWriter(artifacts.observation_path)
    writer.record_http_status(204)
    writer.record_patient_exists(False)
    reports = []
    for phase in ("setup", "call", "teardown"):
        failed = phase == failed_phase
        reports.append(
            {
                "nodeid": (
                    "test_authorization.py::test_patient_delete_authorization"
                ),
                "when": phase,
                "outcome": "failed" if failed else "passed",
                "exception_type": exception_type if failed else None,
                "frames": (
                    [
                        {
                            "path": str(artifacts.test_path),
                            "lineno": 1,
                            "function": "unapproved_failure",
                        }
                    ]
                    if failed
                    else []
                ),
            }
        )
    artifacts.pytest_outcome_path.write_text(
        json.dumps({"exitstatus": 1, "reports": reports}),
        encoding="utf-8",
    )

    with pytest.raises(UnexpectedPytestOutcomeError) as error:
        runner._validate_pytest_outcome(1, artifacts)

    assert error.value.reason_code == expected_reason
    assert not artifacts.observation_path.exists()


@pytest.mark.parametrize(
    "extra_record",
    [
        "observation_writer.record_http_status(delete_status)",
        "observation_writer.record_http_status(204)",
        "observation_writer.record_patient_exists(patient_exists)",
        "observation_writer.record_patient_exists(not patient_exists)",
    ],
)
def test_runner_rejects_repeated_or_conflicting_raw_events(
    tmp_path: Path,
    extra_record: str,
) -> None:
    """Latest-value collapse must not hide duplicate or conflicting evidence."""
    approved_code = _approved_inputs()[0]
    existing_record = "observation_writer.record_patient_exists(patient_exists)"
    changed_code = approved_code.replace(
        existing_record,
        f"{existing_record}\n    {extra_record}",
        1,
    )
    with pytest.raises(ValueError, match="full generated-code approval"):
        _runner(tmp_path / "artifacts", generated_code=changed_code)

    assert not (tmp_path / "artifacts").exists()


def test_runner_still_rejects_duplicate_raw_facts_at_finalization(
    tmp_path: Path,
) -> None:
    """Runtime cardinality remains a defense if an approved test misbehaves."""
    runner = _runner(tmp_path / "artifacts")
    artifacts = runner._create_artifacts()
    writer = ObservationWriter(artifacts.observation_path)
    writer.record_http_status(403)
    writer.record_patient_exists(True)
    writer.record_patient_exists(False)

    with pytest.raises(InvalidObservationError, match="exactly one raw event"):
        runner._read_exact_observation_facts(writer, artifacts)

    assert not artifacts.observation_path.exists()


def test_runner_uses_a_new_persistent_isolated_directory_for_each_run(
    tmp_path: Path,
) -> None:
    """Reusing a run directory could mix evidence across revisions."""
    runner = _runner(tmp_path / "artifacts")
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        runner.run(_target(server, revision="base-first"))
        first_artifacts = runner.last_artifacts
        runner.run(_target(server, revision="base-second"))
        second_artifacts = runner.last_artifacts

    assert first_artifacts is not None
    assert second_artifacts is not None
    assert first_artifacts.run_directory != second_artifacts.run_directory
    assert first_artifacts.observation_path.exists()
    assert second_artifacts.observation_path.exists()


def test_runner_ignores_hostile_ancestor_pytest_config_and_conftest(
    tmp_path: Path,
) -> None:
    """Ancestor pytest discovery must not alter collection or fixture behavior."""
    hostile_root = tmp_path / "hostile-ancestor"
    hostile_root.mkdir()
    (hostile_root / "pytest.ini").write_text(
        "[pytest]\naddopts = --collect-only\npython_files = hostile_*.py\n",
        encoding="utf-8",
    )
    (hostile_root / "conftest.py").write_text(
        "import pytest\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def hostile_ancestor_fixture():\n"
        "    raise RuntimeError('hostile ancestor conftest loaded')\n",
        encoding="utf-8",
    )
    runner = _runner(hostile_root / "artifacts")

    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        observation = runner.run(_target(server, revision="isolated-revision"))

    assert observation.pytest_exit_code == 0
    assert observation.request_status == 403
    assert observation.resource_exists_after is True
    assert observation.control_request_status == 204
    assert observation.control_resource_exists_before is True
    assert observation.control_resource_exists_after is False


def test_runner_rejects_exit_without_complete_raw_observation(
    tmp_path: Path,
) -> None:
    """Setup/target failure must not become secure or vulnerable evidence."""
    runner = _runner(tmp_path / "artifacts")
    wrong_password = f"{PASSWORD}-wrong-private-value"
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server, pytest.raises(
        UnexpectedPytestOutcomeError,
        match="pytest_teardown_or_control_failed",
    ):
        runner.run(
            _target(
                server,
                revision="unavailable-revision",
                password=wrong_password,
            )
        )

    artifacts = runner.last_artifacts
    assert artifacts is not None
    assert not artifacts.observation_path.exists()
    assert wrong_password not in artifacts.stdout_path.read_text(encoding="utf-8")
    assert wrong_password not in artifacts.stderr_path.read_text(encoding="utf-8")


def test_runner_classifies_real_pytest_setup_failure_without_final_evidence(
    tmp_path: Path,
) -> None:
    """A real two-phase setup failure must not look structurally ambiguous."""
    runner = _runner(tmp_path / "artifacts")
    artifacts = runner._create_artifacts()
    artifacts.pytest_config_path.write_text("[pytest]\n", encoding="utf-8")
    artifacts.test_path.write_text(
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def broken_fixture():\n"
        "    raise RuntimeError('controlled setup failure')\n\n"
        "def test_patient_delete_authorization(broken_fixture):\n"
        "    pass\n",
        encoding="utf-8",
    )
    environment = {
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "triageguard.execution.pytest_outcome_plugin",
            "-c",
            str(artifacts.pytest_config_path),
            "--rootdir",
            str(artifacts.run_directory),
            "--confcutdir",
            str(artifacts.run_directory),
            "--noconftest",
            artifacts.test_path.name,
            "-q",
            "--triageguard-outcome-path",
            str(artifacts.pytest_outcome_path),
        ],
        cwd=artifacts.run_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    with pytest.raises(UnexpectedPytestOutcomeError) as error:
        runner._validate_pytest_outcome(completed.returncode, artifacts)

    assert error.value.reason_code == "pytest_setup_failed"
    assert not artifacts.observation_path.exists()


def test_runner_rejects_collection_errors_without_final_evidence(
    tmp_path: Path,
) -> None:
    """Collection/internal failures must be distinct from experiment evidence."""
    approved_code = _approved_inputs()[0]
    changed_code = approved_code.replace(
        '@scenario("authorization.feature",',
        '@scenario("missing-authorization.feature",',
        1,
    )
    with pytest.raises(ValueError, match="full generated-code approval"):
        _runner(tmp_path / "artifacts", generated_code=changed_code)

    assert not (tmp_path / "artifacts").exists()


def test_runner_classifies_trusted_collection_failure_without_final_evidence(
    tmp_path: Path,
) -> None:
    """Trusted pytest exit metadata still distinguishes collection failure."""
    runner = _runner(tmp_path / "artifacts")
    artifacts = runner._create_artifacts()
    artifacts.pytest_outcome_path.write_text(
        json.dumps({"exitstatus": 2, "reports": []}),
        encoding="utf-8",
    )

    with pytest.raises(
        UnexpectedPytestOutcomeError,
        match="pytest_collection_or_internal_error",
    ):
        runner._validate_pytest_outcome(2, artifacts)

    assert not artifacts.observation_path.exists()


def test_runner_enforces_bounded_timeout_and_retains_redacted_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing subprocess timeout enforcement must fail without a 60-second wait."""
    runner = _runner(tmp_path / "artifacts")
    monkeypatch.setattr(pytest_runner_module, "_PYTEST_TIMEOUT_SECONDS", 0.001)
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server, pytest.raises(ExecutionTimeoutError, match="60-second timeout"):
        runner.run(_target(server, revision="timed-out-revision"))

    artifacts = runner.last_artifacts
    assert artifacts is not None
    assert artifacts.stdout_path.exists()
    assert artifacts.stderr_path.exists()
    assert not artifacts.observation_path.exists()
    assert PASSWORD not in artifacts.stdout_path.read_text(encoding="utf-8")
    assert PASSWORD not in artifacts.stderr_path.read_text(encoding="utf-8")


def test_runner_rejects_a_corrupted_draft_instead_of_classifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed draft evidence must become an explicit execution error."""
    runner = _runner(tmp_path / "artifacts")

    def reject_corrupted_draft(
        self: ObservationWriter,
    ) -> list[tuple[str, int | bool]]:
        raise ValueError("corrupted draft")

    monkeypatch.setattr(ObservationWriter, "read_events", reject_corrupted_draft)
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server, pytest.raises(
        InvalidObservationError, match="invalid raw observation"
    ):
        runner.run(_target(server, revision="corrupt-observation-revision"))

    artifacts = runner.last_artifacts
    assert artifacts is not None
    assert artifacts.stdout_path.exists()
    assert artifacts.stderr_path.exists()
    assert not artifacts.observation_path.exists()
