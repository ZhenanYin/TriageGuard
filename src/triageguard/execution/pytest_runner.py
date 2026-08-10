"""Isolated subprocess execution for one approved generated pytest."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triageguard.domain.models import RiskContract, RuntimeObservation, TestPlan
from triageguard.generation.validator import (
    CodeValidationReport,
    validate_generated_code,
)
from triageguard.provenance import canonical_sha256
from triageguard.runtime import ObservationWriter

_PYTEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ExecutionTarget:
    """Private connection material and public revision label for one target."""

    base_url: str
    username: str
    password: str = field(repr=False)
    revision: str

    def __post_init__(self) -> None:
        for label, value in (
            ("base_url", self.base_url),
            ("username", self.username),
            ("password", self.password),
            ("revision", self.revision),
        ):
            if not value:
                raise ValueError(f"{label} must not be empty")
        _validate_controlled_base_url(self.base_url)


@dataclass(frozen=True)
class ExecutionArtifacts:
    """Persistent inspectable files from one isolated subprocess run."""

    run_directory: Path
    pytest_config_path: Path
    feature_path: Path
    test_path: Path
    observation_path: Path
    pytest_outcome_path: Path
    stdout_path: Path
    stderr_path: Path


class ExecutionError(RuntimeError):
    """Base class for explicit execution-boundary failures."""

    def __init__(self, message: str, artifacts: ExecutionArtifacts) -> None:
        super().__init__(message)
        self.artifacts = artifacts


class MissingObservationError(ExecutionError):
    """The subprocess ended without all required raw observation facts."""


class InvalidObservationError(ExecutionError):
    """The subprocess wrote malformed or structurally incomplete raw facts."""


class ExecutionTimeoutError(ExecutionError):
    """The generated pytest exceeded the fixed bounded execution time."""


class UnexpectedPytestOutcomeError(ExecutionError):
    """The subprocess outcome was not a trusted approved experiment result."""

    def __init__(
        self, reason_code: str, artifacts: ExecutionArtifacts
    ) -> None:
        super().__init__(
            f"{reason_code}: trusted pytest outcome rejected",
            artifacts,
        )
        self.reason_code = reason_code


class GeneratedCodeRejectedError(ValueError):
    """The execution boundary independently rejected generated source."""

    def __init__(self, report: CodeValidationReport) -> None:
        reasons = ", ".join(report.reason_codes) or "approval_missing"
        super().__init__(
            f"full generated-code approval failed: {reasons}"
        )
        self.report = report


class _OutcomeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    lineno: int = Field(gt=0)
    function: str = Field(min_length=1)


class _PhaseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nodeid: str = Field(min_length=1)
    when: Literal["setup", "call", "teardown"]
    outcome: Literal["passed", "failed", "skipped"]
    exception_type: str | None
    frames: list[_OutcomeFrame]


class _PytestOutcomeEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exitstatus: int
    reports: list[_PhaseOutcome]


@dataclass(frozen=True)
class _ApprovedExecutionBinding:
    nodeid: str
    primary_assertions: frozenset[tuple[str, int]]


class PytestRunner:
    """Execute exact approved source and finalize its unclassified raw facts."""

    def __init__(
        self,
        *,
        generated_code: str,
        contract: RiskContract,
        plan: TestPlan,
        gherkin: str,
        artifact_root: str | Path,
    ) -> None:
        if not generated_code:
            raise ValueError("generated_code must not be empty")
        if not gherkin:
            raise ValueError("gherkin must not be empty")
        if not isinstance(contract, RiskContract):
            raise TypeError("contract must be a RiskContract")
        if not isinstance(plan, TestPlan):
            raise TypeError("plan must be a TestPlan")
        self._generated_code = generated_code
        self._contract = RiskContract.model_validate(
            contract.model_dump(mode="json")
        )
        self._plan = TestPlan.model_validate(plan.model_dump(mode="json"))
        self._gherkin = gherkin
        self._contract_sha256 = canonical_sha256(
            self._contract.model_dump(mode="json")
        )
        self._artifact_root = Path(artifact_root)
        self._source_root = Path(__file__).resolve().parents[2]
        self._require_full_code_approval()
        self._approved_binding = _derive_approved_execution_binding(
            generated_code, gherkin
        )
        self._last_artifacts: ExecutionArtifacts | None = None

    @property
    def last_artifacts(self) -> ExecutionArtifacts | None:
        return self._last_artifacts

    def run(self, target: ExecutionTarget) -> RuntimeObservation:
        if not isinstance(target, ExecutionTarget):
            raise TypeError("target must be an ExecutionTarget")
        self._require_full_code_approval()
        artifacts = self._create_artifacts()
        self._last_artifacts = artifacts
        artifacts.pytest_config_path.write_text(
            "[pytest]\naddopts =\ntestpaths = .\n"
            "python_files = test_authorization.py\n",
            encoding="utf-8",
        )
        artifacts.feature_path.write_text(self._gherkin, encoding="utf-8")
        artifacts.test_path.write_text(self._generated_code, encoding="utf-8")

        environment = {
            "OPENMRS_BASE_URL": target.base_url,
            "OPENMRS_USERNAME": target.username,
            "OPENMRS_PASSWORD": target.password,
            "TRIAGEGUARD_OBSERVATION_PATH": str(artifacts.observation_path),
            "PYTHONPATH": str(self._source_root),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
        }
        argv = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_bdd.plugin",
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
        ]
        try:
            completed = subprocess.run(
                argv,
                cwd=artifacts.run_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=_PYTEST_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._write_logs(
                artifacts,
                _coerce_output(error.stdout),
                _coerce_output(error.stderr),
                target,
            )
            raise ExecutionTimeoutError(
                "generated pytest exceeded the 60-second timeout",
                artifacts,
            ) from error

        self._write_logs(
            artifacts,
            completed.stdout,
            completed.stderr,
            target,
        )
        trusted_pytest_outcome = self._validate_pytest_outcome(
            completed.returncode, artifacts
        )
        writer = ObservationWriter(artifacts.observation_path)
        facts = self._read_exact_observation_facts(writer, artifacts)
        control_succeeded = trusted_pytest_outcome and (
            facts["control_resource_exists_before"] is True
            and facts["control_request_status"] == 204
            and facts["control_resource_exists_after"] is False
        )
        reason_code = (
            "pytest_completed_with_observation"
            if completed.returncode == 0
            else "pytest_failed_with_complete_observation"
        )
        observation = RuntimeObservation(
            revision=target.revision,
            setup_succeeded=True,
            action_attempted=True,
            control_succeeded=control_succeeded,
            control_request_status=facts["control_request_status"],
            control_resource_exists_before=facts[
                "control_resource_exists_before"
            ],
            control_resource_exists_after=facts["control_resource_exists_after"],
            request_status=facts["request_status"],
            resource_exists_after=facts["resource_exists_after"],
            pytest_exit_code=completed.returncode,
            reason_code=reason_code,
        )
        writer.write(observation, contract_sha256=self._contract_sha256)
        return observation

    @staticmethod
    def _read_exact_observation_facts(
        writer: ObservationWriter,
        artifacts: ExecutionArtifacts,
    ) -> dict[str, int | bool]:
        try:
            events = writer.read_events()
        except FileNotFoundError as error:
            raise MissingObservationError(
                "pytest exited without a complete raw observation",
                artifacts,
            ) from error
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            raise InvalidObservationError(
                "pytest wrote an invalid raw observation",
                artifacts,
            ) from error

        required_fields = (
            "request_status",
            "resource_exists_after",
            "control_resource_exists_before",
            "control_request_status",
            "control_resource_exists_after",
        )
        if any(
            sum(field == required_field for field, _ in events) != 1
            for required_field in required_fields
        ) or len(events) != len(required_fields):
            raise InvalidObservationError(
                "pytest must write exactly one raw event for each required field",
                artifacts,
            )
        return dict(events)

    def _require_full_code_approval(self) -> CodeValidationReport:
        report = validate_generated_code(
            self._generated_code,
            self._contract,
            self._plan,
            self._gherkin,
        )
        if not report.approved:
            raise GeneratedCodeRejectedError(report)
        return report

    def _validate_pytest_outcome(
        self, return_code: int, artifacts: ExecutionArtifacts
    ) -> bool:
        """Validate the structured phases and return the explicit control fact."""
        try:
            outcome = _PytestOutcomeEnvelope.model_validate_json(
                artifacts.pytest_outcome_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise UnexpectedPytestOutcomeError(
                "pytest_outcome_missing_or_invalid",
                artifacts,
            ) from error
        if outcome.exitstatus != return_code:
            raise UnexpectedPytestOutcomeError(
                "pytest_exit_status_mismatch",
                artifacts,
            )
        if return_code in {2, 3, 4, 5}:
            raise UnexpectedPytestOutcomeError(
                "pytest_collection_or_internal_error",
                artifacts,
            )
        reports = outcome.reports
        if _is_strict_setup_failure(
            return_code,
            reports,
            self._approved_binding.nodeid,
        ):
            raise UnexpectedPytestOutcomeError(
                "pytest_setup_failed",
                artifacts,
            )
        expected_phases = ["setup", "call", "teardown"]
        if (
            len(reports) != 3
            or [report.when for report in reports] != expected_phases
            or any(
                report.nodeid != self._approved_binding.nodeid
                for report in reports
            )
        ):
            raise UnexpectedPytestOutcomeError(
                "pytest_report_structure_invalid",
                artifacts,
            )
        setup_report, call_report, teardown_report = reports
        if setup_report.outcome != "passed":
            raise UnexpectedPytestOutcomeError(
                "pytest_setup_failed",
                artifacts,
            )
        if teardown_report.outcome != "passed":
            raise UnexpectedPytestOutcomeError(
                "pytest_teardown_or_control_failed",
                artifacts,
            )
        if return_code == 0:
            if any(
                report.outcome != "passed"
                or report.exception_type is not None
                or report.frames
                for report in reports
            ):
                raise UnexpectedPytestOutcomeError(
                    "pytest_success_report_invalid",
                    artifacts,
                )
            return True
        draft_path = Path(f"{artifacts.observation_path}.events.jsonl")
        if call_report.outcome == "failed" and not draft_path.exists():
            raise UnexpectedPytestOutcomeError(
                "pytest_setup_or_preaction_failed",
                artifacts,
            )
        final_frame = call_report.frames[-1] if call_report.frames else None
        expected_failure = (
            return_code == 1
            and call_report.outcome == "failed"
            and call_report.exception_type == "AssertionError"
            and final_frame is not None
            and Path(final_frame.path).resolve() == artifacts.test_path.resolve()
            and (final_frame.function, final_frame.lineno)
            in self._approved_binding.primary_assertions
        )
        if not expected_failure:
            raise UnexpectedPytestOutcomeError(
                "pytest_unapproved_call_failure",
                artifacts,
            )
        return True

    def _create_artifacts(self) -> ExecutionArtifacts:
        self._artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_directory = Path(
            tempfile.mkdtemp(prefix="triageguard-run-", dir=self._artifact_root)
        )
        return ExecutionArtifacts(
            run_directory=run_directory,
            pytest_config_path=run_directory / "pytest.ini",
            feature_path=run_directory / "authorization.feature",
            test_path=run_directory / "test_authorization.py",
            observation_path=run_directory / "observation.json",
            pytest_outcome_path=run_directory / "pytest-outcome.json",
            stdout_path=run_directory / "pytest.stdout.txt",
            stderr_path=run_directory / "pytest.stderr.txt",
        )

    @staticmethod
    def _write_logs(
        artifacts: ExecutionArtifacts,
        stdout: str,
        stderr: str,
        target: ExecutionTarget,
    ) -> None:
        secrets = (target.username, target.password)
        artifacts.stdout_path.write_text(
            _redact(stdout, secrets), encoding="utf-8"
        )
        artifacts.stderr_path.write_text(
            _redact(stderr, secrets), encoding="utf-8"
        )


def _redact(value: str, secrets: tuple[str, ...]) -> str:
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _is_strict_setup_failure(
    return_code: int,
    reports: list[_PhaseOutcome],
    expected_nodeid: str,
) -> bool:
    if return_code != 1 or len(reports) != 2:
        return False
    setup_report, teardown_report = reports
    return (
        [setup_report.when, teardown_report.when] == ["setup", "teardown"]
        and setup_report.nodeid == expected_nodeid
        and teardown_report.nodeid == expected_nodeid
        and setup_report.outcome == "failed"
        and setup_report.exception_type is not None
        and bool(setup_report.frames)
        and teardown_report.outcome == "passed"
        and teardown_report.exception_type is None
        and not teardown_report.frames
    )


def _derive_approved_execution_binding(
    generated_code: str, gherkin: str
) -> _ApprovedExecutionBinding:
    try:
        tree = ast.parse(generated_code)
    except SyntaxError as error:
        raise ValueError("generated_code must be valid Python") from error
    primary_steps = _primary_then_steps(gherkin)
    scenario_functions: list[str] = []
    primary_assertions: set[tuple[str, int]] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            decorator_name, decorator_text = _decorator_binding(decorator)
            if decorator_name == "scenario":
                scenario_functions.append(node.name)
            if decorator_name == "then" and decorator_text in primary_steps:
                assertions = [
                    child for child in ast.walk(node) if isinstance(child, ast.Assert)
                ]
                if len(assertions) != 1:
                    raise ValueError(
                        "each approved primary Then implementation must contain one assertion"
                    )
                if not _is_canonical_primary_assertion(
                    decorator_text, assertions[0]
                ):
                    raise ValueError(
                        "approved Then implementation changed its canonical primary assertion"
                    )
                primary_assertions.add((node.name, assertions[0].lineno))
    if len(scenario_functions) != 1 or len(primary_assertions) != len(primary_steps):
        raise ValueError("generated_code must bind one scenario and both primary assertions")
    return _ApprovedExecutionBinding(
        nodeid=f"test_authorization.py::{scenario_functions[0]}",
        primary_assertions=frozenset(primary_assertions),
    )


def _primary_then_steps(gherkin: str) -> frozenset[str]:
    steps: list[str] = []
    after_when = False
    for line in gherkin.splitlines():
        stripped = line.strip()
        if stripped.startswith("When "):
            after_when = True
            continue
        if not after_when:
            continue
        if stripped.startswith("Then "):
            steps.append(stripped[5:])
        elif stripped.startswith("And ") and steps:
            steps.append(stripped[4:])
        if len(steps) == 2:
            return frozenset(steps)
    raise ValueError("gherkin must contain two primary post-action assertions")


def _decorator_binding(decorator: ast.expr) -> tuple[str | None, str | None]:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name):
        return None, None
    if not decorator.args or not isinstance(decorator.args[-1], ast.Constant):
        return decorator.func.id, None
    value = decorator.args[-1].value
    return decorator.func.id, value if isinstance(value, str) else None


def _is_canonical_primary_assertion(
    step_text: str, assertion: ast.Assert
) -> bool:
    expected_source = {
        "the deletion request is denied": "assert delete_status == 403",
        "the patient remains": "assert patient_exists is True",
    }.get(step_text)
    if expected_source is None or assertion.msg is not None:
        return False
    expected = ast.parse(expected_source).body[0]
    assert isinstance(expected, ast.Assert)
    return ast.dump(assertion.test, include_attributes=False) == ast.dump(
        expected.test, include_attributes=False
    )


def _validate_controlled_base_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            "base_url must be an exact HTTP loopback target with an explicit port"
        ) from error
    valid = (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.username is None
        and parsed.password is None
        and port is not None
        and 1 <= port <= 65535
        and parsed.path == ""
        and parsed.query == ""
        and parsed.fragment == ""
        and value == f"http://127.0.0.1:{port}"
    )
    if not valid:
        raise ValueError(
            "base_url must be an exact HTTP loopback target with an explicit port"
        )
