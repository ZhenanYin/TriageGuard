"""Structured, allowlisted LLM planning with deterministic fidelity checks."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import ValidationError

from triageguard.contracts.gherkin import validate_gherkin_alignment
from triageguard.domain.models import (
    RiskContract,
    TestAssertion,
    TestControl,
    TestOperation,
    TestPlan,
)
from triageguard.generation.primitives import (
    PRIMITIVE_CATALOG,
    primitive_catalog_prompt_data,
)
from triageguard.llm.gateway import ModelRequest, StructuredModelGateway

_PLANNER_RULES = (
    "Select only primitives in primitive_catalog; do not write Python or shell commands.",
    (
        "Preserve the contract ID, primary actor, actor privileges, missing "
        "privileges, action, secure expectation, observable evidence, and cleanup "
        "intent exactly."
    ),
    "Use canonical $references captured in order; never invent resource, session, status, or evidence values.",
    "Use one primary deletion action followed by an explicit patient read and assert both HTTP denial and persisted patient state.",
    "Create a distinct control patient, verify it exists, delete it as administrator, then verify it is absent.",
    "Capture every runtime result used by an assertion and leave no assertion reference unbound.",
    "Include explicit idempotent authorized cleanup for both controlled patients.",
)
_SYSTEM_PROMPT = "\n".join(  # noqa: FLY002 - stable line-oriented prompt
    (
        "You are a constrained authorization-security test planner.",
        "Return only JSON matching the supplied schema.",
        "Select allowlisted primitives; never write Python, HTTP requests, shell commands, or new helpers.",
        "Do not alter the approved actor, privileges, expected behavior, observable evidence, or cleanup.",
    )
)
_REFERENCE_PATTERN = re.compile(r"^\$[a-z][a-z0-9_]*$")
_PRIMARY_ASSERTIONS = (
    ("record_http_status", "$delete_status", 403),
    ("record_patient_exists", "$patient_exists", True),
)
_CONTROL_ASSERTIONS = (
    (
        "record_control_patient_exists_before",
        "$control_patient_exists_before",
        True,
    ),
    ("record_control_http_status", "$control_delete_status", 204),
    (
        "record_control_patient_exists_after",
        "$control_patient_exists",
        False,
    ),
)


class PlanValidationError(ValueError):
    """A model plan failed deterministic approval for one or more reason codes."""

    def __init__(self, reason_codes: Iterable[str]) -> None:
        self.reason_codes = tuple(dict.fromkeys(reason_codes))
        super().__init__(", ".join(self.reason_codes))


def create_test_plan(
    contract: RiskContract,
    gherkin: str,
    gateway: StructuredModelGateway,
) -> TestPlan:
    """Request and independently validate one primitive-only security test plan."""
    alignment = validate_gherkin_alignment(contract, gherkin)
    if not alignment.approved:
        raise PlanValidationError(f"gherkin_{reason}" for reason in alignment.reason_codes)

    request = ModelRequest(
        purpose="test_plan",
        system_prompt=_SYSTEM_PROMPT,
        payload={
            "approved_contract": contract.model_dump(mode="json"),
            "exact_gherkin": gherkin,
            "primitive_catalog": primitive_catalog_prompt_data(),
            "rules": list(_PLANNER_RULES),
        },
        output_schema=TestPlan.model_json_schema(),
        max_output_tokens=1500,
    )
    response = gateway.generate(request)
    try:
        plan = TestPlan.model_validate(response.data)
    except ValidationError as error:
        raise PlanValidationError(["model_plan_invalid"]) from error

    validate_test_plan(contract, plan)
    return plan


def validate_test_plan(contract: RiskContract, plan: TestPlan) -> None:
    """Reject any plan that expands authority or disconnects a fact from its source."""
    reason_codes: list[str] = []
    if plan.contract_id != contract.contract_id:
        _add_reason(reason_codes, "contract_id_changed")

    _validate_operations(plan.givens, "setup", reason_codes)
    _validate_operation(plan.action, "action", reason_codes)
    _validate_operations(plan.post_action, "post_action", reason_codes)
    _validate_assertions(plan.assertions, reason_codes)
    for control in plan.controls:
        _validate_operations(control.operations, "control", reason_codes)
        _validate_assertions(control.assertions, reason_codes)
    _validate_operations(plan.cleanup, "cleanup", reason_codes)

    _validate_primary_actor(contract, plan, reason_codes)
    _validate_primary_action(plan, reason_codes)
    _validate_primary_post_action(plan, reason_codes)
    captures = _validate_ordered_dataflow(contract, plan, reason_codes)
    _validate_primary_oracles(plan, captures, reason_codes)
    _validate_authorized_control(plan.controls, captures, reason_codes)
    _validate_cleanup(plan.cleanup, captures, reason_codes)
    _validate_all_assertion_references(plan, captures, reason_codes)

    if reason_codes:
        raise PlanValidationError(reason_codes)


def _validate_operations(
    operations: list[TestOperation], phase: str, reason_codes: list[str]
) -> None:
    for operation in operations:
        _validate_operation(operation, phase, reason_codes)


def _validate_operation(
    operation: TestOperation, phase: str, reason_codes: list[str]
) -> None:
    primitive = PRIMITIVE_CATALOG.get(operation.primitive)
    if primitive is None:
        _add_reason(reason_codes, "unknown_primitive")
        return
    if phase not in primitive.allowed_phases:
        _add_reason(reason_codes, f"primitive_not_allowed_in_{phase}")
    if set(operation.inputs) != set(primitive.input_types):
        _add_reason(reason_codes, "primitive_inputs_invalid")
    if any(not value for value in operation.inputs.values()):
        _add_reason(reason_codes, "primitive_inputs_invalid")
    if any(capture not in primitive.output_names for capture in operation.captures):
        _add_reason(reason_codes, "primitive_capture_invalid")


def _validate_assertions(
    assertions: list[TestAssertion], reason_codes: list[str]
) -> None:
    for assertion in assertions:
        primitive = PRIMITIVE_CATALOG.get(assertion.primitive)
        if primitive is None:
            _add_reason(reason_codes, "unknown_primitive")
            continue
        if "assertion" not in primitive.allowed_phases:
            _add_reason(reason_codes, "primitive_not_allowed_in_assertion")
        if not _is_reference(assertion.observed_field):
            _add_reason(reason_codes, "assertion_observed_field_missing")


def _validate_primary_actor(
    contract: RiskContract, plan: TestPlan, reason_codes: list[str]
) -> None:
    actor_logins = [
        operation for operation in plan.givens if operation.primitive == "login_as_actor"
    ]
    if len(actor_logins) != 1 or actor_logins[0].inputs.get("actor") != contract.actor:
        _add_reason(reason_codes, "actor_changed")


def _validate_primary_action(plan: TestPlan, reason_codes: list[str]) -> None:
    if plan.action.primitive != "delete_patient":
        _add_reason(reason_codes, "primary_action_invalid")


def _validate_primary_post_action(
    plan: TestPlan, reason_codes: list[str]
) -> None:
    if len(plan.post_action) != 1 or plan.post_action[0].primitive != "read_patient":
        _add_reason(reason_codes, "primary_post_action_read_missing")


def _validate_ordered_dataflow(
    contract: RiskContract, plan: TestPlan, reason_codes: list[str]
) -> set[str]:
    """Bind one controlled resource and its derived facts in execution order."""
    captures: set[str] = set()
    create_operations = [
        operation for operation in plan.givens if operation.primitive == "create_patient"
    ]
    if len(create_operations) != 1:
        _add_reason(reason_codes, "controlled_patient_missing")

    for operation in plan.givens:
        _validate_input_references(
            operation,
            captures,
            literal_actor=contract.actor,
            reason_codes=reason_codes,
        )
        _bind_captures(operation, "setup", captures, reason_codes)

    _validate_input_references(plan.action, captures, reason_codes=reason_codes)
    if plan.action.inputs.get("patient_id") != "$patient_id":
        _add_reason(reason_codes, "patient_reference_mismatched")
    if plan.action.inputs.get("actor_session") != "$actor_session":
        _add_reason(reason_codes, "primary_session_invalid")
    _bind_captures(plan.action, "action", captures, reason_codes)

    for operation in plan.post_action:
        _validate_input_references(operation, captures, reason_codes=reason_codes)
        if operation.inputs.get("patient_id") != "$patient_id":
            _add_reason(reason_codes, "patient_reference_mismatched")
        if operation.inputs.get("actor_session") != "$actor_session":
            _add_reason(reason_codes, "primary_session_invalid")
        _bind_captures(operation, "post_action", captures, reason_codes)

    for control in plan.controls:
        _bind_control_dataflow(control, captures, reason_codes)

    for operation in plan.cleanup:
        _validate_input_references(operation, captures, reason_codes=reason_codes)
        _bind_captures(operation, "cleanup", captures, reason_codes)
    return captures


def _validate_input_references(
    operation: TestOperation,
    captures: set[str],
    *,
    reason_codes: list[str],
    literal_actor: str | None = None,
) -> None:
    """Require every non-actor operation input to use a previously bound reference."""
    for name, value in operation.inputs.items():
        if name == "actor":
            if literal_actor is not None and value != literal_actor:
                _add_reason(reason_codes, "actor_changed")
            continue
        if not _is_reference(value) or value not in captures:
            _add_reason(reason_codes, "reference_unbound")


def _bind_captures(
    operation: TestOperation,
    phase: str,
    captures: set[str],
    reason_codes: list[str],
) -> None:
    primitive = PRIMITIVE_CATALOG.get(operation.primitive)
    if primitive is None:
        return
    expected = primitive.required_captures.get(phase)
    capture_tuple = tuple(operation.captures)
    allowed_dynamic = (
        operation.primitive == "read_patient"
        and phase == "control"
        and capture_tuple
        in {
            ("$control_patient_exists_before",),
            ("$control_patient_exists",),
        }
    ) or (
        operation.primitive == "authorized_cleanup_patient"
        and phase == "cleanup"
        and capture_tuple in {("$cleanup_complete",), ("$control_cleanup_complete",)}
    )
    if (expected is not None and capture_tuple != expected) or (
        expected is None and not allowed_dynamic
    ):
        _add_reason(reason_codes, "required_capture_missing")
    for capture in operation.captures:
        if not _is_reference(capture):
            _add_reason(reason_codes, "capture_reference_invalid")
            continue
        if capture in captures:
            _add_reason(reason_codes, "duplicate_capture")
            continue
        captures.add(capture)


def _validate_primary_oracles(
    plan: TestPlan, captures: set[str], reason_codes: list[str]
) -> None:
    _validate_assertion_sequence(
        plan.assertions,
        expected=_PRIMARY_ASSERTIONS,
        phase="primary",
        reason_codes=reason_codes,
    )
    if "$delete_status" not in captures:
        _add_reason(reason_codes, "primary_assertion_reference_invalid")
        _add_reason(reason_codes, "evidence_reference_invalid")
    if "$patient_exists" not in captures:
        _add_reason(reason_codes, "primary_assertion_reference_invalid")
        _add_reason(reason_codes, "evidence_reference_invalid")


def _bind_control_dataflow(
    control: TestControl, captures: set[str], reason_codes: list[str]
) -> None:
    expected = (
        ("create_patient", {}, ("$control_patient_id",)),
        ("login_as_actor", {"actor": "administrator"}, ("$control_actor_session",)),
        (
            "read_patient",
            {
                "patient_id": "$control_patient_id",
                "actor_session": "$control_actor_session",
            },
            ("$control_patient_exists_before",),
        ),
        (
            "delete_patient",
            {
                "patient_id": "$control_patient_id",
                "actor_session": "$control_actor_session",
            },
            ("$control_delete_status",),
        ),
        (
            "read_patient",
            {
                "patient_id": "$control_patient_id",
                "actor_session": "$control_actor_session",
            },
            ("$control_patient_exists",),
        ),
    )
    if len(control.operations) != len(expected):
        _add_reason(reason_codes, "control_operation_sequence_invalid")
    for index, operation in enumerate(control.operations):
        if index < len(expected):
            primitive, inputs, expected_captures = expected[index]
            if (
                operation.primitive != primitive
                or operation.inputs != inputs
                or tuple(operation.captures) != expected_captures
            ):
                _add_reason(reason_codes, "control_operation_sequence_invalid")
        if operation.primitive == "login_as_actor":
            if operation.inputs.get("actor") != "administrator":
                _add_reason(reason_codes, "control_administrator_missing")
        else:
            _validate_input_references(operation, captures, reason_codes=reason_codes)
        _bind_captures(operation, "control", captures, reason_codes)


def _validate_authorized_control(
    controls: list[TestControl], captures: set[str], reason_codes: list[str]
) -> None:
    if len(controls) != 1:
        _add_reason(reason_codes, "authorized_control_missing")
    for control in controls:
        _validate_assertion_sequence(
            control.assertions,
            expected=_CONTROL_ASSERTIONS,
            phase="control",
            reason_codes=reason_codes,
        )
    if not {
        "$control_patient_exists_before",
        "$control_delete_status",
        "$control_patient_exists",
    }.issubset(captures):
        _add_reason(reason_codes, "control_assertion_reference_invalid")

    for control in controls:
        has_administrator_login = any(
            operation.primitive == "login_as_actor"
            and operation.inputs.get("actor") == "administrator"
            for operation in control.operations
        )
        has_delete = any(
            operation.primitive == "delete_patient" for operation in control.operations
        )
        has_distinct_create = any(
            operation.primitive == "create_patient"
            and operation.captures == ["$control_patient_id"]
            for operation in control.operations
        )
        if has_administrator_login and has_delete and has_distinct_create:
            return
    _add_reason(reason_codes, "authorized_control_missing")


def _validate_assertion_sequence(
    assertions: list[TestAssertion],
    *,
    expected: tuple[tuple[str, str, str | int | bool], ...],
    phase: str,
    reason_codes: list[str],
) -> None:
    """Require every assertion in a phase to be one approved fact in order."""
    if len(assertions) > len(expected):
        _add_reason(reason_codes, f"{phase}_assertion_extra")
    if len(assertions) < len(expected):
        _add_reason(reason_codes, f"{phase}_assertion_missing")

    actual = [_assertion_key(assertion) for assertion in assertions]
    expected_keys = [_expected_assertion_key(item) for item in expected]
    if (
        len(assertions) == len(expected)
        and set(actual) == set(expected_keys)
        and actual != expected_keys
    ):
        _add_reason(reason_codes, f"{phase}_assertion_reordered")

    expected_by_fact = {
        (primitive, observed_field): expected_value
        for primitive, observed_field, expected_value in expected
    }
    expected_fields = {observed_field for _, observed_field, _ in expected}
    other_phase_fields = (
        {observed_field for _, observed_field, _ in _CONTROL_ASSERTIONS}
        if phase == "primary"
        else {observed_field for _, observed_field, _ in _PRIMARY_ASSERTIONS}
    )
    for actual_key in actual:
        if actual.count(actual_key) > 1:
            _add_reason(reason_codes, f"{phase}_assertion_duplicate")
    for assertion in assertions:
        fact = (assertion.primitive, assertion.observed_field)
        if assertion.observed_field not in expected_fields:
            _add_reason(
                reason_codes,
                f"{phase}_assertion_cross_phase"
                if assertion.observed_field in other_phase_fields
                else f"{phase}_assertion_unbound",
            )
            continue
        expected_value = expected_by_fact.get(fact)
        if expected_value is not None and not _same_value(
            assertion.expected_value, expected_value
        ):
            _add_reason(reason_codes, f"{phase}_assertion_contradictory")

    for index, required in enumerate(expected):
        if index >= len(assertions):
            _add_missing_oracle_reason(required[0], phase, reason_codes)
            continue
        assertion = assertions[index]
        primitive, observed_field, expected_value = required
        if assertion.primitive != primitive:
            _add_reason(reason_codes, f"{phase}_assertion_sequence_invalid")
            _add_missing_oracle_reason(primitive, phase, reason_codes)
            continue
        if assertion.observed_field != observed_field:
            _add_reason(reason_codes, f"{phase}_assertion_reference_invalid")
            if phase == "primary":
                _add_reason(reason_codes, "evidence_reference_invalid")
        if not _same_value(assertion.expected_value, expected_value):
            _add_reason(reason_codes, f"{phase}_assertion_value_invalid")

    for assertion in assertions[len(expected) :]:
        if assertion.primitive not in {item[0] for item in expected}:
            _add_reason(reason_codes, f"{phase}_assertion_sequence_invalid")


def _add_missing_oracle_reason(
    primitive: str, phase: str, reason_codes: list[str]
) -> None:
    if phase != "primary":
        return
    if primitive == "record_http_status":
        _add_reason(reason_codes, "http_status_assertion_missing")
    elif primitive == "record_patient_exists":
        _add_reason(reason_codes, "patient_persistence_assertion_missing")


def _assertion_key(assertion: TestAssertion) -> tuple[str, str, type[object], object]:
    return (
        assertion.primitive,
        assertion.observed_field,
        type(assertion.expected_value),
        assertion.expected_value,
    )


def _expected_assertion_key(
    assertion: tuple[str, str, str | int | bool],
) -> tuple[str, str, type[object], object]:
    primitive, observed_field, expected_value = assertion
    return primitive, observed_field, type(expected_value), expected_value


def _same_value(actual: str | int | bool, expected: str | int | bool) -> bool:
    return type(actual) is type(expected) and actual == expected


def _validate_cleanup(
    cleanup: list[TestOperation], captures: set[str], reason_codes: list[str]
) -> None:
    expected = [
        ("$patient_id", ["$cleanup_complete"]),
        ("$control_patient_id", ["$control_cleanup_complete"]),
    ]
    actual = [
        (operation.inputs.get("patient_id"), operation.captures)
        for operation in cleanup
        if operation.primitive == "authorized_cleanup_patient"
    ]
    if actual != expected or not {"$patient_id", "$control_patient_id"}.issubset(
        captures
    ):
        _add_reason(reason_codes, "authorized_cleanup_missing")


def _validate_all_assertion_references(
    plan: TestPlan,
    captures: set[str],
    reason_codes: list[str],
) -> None:
    assertions = list(plan.assertions)
    for control in plan.controls:
        assertions.extend(control.assertions)
    if any(assertion.observed_field not in captures for assertion in assertions):
        _add_reason(reason_codes, "assertion_reference_unbound")


def _is_reference(value: str) -> bool:
    return _REFERENCE_PATTERN.fullmatch(value) is not None


def _add_reason(reason_codes: list[str], reason_code: str) -> None:
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)
