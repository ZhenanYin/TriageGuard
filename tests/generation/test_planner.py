"""Tests for the bounded, replayable security-test planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triageguard.contracts.gherkin import render_gherkin
from triageguard.domain.models import RiskContract, TestPlan
from triageguard.generation.planner import PlanValidationError, create_test_plan
from triageguard.generation.primitives import PRIMITIVE_CATALOG
from triageguard.llm.replay_gateway import ReplayGateway

EXPECTED_PRIMITIVES = {
    "create_patient",
    "login_as_actor",
    "delete_patient",
    "read_patient",
    "record_http_status",
    "record_patient_exists",
    "record_control_http_status",
    "record_control_patient_exists_before",
    "record_control_patient_exists_after",
    "authorized_cleanup_patient",
}


@pytest.fixture
def contract() -> RiskContract:
    return RiskContract.model_validate(_read_fixture("approved_contract.json"))


@pytest.fixture
def planner_response() -> dict[str, object]:
    return _read_fixture("planner_response.json")


def test_catalog_contains_only_supported_operations():
    """An added operation would expand model authority without a reviewed runtime path."""
    assert set(PRIMITIVE_CATALOG) == EXPECTED_PRIMITIVES


def test_replay_planner_preserves_the_approved_actor_and_two_security_oracles(
    contract: RiskContract, planner_response: dict[str, object]
):
    """A planner change that drops either evidence channel must not yield an executable plan."""
    assert "post_action" in planner_response
    assert [
        operation["primitive"]
        for operation in planner_response["controls"][0]["operations"]
    ] == [
        "create_patient",
        "login_as_actor",
        "read_patient",
        "delete_patient",
        "read_patient",
    ]
    plan = create_test_plan(
        contract,
        render_gherkin(contract),
        ReplayGateway({"test_plan": planner_response}),
    )

    assert isinstance(plan, TestPlan)
    assert plan.contract_id == contract.contract_id
    assert plan.action.primitive == "delete_patient"
    assert [(assertion.primitive, assertion.expected_value) for assertion in plan.assertions] == [
        ("record_http_status", 403),
        ("record_patient_exists", True),
    ]
    assert plan.givens[0].captures == ["$patient_id"]
    assert plan.action.inputs == {
        "patient_id": "$patient_id",
        "actor_session": "$actor_session",
    }
    assert plan.action.captures == ["$delete_status"]
    assert [operation.model_dump() for operation in plan.post_action] == [
        {
            "primitive": "read_patient",
            "inputs": {
                "patient_id": "$patient_id",
                "actor_session": "$actor_session",
            },
            "captures": ["$patient_exists"],
        }
    ]
    assert [assertion.observed_field for assertion in plan.assertions] == [
        "$delete_status",
        "$patient_exists",
    ]
    assert plan.givens[1].inputs["actor"] == "clerk"
    assert plan.controls[0].name == "authorized administrator deletion control"
    assert [
        operation.captures for operation in plan.controls[0].operations
    ] == [
        ["$control_patient_id"],
        ["$control_actor_session"],
        ["$control_patient_exists_before"],
        ["$control_delete_status"],
        ["$control_patient_exists"],
    ]
    assert [assertion.observed_field for assertion in plan.controls[0].assertions] == [
        "$control_patient_exists_before",
        "$control_delete_status",
        "$control_patient_exists",
    ]
    assert [operation.inputs["patient_id"] for operation in plan.cleanup] == [
        "$patient_id",
        "$control_patient_id",
    ]


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda plan: plan["action"].update({"primitive": "execute_shell"}),
            "unknown_primitive",
        ),
        (
            lambda plan: plan.update(
                {
                    "assertions": [
                        assertion
                        for assertion in plan["assertions"]
                        if assertion["primitive"] != "record_patient_exists"
                    ]
                }
            ),
            "patient_persistence_assertion_missing",
        ),
        (
            lambda plan: plan["givens"][1]["inputs"].update({"actor": "administrator"}),
            "actor_changed",
        ),
        (lambda plan: plan.update({"controls": []}), "authorized_control_missing"),
    ],
)
def test_planner_rejects_a_plan_that_weakens_the_approved_experiment(
    contract: RiskContract,
    planner_response: dict[str, object],
    mutate,
    reason_code: str,
):
    """Removing an allowlist, evidence, actor, or control guard must stop planning."""
    response = json.loads(json.dumps(planner_response))
    mutate(response)

    with pytest.raises(PlanValidationError) as error:
        create_test_plan(
            contract,
            render_gherkin(contract),
            ReplayGateway({"test_plan": response}),
        )

    assert reason_code in error.value.reason_codes


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda plan: plan["givens"][0].update(
                {
                    "primitive": "delete_patient",
                    "inputs": {
                        "patient_id": "$patient_id",
                        "actor_session": "$actor_session",
                    },
                    "captures": ["$delete_status"],
                }
            ),
            "primitive_not_allowed_in_setup",
        ),
        (
            lambda plan: plan.update(
                {
                    "assertions": [
                        assertion
                        for assertion in plan["assertions"]
                        if assertion["primitive"] != "record_http_status"
                    ]
                }
            ),
            "http_status_assertion_missing",
        ),
        (lambda plan: plan.update({"cleanup": []}), "authorized_cleanup_missing"),
        (
            lambda plan: plan["action"]["inputs"].update(
                {"patient_id": "$other_patient"}
            ),
            "patient_reference_mismatched",
        ),
        (
            lambda plan: plan["assertions"][0].update(
                {"observed_field": "$forged_status"}
            ),
            "evidence_reference_invalid",
        ),
        (
            lambda plan: plan["action"].update({"captures": []}),
            "required_capture_missing",
        ),
        (
            lambda plan: plan["controls"][0]["operations"][0].update(
                {"captures": ["$actor_session"]}
            ),
            "duplicate_capture",
        ),
        (
            lambda plan: plan["action"]["inputs"].update(
                {"patient_id": "$unbound_patient"}
            ),
            "reference_unbound",
        ),
        (
            lambda plan: plan["controls"][0].update(
                {
                    "operations": list(
                        reversed(plan["controls"][0]["operations"])
                    )
                }
            ),
            "control_operation_sequence_invalid",
        ),
        (
            lambda plan: plan["controls"][0]["operations"].insert(
                1,
                {
                    "primitive": "login_as_actor",
                    "inputs": {"actor": "clerk"},
                    "captures": ["$replacement_session"],
                },
            ),
            "control_operation_sequence_invalid",
        ),
        (
            lambda plan: plan["controls"][0]["operations"][3]["inputs"].update(
                {"actor_session": "$actor_session"}
            ),
            "control_operation_sequence_invalid",
        ),
    ],
)
def test_planner_rejects_unbound_or_reordered_experiment_dataflow(
    contract: RiskContract,
    planner_response: dict[str, object],
    mutate,
    reason_code: str,
):
    """A plan that disconnects any security fact from the approved resource must fail."""
    response = json.loads(json.dumps(planner_response))
    mutate(response)

    with pytest.raises(PlanValidationError) as error:
        create_test_plan(
            contract,
            render_gherkin(contract),
            ReplayGateway({"test_plan": response}),
        )

    assert reason_code in error.value.reason_codes


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda plan: plan["assertions"].append(
                {
                    "primitive": "record_http_status",
                    "observed_field": "$forged_status",
                    "expected_value": 403,
                }
            ),
            "primary_assertion_unbound",
        ),
        (
            lambda plan: plan["assertions"].append(
                {
                    "primitive": "record_patient_exists",
                    "observed_field": "$patient_exists",
                    "expected_value": False,
                }
            ),
            "primary_assertion_contradictory",
        ),
        (
            lambda plan: plan["controls"][0]["assertions"][0].update(
                {"observed_field": "$delete_status"}
            ),
            "control_assertion_cross_phase",
        ),
        (
            lambda plan: plan["assertions"].append(plan["assertions"][0].copy()),
            "primary_assertion_duplicate",
        ),
        (
            lambda plan: plan.update(
                {"assertions": list(reversed(plan["assertions"]))}
            ),
            "primary_assertion_reordered",
        ),
    ],
)
def test_planner_rejects_additive_or_reordered_oracles(
    contract: RiskContract,
    planner_response: dict[str, object],
    mutate,
    reason_code: str,
):
    """One valid oracle cannot authorize contradictory, extra, or cross-phase facts."""
    response = json.loads(json.dumps(planner_response))
    mutate(response)

    with pytest.raises(PlanValidationError) as error:
        create_test_plan(
            contract,
            render_gherkin(contract),
            ReplayGateway({"test_plan": response}),
        )

    assert reason_code in error.value.reason_codes


def _read_fixture(name: str) -> dict[str, object]:
    fixture_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "patient_delete_authorization"
        / name
    )
    return json.loads(fixture_path.read_text())
