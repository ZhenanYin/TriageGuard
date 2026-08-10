"""AST and semantic-fidelity tests for generated pytest-bdd code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triageguard.contracts.gherkin import render_gherkin
from triageguard.domain.models import RiskContract, TestPlan
from triageguard.generation.validator import validate_generated_code


@pytest.fixture
def contract() -> RiskContract:
    return RiskContract.model_validate(_read_fixture("approved_contract.json"))


@pytest.fixture
def plan() -> TestPlan:
    return TestPlan.model_validate(_read_fixture("planner_response.json"))


@pytest.fixture
def gherkin(contract: RiskContract) -> str:
    return render_gherkin(contract)


@pytest.fixture
def valid_code() -> str:
    return str(_read_fixture("generator_response.json")["code"])


def test_validator_approves_exact_replay_code_and_reports_derived_facts(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Removing independent AST derivation would trust model-declared fidelity."""
    report = validate_generated_code(valid_code, contract, plan, gherkin)

    assert report.approved is True
    assert report.reason_codes == []
    assert report.implemented_steps == [
        "a test patient exists",
        "the clerk is authenticated",
        "the clerk has privileges: View Patients",
        "the clerk lacks privileges: Delete Patients",
        "the clerk attempts to delete the patient",
        "the deletion request is denied",
        "the patient remains",
        "evidence is collected through HTTP deletion status",
        "evidence is collected through patient existence after deletion attempt",
    ]
    assert set(report.used_primitives) == {
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
    assert len(report.code_sha256) == 64


@pytest.mark.parametrize(
    "forbidden",
    [
        "pytest.skip('not ready')",
        "subprocess.run(['curl'])",
        "eval('1 + 1')",
        "os.getenv('PASSWORD', 'Admin123')",
        "os.system('curl http://target')",
        "from os import system\nsystem('curl http://target')",
        "import os as safe\nsafe.system('curl http://target')",
    ],
)
def test_validator_rejects_forbidden_code(
    valid_code: str,
    forbidden: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A generated test cannot escape the runtime boundary or conceal missing config."""
    report = validate_generated_code(
        valid_code + "\n" + forbidden, contract, plan, gherkin
    )

    assert report.approved is False


def test_validator_rejects_a_missing_exact_step_implementation(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A scenario must not run with one approved contract step silently unbound."""
    changed = valid_code.replace(
        '@given("the clerk lacks privileges: Delete Patients")',
        '@given("the clerk lacks privileges: Delete Everything")',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "step_implementation_missing" in report.reason_codes


def test_validator_rejects_an_invented_runtime_helper(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A plausible helper name must not expand the reviewed runtime capability set."""
    changed = valid_code.replace(
        "    delete_status = openmrs_client.delete_patient(patient_id, actor_session)",
        "    openmrs_client.force_delete_patient(patient_id)\n"
        "    delete_status = openmrs_client.delete_patient(patient_id, actor_session)",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_method_unknown" in report.reason_codes


def test_validator_rejects_error_page_only_assertions(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Page text cannot replace the contract's status and persisted-state oracles."""
    changed = valid_code.replace(
        "assert delete_status == 403", 'assert "Access Denied" in error_page'
    ).replace("assert patient_exists is True", 'assert "Patient" in error_page')

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "denial_assertion_missing" in report.reason_codes
    assert "patient_existence_assertion_missing" in report.reason_codes


def test_validator_rejects_swallowed_setup_or_assertion_failures(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Catching failures would let setup or a security oracle appear successful."""
    changed = (
        valid_code
        + "\ntry:\n"
        + "    openmrs_client.create_patient()\n"
        + "except Exception:\n"
        + "    pass\n"
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "exception_swallowing" in report.reason_codes


def test_validator_rejects_a_return_that_can_bypass_an_oracle(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A step must not have a success path that exits before its security assertion."""
    changed = valid_code.replace(
        "    assert delete_status == 403",
        "    if delete_status:\n        return\n    assert delete_status == 403",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "assertion_return_bypass" in report.reason_codes


def test_validator_rejects_runtime_calls_disconnected_from_plan_references(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Using another resource would sever the generated test from its controlled patient."""
    changed = valid_code.replace(
        "openmrs_client.delete_patient(patient_id, actor_session)",
        "openmrs_client.delete_patient(other_patient_id, actor_session)",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_dataflow_mismatch" in report.reason_codes


def test_validator_rejects_an_extra_allowlisted_primitive_occurrence(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
) -> None:
    """An allowlisted method name cannot expand the exact typed plan."""
    changed = valid_code.replace(
        "        observation_writer.record_control_http_status("
        "control_delete_status)",
        "        observation_writer.record_control_http_status("
        "control_delete_status)\n"
        "        observation_writer.record_control_http_status("
        "control_delete_status)",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "planned_primitive_occurrence_mismatch" in report.reason_codes


def test_validator_rejects_the_primary_read_before_the_delete_phase(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
) -> None:
    """The bound patient read must remain a post-action operation, not a pre-read."""
    read = (
        "    patient_exists = openmrs_client.read_patient("
        "patient_id, actor_session)\n"
    )
    changed = valid_code.replace(read, "", 1).replace(
        "    delete_status = openmrs_client.delete_patient("
        "patient_id, actor_session)\n",
        read
        + "    delete_status = openmrs_client.delete_patient("
        "patient_id, actor_session)\n",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "planned_primitive_phase_mismatch" in report.reason_codes


def test_validator_rejects_an_unreachable_setup_primitive(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
) -> None:
    """A setup primitive must execute directly in its approved Given phase."""
    changed = valid_code.replace(
        "    patient_id = openmrs_client.create_patient()",
        "    if False:\n"
        "        patient_id = openmrs_client.create_patient()",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "planned_primitive_phase_mismatch" in report.reason_codes


def test_validator_rejects_a_fabricated_observation_value(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Observation calls must receive captured runtime facts, not expected constants."""
    changed = valid_code.replace(
        "observation_writer.record_http_status(delete_status)",
        "observation_writer.record_http_status(403)",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "http_status_observation_mismatched" in report.reason_codes


def test_validator_rejects_literal_target_credentials(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Generated source cannot embed a credential or fallback outside the harness."""
    changed = valid_code.replace(
        'password=os.environ["OPENMRS_PASSWORD"]',
        'password="Admin123"',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_configuration_invalid" in report.reason_codes


def test_validator_rejects_a_missing_authorized_control_oracle(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """An admin call without its exact outcome checks is not a positive control."""
    changed = valid_code.replace(
        "assert control_patient_exists is False",
        'assert "deleted" in control_page',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "authorized_control_assertion_missing" in report.reason_codes


def test_validator_rejects_a_skip_decorator_without_parentheses(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A bare pytest skip marker must not evade call-only skip detection."""
    changed = valid_code.replace(
        '@scenario("authorization.feature", "patient-delete-authz-001")',
        '@pytest.mark.skip\n'
        '@scenario("authorization.feature", "patient-delete-authz-001")',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "skip_forbidden" in report.reason_codes


def test_validator_rejects_an_extra_destructive_runtime_call(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Repeating an allowlisted delete still expands the canonical plan."""
    changed = valid_code.replace(
        "    delete_status = openmrs_client.delete_patient(patient_id, actor_session)",
        "    openmrs_client.delete_patient(patient_id, actor_session)\n"
        "    delete_status = openmrs_client.delete_patient(patient_id, actor_session)",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_dataflow_mismatch" in report.reason_codes


def test_validator_rejects_known_methods_on_an_unbound_fixture(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A method name alone cannot prove the receiver is the allowlisted runtime client."""
    changed = valid_code.replace(
        "    delete_status = openmrs_client.delete_patient(patient_id, actor_session)",
        "    delete_status = fixture_client.delete_patient(patient_id, actor_session)",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_receiver_unknown" in report.reason_codes


def test_validator_reports_a_noncanonical_plan_without_analyzing_missing_oracles(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A weakened plan must produce a denial report instead of crashing validation."""
    weakened = plan.model_copy(update={"assertions": [plan.assertions[0]]})

    report = validate_generated_code(valid_code, contract, weakened, gherkin)

    assert report.approved is False
    assert "plan_patient_persistence_assertion_missing" in report.reason_codes


def test_validator_rejects_fabricated_reassignments_after_runtime_calls(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Required calls do not authorize replacing their returned evidence with constants."""
    changed = valid_code.replace(
        "    patient_exists = openmrs_client.read_patient(patient_id, actor_session)",
        "    patient_exists = openmrs_client.read_patient(patient_id, actor_session)\n"
        "    delete_status = 403\n"
        "    patient_exists = True",
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_result_provenance_invalid" in report.reason_codes


def test_validator_rejects_unreachable_control_and_cleanup(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Dead AST nodes cannot satisfy the authorized control or cleanup requirements."""
    changed = valid_code.replace(
        "        control_patient_id = openmrs_client.create_patient()",
        "        if False:\n"
        "            control_patient_id = openmrs_client.create_patient()",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "authorized_control_unreachable" in report.reason_codes


def test_validator_requires_cleanup_in_a_finally_path(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A control failure must not prevent the authorized cleanup operation."""
    cleanup = (
        "control_cleanup_complete = "
        "openmrs_client.authorized_cleanup_patient(\n"
        "                control_patient_id\n"
        "            )"
    )
    changed = valid_code.replace(
        "    try:\n",
        f"    {cleanup}\n    try:\n",
        1,
    ).replace(
        "        if control_patient_id is not None:\n"
        f"            {cleanup}",
        "        if control_patient_id is not None:\n"
        "            pass",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "authorized_cleanup_not_guaranteed" in report.reason_codes


def test_validator_rejects_replacing_the_shared_evidence_context(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Replacing the context can disconnect later assertions from runtime results."""
    changed = valid_code.replace(
        '    test_context["patient_exists"] = patient_exists',
        '    test_context["patient_exists"] = patient_exists\n'
        '    test_context = {"delete_status": 403, "patient_exists": True}',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_result_provenance_invalid" in report.reason_codes


def test_validator_rejects_mutating_harness_environment_values(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Required environment reads cannot be preceded by generated target overrides."""
    changed = valid_code.replace(
        "import os\n",
        'import os\nos.environ["OPENMRS_BASE_URL"] = "http://other-target"\n',
        1,
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_configuration_mutated" in report.reason_codes


def test_validator_rejects_destructuring_overwrites_of_runtime_evidence(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Tuple bindings cannot evade canonical evidence provenance checks."""
    changed = valid_code.replace(
        '    test_context["delete_status"] = delete_status',
        "    delete_status, patient_exists = 403, True\n"
        '    test_context["delete_status"] = delete_status',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_result_provenance_invalid" in report.reason_codes


def test_validator_rejects_control_wrapped_in_a_dead_loop(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """A post-yield control hidden under `while False` is not executable."""
    changed = valid_code.replace(
        "        control_patient_id = openmrs_client.create_patient()",
        "        while False:\n"
        "            control_patient_id = openmrs_client.create_patient()",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "authorized_control_unreachable" in report.reason_codes


def test_validator_rejects_context_rebinding_through_a_loop_target(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Loop targets cannot replace the shared context with fabricated evidence."""
    changed = valid_code.replace(
        '    delete_status = test_context["delete_status"]',
        '    for test_context in [{"delete_status": 403}]:\n'
        '        pass\n'
        '    delete_status = test_context["delete_status"]',
    )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_result_provenance_invalid" in report.reason_codes


def test_validator_rejects_unreachable_control_assertions(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
):
    """Dead assertions cannot satisfy the authorized positive control."""
    changed = valid_code.replace(
        "        assert control_delete_status == 204",
        "        while False:\n"
        "            assert control_delete_status == 204",
        1,
    ).replace(
        "        assert control_patient_exists is False",
        "        while False:\n"
        "            assert control_patient_exists is False",
        1,
    )
    assert changed != valid_code

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "authorized_control_unreachable" in report.reason_codes


@pytest.mark.parametrize(
    "mutate",
    [
        lambda code: code.replace(
            '@scenario("authorization.feature", "patient-delete-authz-001")\n',
            "",
            1,
        ),
        lambda code: code.replace("authorization.feature", "other.feature", 1),
        lambda code: code.replace(
            '"patient-delete-authz-001")', '"different-scenario")', 1
        ),
        lambda code: code
        + '\n@scenario("authorization.feature", "patient-delete-authz-001")\n'
        + "def test_duplicate_scenario():\n    pass\n",
        lambda code: code + "\ndef test_empty_collection():\n    pass\n",
        lambda code: code.replace(
            '@scenario("authorization.feature", "patient-delete-authz-001")',
            'scenarios("authorization.feature")',
            1,
        ),
    ],
    ids=[
        "missing",
        "wrong-feature",
        "wrong-title",
        "extra-binding",
        "ordinary-empty-test",
        "bulk-scenarios-call",
    ],
)
def test_validator_requires_one_exact_scenario_binding(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutate,
):
    """Step decorators cannot substitute for the approved executable scenario binding."""
    report = validate_generated_code(mutate(valid_code), contract, plan, gherkin)

    assert report.approved is False
    assert "scenario_binding_invalid" in report.reason_codes


@pytest.mark.parametrize(
    "mutation",
    [
        "OpenMrsTestClient = object",
        "ObservationWriter += object",
        "del scenario",
        "OpenMrsTestClient.delete_patient = None",
        "openmrs_client.read_patient = None",
        "def given():\n    pass",
        "class ObservationWriter:\n    pass",
        "def shadow_runtime(OpenMrsTestClient):\n    pass",
        "for then in []:\n    pass",
        'setattr(OpenMrsTestClient, "delete_patient", None)',
        'delattr(ObservationWriter, "record_http_status")',
    ],
    ids=[
        "class-assignment",
        "class-augassign",
        "decorator-delete",
        "class-method-assignment",
        "fixture-method-assignment",
        "fake-decorator-function",
        "fake-runtime-class",
        "parameter-shadow",
        "loop-target-shadow",
        "setattr",
        "delattr",
    ],
)
def test_validator_rejects_protected_runtime_monkeypatching(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutation: str,
):
    """Generated code cannot replace any imported or allowlisted capability."""
    report = validate_generated_code(
        valid_code + "\n" + mutation + "\n", contract, plan, gherkin
    )

    assert report.approved is False
    assert "protected_symbol_rebound" in report.reason_codes


@pytest.mark.parametrize(
    "mutate",
    [
        lambda code: code.replace(
            "import os\n", "import os\nfrom os import environ\n", 1
        ),
        lambda code: code.replace(
            'os.environ["OPENMRS_PASSWORD"]', 'os.environ["PASSWORD"]', 1
        ),
        lambda code: code + "\nenvironment = os.environ\n",
        lambda code: code + '\npassword = "Admin123"\n',
        lambda code: code + '\nos.environ["OPENMRS_PASSWORD"] += "x"\n',
        lambda code: code + '\ndel os.environ["OPENMRS_PASSWORD"]\n',
        lambda code: code + "\nos.environ.clear()\n",
        lambda code: code + '\nos.environ.update({"OPENMRS_PASSWORD": "x"})\n',
        lambda code: code + '\nos.environ.setdefault("OPENMRS_PASSWORD", "x")\n',
        lambda code: code + '\nos.environ.pop("OPENMRS_PASSWORD")\n',
        lambda code: code + '\nos.putenv("OPENMRS_PASSWORD", "x")\n',
        lambda code: code + '\nos.unsetenv("OPENMRS_PASSWORD")\n',
        lambda code: code.replace(
            'password=os.environ["OPENMRS_PASSWORD"]',
            'password=os.getenv("OPENMRS_PASSWORD")',
            1,
        ),
    ],
    ids=[
        "from-environ",
        "unapproved-key",
        "environment-alias",
        "hardcoded-password",
        "augassign",
        "delete",
        "clear",
        "update",
        "setdefault",
        "pop",
        "putenv",
        "unsetenv",
        "getenv-read",
    ],
)
def test_validator_allows_only_approved_environment_reads(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutate,
):
    """Harness URL and credentials are immutable required-key reads."""
    report = validate_generated_code(mutate(valid_code), contract, plan, gherkin)

    assert report.approved is False
    assert "runtime_configuration_invalid" in report.reason_codes


@pytest.mark.parametrize(
    "mutate",
    [
        lambda code: code.replace(
            "        if control_patient_id is not None:",
            "        if control_patient_id is None:",
            1,
        ),
        lambda code: code.replace(
            "        if control_patient_id is not None:\n"
            "            control_cleanup_complete = "
            "openmrs_client.authorized_cleanup_patient(\n"
            "                control_patient_id\n"
            "            )",
            "        if control_patient_id is not None:\n"
            "            pass\n"
            "        else:\n"
            "            control_cleanup_complete = "
            "openmrs_client.authorized_cleanup_patient(\n"
            "                control_patient_id\n"
            "            )",
            1,
        ),
    ],
    ids=["inverted-guard", "relocated-to-else"],
)
def test_validator_requires_cleanup_in_the_executing_guard_body(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutate,
):
    """Predicate text is insufficient when cleanup occupies the wrong branch."""
    changed = mutate(valid_code)
    assert changed != valid_code
    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "authorized_control_structure_invalid" in report.reason_codes


@pytest.mark.parametrize(
    "mutate",
    [
        lambda code: _wrap_observations(code, "    if False:\n"),
        lambda code: _wrap_observations(
            code, "    if True:\n        pass\n    else:\n"
        ),
        lambda code: code.replace(
            "    observation_writer.record_http_status(delete_status)",
            "    return\n"
            "    observation_writer.record_http_status(delete_status)",
            1,
        ),
        lambda code: code.replace(
            "    observation_writer.record_http_status(delete_status)",
            "    raise\n"
            "    observation_writer.record_http_status(delete_status)",
            1,
        ),
        lambda code: _wrap_observations(code, "    while False:\n"),
        lambda code: _move_observations_after_oracles(code),
    ],
    ids=[
        "if-false",
        "dead-else",
        "after-return",
        "after-raise",
        "while-false",
        "after-consuming-assertions",
    ],
)
def test_validator_requires_reachable_observations_before_oracles(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutate,
):
    """Observation method names count only on the executable action path before oracles."""
    report = validate_generated_code(mutate(valid_code), contract, plan, gherkin)

    assert report.approved is False
    assert "observation_unreachable" in report.reason_codes


@pytest.mark.parametrize(
    "mutation",
    [
        "__test__ = False",
        "test_patient_delete_authorization.__test__ = False",
    ],
    ids=["module-collection-disabled", "scenario-collection-disabled"],
)
def test_validator_rejects_disabling_pytest_collection(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutation: str,
):
    """An exact scenario binding is invalid if pytest is instructed not to collect it."""
    report = validate_generated_code(
        valid_code + "\n" + mutation + "\n", contract, plan, gherkin
    )

    assert report.approved is False
    assert "test_collection_disabled" in report.reason_codes


@pytest.mark.parametrize(
    "mutation",
    [
        "def pytest_sessionfinish(session, exitstatus):\n    pass",
        'pytest_plugins = ("forged_outcome_plugin",)',
    ],
)
def test_validator_rejects_generated_pytest_extension_points(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutation: str,
) -> None:
    """Generated source cannot register hooks or plugins around the trusted gate."""
    report = validate_generated_code(
        f"{valid_code}\n{mutation}\n", contract, plan, gherkin
    )

    assert report.approved is False
    assert "pytest_extension_forbidden" in report.reason_codes


@pytest.mark.parametrize(
    "mutation",
    [
        'openmrs_client.__dict__["delete_patient"] = None',
        (
            '    openmrs_client.__dict__["delete_patient"] = lambda *_: 403\n'
            '    openmrs_client.__dict__["read_patient"] = lambda *_: True\n'
            '    observation_writer.__dict__["record_http_status"] = lambda *_: None\n'
            '    observation_writer.__dict__["record_patient_exists"] = lambda *_: None'
        ),
    ],
    ids=["runtime-dunder-write", "full-fake-runtime"],
)
def test_validator_rejects_dunder_runtime_monkeypatching(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutation: str,
):
    """Runtime instance dictionaries cannot replace reviewed capability methods."""
    changed = valid_code + "\n" + mutation + "\n"
    if mutation.startswith("    "):
        changed = valid_code.replace(
            "    patient_id = test_context[\"patient_id\"]",
            mutation + "\n    patient_id = test_context[\"patient_id\"]",
            1,
        )

    report = validate_generated_code(changed, contract, plan, gherkin)

    assert report.approved is False
    assert "dunder_attribute_forbidden" in report.reason_codes


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            'os.__dict__["environ"]["OPENMRS_PASSWORD"] = "x"',
            "dunder_attribute_forbidden",
        ),
        (
            'os.getenv.__globals__["environ"]["OPENMRS_PASSWORD"] = "x"',
            "dunder_attribute_forbidden",
        ),
        ('globals()["os"].environ["OPENMRS_PASSWORD"] = "x"', "forbidden_call"),
        ('locals()["os"].environ["OPENMRS_PASSWORD"] = "x"', "forbidden_call"),
        ('vars(os)["environ"]["OPENMRS_PASSWORD"] = "x"', "forbidden_call"),
        ('getattr(os, "environ")["OPENMRS_PASSWORD"] = "x"', "forbidden_call"),
        ('setattr(os, "environ", {})', "forbidden_call"),
    ],
    ids=[
        "os-dict",
        "getenv-globals",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
    ],
)
def test_validator_rejects_indirect_environment_access(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutation: str,
    reason: str,
):
    """Reflection and namespace dictionaries cannot redirect harness configuration."""
    report = validate_generated_code(
        valid_code + "\n" + mutation + "\n", contract, plan, gherkin
    )

    assert report.approved is False
    assert reason in report.reason_codes


@pytest.mark.parametrize(
    "mutate",
    [
        lambda code: code.replace(
            '    patient_id = test_context["patient_id"]',
            '    yield\n    patient_id = test_context["patient_id"]',
            1,
        ),
        lambda code: code.replace(
            "    observation_writer.record_http_status(delete_status)",
            "    yield\n    observation_writer.record_http_status(delete_status)",
            1,
        ),
    ],
    ids=["yield-before-action", "yield-before-observation"],
)
def test_validator_rejects_yield_in_bdd_execution_functions(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutate,
):
    """A BDD step cannot become a generator that defers its action or observation."""
    report = validate_generated_code(mutate(valid_code), contract, plan, gherkin)

    assert report.approved is False
    assert "generated_yield_forbidden" in report.reason_codes


@pytest.mark.parametrize(
    "mutation",
    [
        (
            "safe_os = os\n"
            'safe_os.environ["OPENMRS_PASSWORD"] = "redirected"'
        ),
        "safe_os, safe_runtime = os, OpenMrsTestClient",
        'protected = [os, OpenMrsTestClient, given, scenario]',
        'protected = {"os": os, "runtime": ObservationWriter, "bdd": when}',
        "safe_os = (os_alias := os)",
        "def carry(value):\n    return value\nsafe_os = carry(os)",
        "safe_client = OpenMrsTestClient",
        "safe_decorator = then",
        "safe_fixture = pytest.fixture",
    ],
    ids=[
        "exact-os-redirect",
        "tuple-alias",
        "list-container",
        "dict-container",
        "walrus-alias",
        "helper-argument",
        "runtime-class-alias",
        "bdd-decorator-alias",
        "pytest-fixture-alias",
    ],
)
def test_validator_rejects_propagating_protected_imports(
    valid_code: str,
    contract: RiskContract,
    plan: TestPlan,
    gherkin: str,
    mutation: str,
):
    """Protected imports are capabilities, not values generated code may propagate."""
    report = validate_generated_code(
        valid_code + "\n" + mutation + "\n", contract, plan, gherkin
    )

    assert report.approved is False
    assert "protected_symbol_propagated" in report.reason_codes


def _wrap_observations(code: str, prefix: str) -> str:
    observations = (
        "    observation_writer.record_http_status(delete_status)\n"
        "    observation_writer.record_patient_exists(patient_exists)"
    )
    indented = observations.replace("    observation", "        observation")
    return code.replace(observations, prefix + indented, 1)


def _move_observations_after_oracles(code: str) -> str:
    code = code.replace(
        "    observation_writer.record_http_status(delete_status)\n"
        "    observation_writer.record_patient_exists(patient_exists)\n",
        "",
        1,
    )
    code = code.replace(
        "def the_deletion_request_is_denied(test_context):",
        "def the_deletion_request_is_denied(test_context, observation_writer):",
        1,
    ).replace(
        "    assert delete_status == 403",
        "    assert delete_status == 403\n"
        "    observation_writer.record_http_status(delete_status)",
        1,
    )
    return code.replace(
        "def the_patient_remains(test_context):",
        "def the_patient_remains(test_context, observation_writer):",
        1,
    ).replace(
        "    assert patient_exists is True",
        "    assert patient_exists is True\n"
        "    observation_writer.record_patient_exists(patient_exists)",
        1,
    )


def _read_fixture(name: str) -> dict[str, object]:
    fixture_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "patient_delete_authorization"
        / name
    )
    return json.loads(fixture_path.read_text())
