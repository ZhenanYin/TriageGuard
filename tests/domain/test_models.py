import pytest
from pydantic import ValidationError

from triageguard.domain.models import RiskContract, RuntimeObservation, TestPlan


def test_contract_requires_independent_observable_evidence():
    """A contract without observable evidence cannot support a conclusion."""
    with pytest.raises(ValidationError):
        RiskContract(
            contract_id="contract-1",
            actor="clerk",
            actor_privileges=["View Patients"],
            missing_privileges=["Delete Patients"],
            preconditions=["patient exists"],
            action="delete patient",
            secure_expectation="request is denied",
            observable_evidence=[],
            base_expectation="secure",
            candidate_expectation="vulnerable",
            cleanup=["remove test patient"],
        )


def test_runtime_observation_separates_setup_from_security_behavior():
    """Failed setup must remain distinct from a measured security outcome."""
    observation = RuntimeObservation(
        revision="base",
        setup_succeeded=False,
        action_attempted=False,
        control_succeeded=None,
        control_request_status=None,
        control_resource_exists_before=None,
        control_resource_exists_after=None,
        request_status=None,
        resource_exists_after=None,
        pytest_exit_code=1,
        reason_code="fixture_setup_failed",
    )

    assert observation.setup_succeeded is False
    assert observation.security_behavior is None


@pytest.mark.parametrize("control_succeeded", [False, None])
def test_runtime_observation_requires_successful_control_for_security_behavior(
    control_succeeded: bool | None,
) -> None:
    """An apparent denial without a working authorized delete is vacuous."""
    observation = RuntimeObservation(
        revision="base",
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=control_succeeded,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=403,
        resource_exists_after=True,
        pytest_exit_code=0,
        reason_code="raw_execution_complete",
    )

    assert observation.security_behavior is None


def test_runtime_observation_requires_explicit_control_fact() -> None:
    """Omitting control outcome must fail schema validation, not imply success."""
    with pytest.raises(ValidationError):
        RuntimeObservation(
            revision="base",
            setup_succeeded=True,
            action_attempted=True,
            request_status=403,
            resource_exists_after=True,
            pytest_exit_code=0,
            reason_code="raw_execution_complete",
        )


def test_runtime_observation_rejects_coerced_control_fact() -> None:
    """Text that merely looks truthy cannot establish control success."""
    with pytest.raises(ValidationError):
        RuntimeObservation(
            revision="base",
            setup_succeeded=True,
            action_attempted=True,
            control_succeeded="true",  # type: ignore[arg-type]
            control_request_status=204,
            control_resource_exists_before=True,
            control_resource_exists_after=False,
            request_status=403,
            resource_exists_after=True,
            pytest_exit_code=0,
            reason_code="raw_execution_complete",
        )


def test_runtime_observation_requires_complete_raw_control_facts() -> None:
    """A derived control flag without its independently observed tuple is insufficient."""
    with pytest.raises(ValidationError):
        RuntimeObservation(
            revision="base",
            setup_succeeded=True,
            action_attempted=True,
            control_succeeded=True,
            request_status=403,
            resource_exists_after=True,
            pytest_exit_code=0,
            reason_code="raw_execution_complete",
        )


def test_authenticated_unauthorized_status_is_not_the_approved_secure_tuple() -> None:
    """Only 403 plus persistence is secure for the authenticated clerk contract."""
    observation = RuntimeObservation(
        revision="base",
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=True,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=401,
        resource_exists_after=True,
        pytest_exit_code=0,
        reason_code="raw_execution_complete",
    )

    assert observation.security_behavior is None


def test_plan_requires_at_least_one_assertion():
    """A plan with no assertion cannot evaluate a security expectation."""
    with pytest.raises(ValidationError):
        TestPlan(
            plan_id="plan-1",
            contract_id="contract-1",
            givens=[],
            action={"primitive": "delete_patient", "inputs": {"patient_id": "123"}},
            post_action=[],
            assertions=[],
            controls=[],
            cleanup=[],
        )


def test_research_artifacts_reject_unknown_fields():
    """Unexpected fields must not silently enter immutable research records."""
    with pytest.raises(ValidationError):
        RuntimeObservation(
            revision="base",
            setup_succeeded=True,
            action_attempted=True,
            control_succeeded=True,
            control_request_status=204,
            control_resource_exists_before=True,
            control_resource_exists_after=False,
            request_status=403,
            resource_exists_after=True,
            pytest_exit_code=0,
            reason_code="expected_denial",
            invented_field=True,
        )
