import json
from pathlib import Path

import pytest

from triageguard.contracts.gherkin import render_gherkin, validate_gherkin_alignment
from triageguard.domain.models import RiskContract


@pytest.fixture
def contract() -> RiskContract:
    fixture_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "patient_delete_authorization"
        / "approved_contract.json"
    )
    return RiskContract.model_validate(json.loads(fixture_path.read_text()))


def test_rendered_gherkin_preserves_security_meaning(contract: RiskContract):
    """Changing either security oracle must break the rendered contract."""
    feature = render_gherkin(contract)

    assert "Given a test patient exists" in feature
    assert "And the clerk is authenticated" in feature
    assert "When the clerk attempts to delete the patient" in feature
    assert "Then the deletion request is denied" in feature
    assert "And the patient remains" in feature


def test_alignment_rejects_changed_actor(contract: RiskContract):
    """An actor substitution must not receive deterministic approval."""
    feature = render_gherkin(contract).replace("clerk", "administrator")

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "actor_changed" in report.reason_codes


def test_alignment_approves_the_deterministically_rendered_contract(
    contract: RiskContract,
):
    """A renderer output must include each required security assertion."""
    report = validate_gherkin_alignment(contract, render_gherkin(contract))

    assert report.approved is True
    assert report.reason_codes == []
    assert set(report.matched_steps) == {
        "scenario",
        "actor",
        "action",
        "denial_oracle",
        "persistence_oracle",
    }


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda feature: "\n".join(
                line
                for line in feature.splitlines()
                if not line.lstrip().startswith("Scenario:")
            ),
            "scenario_missing",
        ),
        (
            lambda feature: feature.replace(
                "Scenario: patient-delete-authz-001",
                "Scenario: renamed-contract",
            ),
            "scenario_title_changed",
        ),
        (
            lambda feature: feature.replace(
                "  Scenario: patient-delete-authz-001",
                "  Scenario: patient-delete-authz-001\n"
                "  Scenario: patient-delete-authz-001",
            ),
            "scenario_duplicate",
        ),
    ],
)
def test_alignment_requires_one_exact_contract_scenario_header(
    contract: RiskContract,
    mutate,
    reason_code: str,
) -> None:
    """Missing, renamed, or duplicate scenarios must stop generation binding."""
    report = validate_gherkin_alignment(contract, mutate(render_gherkin(contract)))

    assert report.approved is False
    assert reason_code in report.reason_codes


def test_rendering_without_preconditions_still_starts_with_a_given(
    contract: RiskContract,
):
    """A structurally valid scenario needs a Given even for sparse contracts."""
    sparse_contract = contract.model_copy(update={"preconditions": []})

    feature = render_gherkin(sparse_contract)

    assert "Given the clerk has privileges: View Patients" in feature


def test_alignment_rejects_an_action_moved_to_a_given(contract: RiskContract):
    """A literal action does not count when it occurs before the When phase."""
    feature = render_gherkin(contract).replace(
        "When the clerk attempts to delete the patient",
        "Given the clerk attempts to delete the patient",
    )

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "action_wrong_phase" in report.reason_codes


def test_alignment_rejects_another_actor_in_the_actual_when(contract: RiskContract):
    """The approved actor cannot be retained only in a non-action step."""
    feature = render_gherkin(contract).replace(
        "When the clerk attempts to delete the patient",
        "Given the clerk attempts to delete the patient\n"
        "    When the administrator attempts to delete the patient",
    )

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "actor_changed" in report.reason_codes
    assert "action_wrong_phase" in report.reason_codes


@pytest.mark.parametrize(
    "replacement",
    [
        "Then the deletion request is denied unless the clerk is authenticated",
        "Then the deletion request is denied but the patient is deleted",
    ],
)
def test_alignment_rejects_qualified_or_contradictory_denial(
    contract: RiskContract,
    replacement: str,
):
    """A denial oracle must be the exact approved Then step."""
    feature = render_gherkin(contract).replace(
        "Then the deletion request is denied",
        replacement,
    )

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "denial_oracle_changed" in report.reason_codes


def test_alignment_rejects_a_qualified_persistence_oracle(contract: RiskContract):
    """A persistence oracle cannot be weakened by an added qualifier."""
    feature = render_gherkin(contract).replace(
        "And the patient remains",
        "And the patient remains only in the client cache",
    )

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "persistence_oracle_changed" in report.reason_codes


def test_alignment_rejects_duplicated_critical_step(contract: RiskContract):
    """Two literal denial assertions are ambiguous rather than approved."""
    feature = render_gherkin(contract).replace(
        "Then the deletion request is denied",
        "Then the deletion request is denied\n"
        "    Then the deletion request is denied",
    )

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "duplicate_denial_oracle" in report.reason_codes


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "Then the deletion request is denied",
            (
                "When the administrator attempts to delete the patient\n"
                "    Then the deletion request is denied"
            ),
        ),
        (
            "Then the deletion request is denied",
            (
                "When the clerk attempts to export every patient\n"
                "    Then the deletion request is denied"
            ),
        ),
        (
            "Then the deletion request is denied",
            (
                "Then the deletion request succeeds\n"
                "    Then the deletion request is denied"
            ),
        ),
        (
            "And the patient remains",
            (
                "And the patient remains\n"
                "    And the patient is deleted"
            ),
        ),
    ],
)
def test_alignment_rejects_an_added_conflicting_executable_step(
    contract: RiskContract,
    needle: str,
    replacement: str,
):
    """A valid expected scenario cannot hide an additional executable outcome."""
    feature = render_gherkin(contract).replace(needle, replacement)

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "executable_step_added" in report.reason_codes


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "And the patient remains",
            (
                "And the patient remains\n"
                "    * the patient is deleted"
            ),
        ),
        (
            "Then the deletion request is denied",
            (
                "* the clerk exports every patient\n"
                "    Then the deletion request is denied"
            ),
        ),
        (
            "Given a test patient exists",
            (
                "* an unauthorized mutation has already occurred\n"
                "    Given a test patient exists"
            ),
        ),
    ],
)
def test_alignment_rejects_an_added_wildcard_executable_step(
    contract: RiskContract,
    needle: str,
    replacement: str,
):
    """Wildcard steps are executable and cannot be omitted from alignment."""
    feature = render_gherkin(contract).replace(needle, replacement)

    report = validate_gherkin_alignment(contract, feature)

    assert report.approved is False
    assert "executable_step_added" in report.reason_codes
