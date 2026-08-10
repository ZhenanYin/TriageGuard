from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from triageguard.domain.models import RiskContract, RuntimeObservation
from triageguard.domain.statuses import WorkflowStatus
from triageguard.evidence import (
    UnsupportedRiskContractError,
    classify_differential,
)


def _contract() -> RiskContract:
    fixture_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "patient_delete_authorization"
        / "approved_contract.json"
    )
    return RiskContract.model_validate_json(
        fixture_path.read_text(encoding="utf-8")
    )


def _observation(
    revision: str,
    behavior: str,
    *,
    setup_succeeded: bool = True,
    action_attempted: bool = True,
    control_succeeded: bool | None = True,
    control_request_status: int | None = 204,
    control_resource_exists_before: bool | None = True,
    control_resource_exists_after: bool | None = False,
    pytest_exit_code: int | None = None,
    reason_code: str = "raw_execution_complete",
) -> RuntimeObservation:
    request_status: int | None
    resource_exists_after: bool | None
    if behavior == "secure":
        request_status = 403
        resource_exists_after = True
        default_exit_code = 0
    elif behavior == "vulnerable":
        request_status = 204
        resource_exists_after = False
        default_exit_code = 1
    elif behavior == "missing":
        request_status = None
        resource_exists_after = None
        default_exit_code = 1
    elif behavior == "unsupported":
        request_status = 200
        resource_exists_after = True
        default_exit_code = 0
    else:
        raise ValueError(f"unsupported test behavior: {behavior}")
    return RuntimeObservation(
        revision=revision,
        setup_succeeded=setup_succeeded,
        action_attempted=action_attempted,
        control_succeeded=control_succeeded,
        control_request_status=control_request_status,
        control_resource_exists_before=control_resource_exists_before,
        control_resource_exists_after=control_resource_exists_after,
        request_status=request_status,
        resource_exists_after=resource_exists_after,
        pytest_exit_code=(
            default_exit_code if pytest_exit_code is None else pytest_exit_code
        ),
        reason_code=reason_code,
    )


@pytest.mark.parametrize(
    ("base_behavior", "candidate_behavior", "status", "reason_code", "explanation"),
    [
        (
            "secure",
            "vulnerable",
            WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
            "candidate_regression_observed",
            (
                "The base denied unauthorized deletion and preserved the patient, "
                "while the candidate allowed deletion and removed the patient."
            ),
        ),
        (
            "vulnerable",
            "secure",
            WorkflowStatus.CANDIDATE_FIX_OBSERVED,
            "candidate_fix_observed",
            (
                "The base allowed unauthorized deletion and removed the patient, "
                "while the candidate denied deletion and preserved the patient."
            ),
        ),
        (
            "secure",
            "secure",
            WorkflowStatus.NO_REGRESSION_OBSERVED,
            "no_regression_observed",
            "Both revisions denied unauthorized deletion and preserved the patient.",
        ),
        (
            "vulnerable",
            "vulnerable",
            WorkflowStatus.PRE_EXISTING_RISK_OBSERVED,
            "pre_existing_risk_observed",
            "Both revisions allowed unauthorized deletion and removed the patient.",
        ),
    ],
)
def test_classifier_applies_the_fixed_differential_matrix(
    base_behavior: str,
    candidate_behavior: str,
    status: WorkflowStatus,
    reason_code: str,
    explanation: str,
) -> None:
    """Changing any matrix branch must change the observable classification."""
    base = [_observation("base-sha", base_behavior) for _ in range(3)]
    candidate = [
        _observation("candidate-sha", candidate_behavior) for _ in range(3)
    ]

    evidence = classify_differential(base, candidate, _contract())

    assert evidence.status is status
    assert evidence.reason_code == reason_code
    assert evidence.explanation == explanation
    assert evidence.repetitions == 3
    assert evidence.stable is True
    assert evidence.base_differing_run_indexes == []
    assert evidence.candidate_differing_run_indexes == []
    assert evidence.base is base[0]
    assert evidence.candidate is candidate[0]


@pytest.mark.parametrize(
    ("side", "field", "value", "reason_code", "explanation"),
    [
        (
            "base",
            "setup_succeeded",
            False,
            "base_setup_failed",
            "At least one base run did not complete setup; differential evidence is inconclusive.",
        ),
        (
            "candidate",
            "setup_succeeded",
            False,
            "candidate_setup_failed",
            "At least one candidate run did not complete setup; differential evidence is inconclusive.",
        ),
        (
            "base",
            "action_attempted",
            False,
            "base_action_not_attempted",
            "At least one base run did not attempt the approved action; differential evidence is inconclusive.",
        ),
        (
            "candidate",
            "action_attempted",
            False,
            "candidate_action_not_attempted",
            "At least one candidate run did not attempt the approved action; differential evidence is inconclusive.",
        ),
        (
            "base",
            "control_succeeded",
            False,
            "base_control_failed_or_missing",
            "At least one base run lacks a successful authorized control; differential evidence is inconclusive.",
        ),
        (
            "candidate",
            "control_succeeded",
            None,
            "candidate_control_failed_or_missing",
            "At least one candidate run lacks a successful authorized control; differential evidence is inconclusive.",
        ),
    ],
)
def test_execution_failures_take_precedence_over_behavior_and_instability(
    side: str,
    field: str,
    value: bool | None,
    reason_code: str,
    explanation: str,
) -> None:
    """A broken experiment cannot be upgraded to secure, vulnerable, or unstable."""
    base = [_observation("base-sha", "secure") for _ in range(2)]
    candidate = [_observation("candidate-sha", "vulnerable") for _ in range(2)]
    selected = base if side == "base" else candidate
    selected[1] = selected[1].model_copy(update={field: value})

    evidence = classify_differential(base, candidate, _contract())

    assert evidence.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert evidence.reason_code == reason_code
    assert evidence.explanation == explanation
    assert evidence.stable is False


@pytest.mark.parametrize("behavior", ["missing", "unsupported"])
def test_missing_or_unsupported_raw_fact_tuple_is_inconclusive(
    behavior: str,
) -> None:
    """Only the two approved HTTP/state tuples can support evidence."""
    base = [_observation("base-sha", behavior)]
    candidate = [_observation("candidate-sha", "secure")]

    evidence = classify_differential(base, candidate, _contract())

    assert evidence.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert evidence.reason_code == "base_raw_facts_missing_or_unsupported"
    assert evidence.explanation == (
        "At least one base run has missing or unsupported HTTP/state facts; "
        "differential evidence is inconclusive."
    )
    assert evidence.stable is False


def test_authenticated_primary_401_is_inconclusive_not_secure() -> None:
    """The approved authenticated-clerk contract supports only a 403 denial."""
    required_control_fields = {
        "control_request_status",
        "control_resource_exists_before",
        "control_resource_exists_after",
    }
    assert required_control_fields <= set(RuntimeObservation.model_fields)
    base = RuntimeObservation.model_validate(
        {
            **_observation("base-sha", "secure").model_dump(mode="json"),
            "request_status": 401,
            "control_request_status": 204,
            "control_resource_exists_before": True,
            "control_resource_exists_after": False,
        }
    )
    candidate = RuntimeObservation.model_validate(
        {
            **_observation("candidate-sha", "secure").model_dump(mode="json"),
            "control_request_status": 204,
            "control_resource_exists_before": True,
            "control_resource_exists_after": False,
        }
    )

    evidence = classify_differential([base], [candidate], _contract())

    assert evidence.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert evidence.reason_code == "base_raw_facts_missing_or_unsupported"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("control_resource_exists_before", False),
        ("control_request_status", 404),
        ("control_resource_exists_after", True),
    ],
)
def test_classifier_requires_the_exact_raw_control_tuple(
    field: str,
    value: object,
) -> None:
    """A truthy derived flag cannot override a vacuous or malformed control."""
    required_control_fields = {
        "control_request_status",
        "control_resource_exists_before",
        "control_resource_exists_after",
    }
    assert required_control_fields <= set(RuntimeObservation.model_fields)
    base_payload = _observation("base-sha", "secure").model_dump(mode="json")
    base_payload.update(
        {
            "control_request_status": 204,
            "control_resource_exists_before": True,
            "control_resource_exists_after": False,
            field: value,
        }
    )
    candidate_payload = _observation(
        "candidate-sha", "vulnerable"
    ).model_dump(mode="json")
    candidate_payload.update(
        {
            "control_request_status": 204,
            "control_resource_exists_before": True,
            "control_resource_exists_after": False,
        }
    )

    evidence = classify_differential(
        [RuntimeObservation.model_validate(base_payload)],
        [RuntimeObservation.model_validate(candidate_payload)],
        _contract(),
    )

    assert evidence.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert evidence.reason_code == "base_control_failed_or_missing"


@pytest.mark.parametrize("control_succeeded", [False, None])
def test_apparent_denial_without_authorized_control_is_inconclusive(
    control_succeeded: bool | None,
) -> None:
    """A generally broken delete endpoint must not look securely authorized."""
    base = [
        _observation(
            "base-sha", "secure", control_succeeded=control_succeeded
        )
    ]
    candidate = [_observation("candidate-sha", "vulnerable")]

    evidence = classify_differential(base, candidate, _contract())

    assert evidence.status is WorkflowStatus.EXECUTION_INCONCLUSIVE
    assert evidence.status is not WorkflowStatus.VALIDATED_EVIDENCE
    assert evidence.reason_code == "base_control_failed_or_missing"


def test_supported_repetition_disagreement_is_unstable_with_one_based_indexes(
) -> None:
    """A changed supported tuple after run one must identify its exact run index."""
    base = [
        _observation("base-sha", "secure"),
        _observation("base-sha", "vulnerable"),
        _observation("base-sha", "secure"),
    ]
    candidate = [
        _observation("candidate-sha", "vulnerable"),
        _observation("candidate-sha", "vulnerable"),
        _observation("candidate-sha", "secure"),
    ]

    evidence = classify_differential(base, candidate, _contract())

    assert evidence.status is WorkflowStatus.UNSTABLE_RESULT
    assert evidence.reason_code == "security_relevant_tuple_unstable"
    assert evidence.explanation == (
        "Repeated security-relevant facts differed from run 1; "
        "one-based differing run indexes are reported separately for base and candidate."
    )
    assert evidence.stable is False
    assert evidence.base_differing_run_indexes == [2]
    assert evidence.candidate_differing_run_indexes == [3]


def test_pytest_exit_and_reason_codes_do_not_affect_raw_fact_classification(
) -> None:
    """Classifier changes based on prose or pytest metadata would violate determinism."""
    base = [
        _observation(
            "base-sha",
            "secure",
            pytest_exit_code=73,
            reason_code="pretend_vulnerable",
        ),
        _observation(
            "base-sha",
            "secure",
            pytest_exit_code=0,
            reason_code="different_arbitrary_code",
        ),
    ]
    candidate = [
        _observation(
            "candidate-sha",
            "secure",
            pytest_exit_code=1,
            reason_code="pretend_secure",
        ),
        _observation(
            "candidate-sha",
            "secure",
            pytest_exit_code=99,
            reason_code="more_arbitrary_code",
        ),
    ]

    evidence = classify_differential(base, candidate, _contract())

    assert evidence.status is WorkflowStatus.NO_REGRESSION_OBSERVED
    assert evidence.stable is True


@pytest.mark.parametrize(
    ("base", "candidate", "message"),
    [
        ([], [_observation("candidate-sha", "secure")], "must be nonempty"),
        ([_observation("base-sha", "secure")], [], "must be nonempty"),
        (
            [_observation("base-sha", "secure")],
            [
                _observation("candidate-sha", "secure"),
                _observation("candidate-sha", "secure"),
            ],
            "equal positive repeat counts",
        ),
        (
            [
                _observation("base-sha", "secure"),
                _observation("other-base-sha", "secure"),
            ],
            [
                _observation("candidate-sha", "secure"),
                _observation("candidate-sha", "secure"),
            ],
            "base observations must share one revision",
        ),
        (
            [
                _observation("base-sha", "secure"),
                _observation("base-sha", "secure"),
            ],
            [
                _observation("candidate-sha", "secure"),
                _observation("other-candidate-sha", "secure"),
            ],
            "candidate observations must share one revision",
        ),
        (
            [_observation("same-sha", "secure")],
            [_observation("same-sha", "secure")],
            "must identify different revisions",
        ),
    ],
)
def test_invalid_repeat_sets_raise_explicit_value_errors(
    base: list[RuntimeObservation],
    candidate: list[RuntimeObservation],
    message: str,
) -> None:
    """Malformed caller inputs must not become indexing errors or evidence."""
    with pytest.raises(ValueError, match=message):
        classify_differential(base, candidate, _contract())


def test_classifier_rejects_unvalidated_observation_or_contract_inputs() -> None:
    """Loose dictionaries cannot cross the deterministic evidence boundary."""
    secure = _observation("base-sha", "secure")
    candidate = _observation("candidate-sha", "secure")

    with pytest.raises(TypeError, match="RuntimeObservation"):
        classify_differential(  # type: ignore[list-item]
            [secure.model_dump()], [candidate], _contract()
        )
    with pytest.raises(TypeError, match="RiskContract"):
        classify_differential(  # type: ignore[arg-type]
            [secure], [candidate], _contract().model_dump()
        )


def test_classifier_rejects_non_sequence_repeat_inputs_explicitly() -> None:
    """A non-iterable caller error must not leak an incidental indexing failure."""
    candidate = _observation("candidate-sha", "secure")

    with pytest.raises(TypeError, match="observation sequences"):
        classify_differential(  # type: ignore[arg-type]
            None, [candidate], _contract()
        )


def test_classifier_rejects_empty_revision_labels_explicitly() -> None:
    """Unattributable observations cannot support differential evidence."""
    base = [
        _observation("base", "secure").model_copy(update={"revision": ""})
    ]
    candidate = [_observation("candidate-sha", "secure")]

    with pytest.raises(ValueError, match="nonempty revision"):
        classify_differential(base, candidate, _contract())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("contract_id", "unrelated-contract"),
        ("actor", "nurse"),
        ("actor_privileges", ["View Patients", "Delete Patients"]),
        ("missing_privileges", ["Purge Patients"]),
        ("preconditions", ["a different patient exists"]),
        ("action", "the clerk reads the patient"),
        ("secure_expectation", "the deletion request succeeds"),
        ("observable_evidence", ["pytest output text"]),
        ("base_expectation", "vulnerable"),
        ("candidate_expectation", "secure"),
        ("cleanup", ["leave the patient behind"]),
    ],
)
def test_classifier_rejects_every_semantic_contract_mutation(
    field: str,
    replacement: str | list[str],
) -> None:
    """A same-id or related valid contract cannot borrow fixture evidence."""
    changed_contract = _contract().model_copy(update={field: replacement})
    base = [_observation("base", "secure")]
    candidate = [_observation("candidate", "vulnerable")]

    with pytest.raises(UnsupportedRiskContractError) as error:
        classify_differential(base, candidate, changed_contract)

    assert error.value.reason_code == "approved_contract_mismatch"
    assert str(error.value) == (
        "approved_contract_mismatch: contract does not match the approved "
        "patient-delete authorization fixture"
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("setup_succeeded", "true"),
        ("setup_succeeded", 1),
        ("action_attempted", "false"),
        ("action_attempted", 0),
        ("control_succeeded", "true"),
        ("control_succeeded", 1),
        ("request_status", "204"),
        ("request_status", False),
        ("resource_exists_after", "false"),
        ("resource_exists_after", 0),
        ("pytest_exit_code", "0"),
        ("pytest_exit_code", False),
        ("revision", 123),
        ("reason_code", 123),
    ],
)
def test_runtime_observation_rejects_coerced_security_facts(
    field: str,
    invalid_value: object,
) -> None:
    """Coerced booleans, integers, and provenance strings are not raw facts."""
    payload = _observation("base", "secure").model_dump(mode="json")
    payload[field] = invalid_value

    with pytest.raises(ValidationError):
        RuntimeObservation.model_validate(payload)


@pytest.mark.parametrize(
    "revision",
    [
        "",
        " base",
        "base ",
        "base\tcandidate",
        "base\ncandidate",
        "BASE",
        "-base",
        "base-",
        "base_candidate",
        "base/candidate",
        "base-" + ("x" * 124),
    ],
)
def test_runtime_observation_rejects_noncanonical_revision_labels(
    revision: str,
) -> None:
    """Whitespace, controls, and noncanonical spellings cannot alias revisions."""
    payload = _observation("base", "secure").model_dump(mode="json")
    payload["revision"] = revision

    with pytest.raises(ValidationError):
        RuntimeObservation.model_validate(payload)


@pytest.mark.parametrize(
    "revision",
    [
        "base",
        "candidate",
        "base-revision",
        "candidate-controlled-2",
        "0123456789abcdef0123456789abcdef01234567",
    ],
)
def test_runtime_observation_accepts_canonical_controlled_and_sha_labels(
    revision: str,
) -> None:
    """Controlled labels and lowercase immutable commit identifiers remain valid."""
    observation = _observation("base", "secure").model_copy(
        update={"revision": revision}
    )

    validated = RuntimeObservation.model_validate(
        observation.model_dump(mode="json")
    )

    assert validated.revision == revision
