"""Fixed-rule differential evidence classification without model inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from triageguard.domain.models import (
    DifferentialEvidence,
    RiskContract,
    RuntimeObservation,
)
from triageguard.domain.statuses import WorkflowStatus
from triageguard.provenance import canonical_sha256

_Behavior = Literal["secure", "vulnerable"]
_SecurityTuple = tuple[
    bool,
    bool,
    int | None,
    bool | None,
    bool | None,
    int | None,
    bool | None,
    bool | None,
]

_MATRIX: dict[
    tuple[_Behavior, _Behavior],
    tuple[WorkflowStatus, str, str],
] = {
    ("secure", "vulnerable"): (
        WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
        "candidate_regression_observed",
        (
            "The base denied unauthorized deletion and preserved the patient, "
            "while the candidate allowed deletion and removed the patient."
        ),
    ),
    ("vulnerable", "secure"): (
        WorkflowStatus.CANDIDATE_FIX_OBSERVED,
        "candidate_fix_observed",
        (
            "The base allowed unauthorized deletion and removed the patient, "
            "while the candidate denied deletion and preserved the patient."
        ),
    ),
    ("secure", "secure"): (
        WorkflowStatus.NO_REGRESSION_OBSERVED,
        "no_regression_observed",
        "Both revisions denied unauthorized deletion and preserved the patient.",
    ),
    ("vulnerable", "vulnerable"): (
        WorkflowStatus.PRE_EXISTING_RISK_OBSERVED,
        "pre_existing_risk_observed",
        "Both revisions allowed unauthorized deletion and removed the patient.",
    ),
}

_INCONCLUSIVE_TEMPLATES: dict[str, str] = {
    "base_setup_failed": (
        "At least one base run did not complete setup; differential evidence is "
        "inconclusive."
    ),
    "candidate_setup_failed": (
        "At least one candidate run did not complete setup; differential evidence "
        "is inconclusive."
    ),
    "base_action_not_attempted": (
        "At least one base run did not attempt the approved action; differential "
        "evidence is inconclusive."
    ),
    "candidate_action_not_attempted": (
        "At least one candidate run did not attempt the approved action; "
        "differential evidence is inconclusive."
    ),
    "base_control_failed_or_missing": (
        "At least one base run lacks a successful authorized control; differential "
        "evidence is inconclusive."
    ),
    "candidate_control_failed_or_missing": (
        "At least one candidate run lacks a successful authorized control; "
        "differential evidence is inconclusive."
    ),
    "base_raw_facts_missing_or_unsupported": (
        "At least one base run has missing or unsupported HTTP/state facts; "
        "differential evidence is inconclusive."
    ),
    "candidate_raw_facts_missing_or_unsupported": (
        "At least one candidate run has missing or unsupported HTTP/state facts; "
        "differential evidence is inconclusive."
    ),
}

_UNSTABLE_REASON = "security_relevant_tuple_unstable"
_UNSTABLE_EXPLANATION = (
    "Repeated security-relevant facts differed from run 1; one-based differing "
    "run indexes are reported separately for base and candidate."
)

_APPROVED_PATIENT_DELETE_CONTRACT_SHA256 = (
    "afb741efbf3fdfd5cc1ca2a1fcfe5e7f5ea72efd94f3831289af490e020c6441"
)


class UnsupportedRiskContractError(ValueError):
    """The classifier received a contract outside its fixture-specific scope."""

    reason_code = "approved_contract_mismatch"

    def __init__(self) -> None:
        super().__init__(
            f"{self.reason_code}: contract does not match the approved "
            "patient-delete authorization fixture"
        )


def classify_differential(
    base_observations: Sequence[RuntimeObservation],
    candidate_observations: Sequence[RuntimeObservation],
    contract: RiskContract,
) -> DifferentialEvidence:
    """Classify repeated raw facts using a fixed, non-LLM decision matrix.

    Differing run indexes are one-based and identify observations whose
    security-relevant tuple differs from the deterministic run-1 representative.
    """
    base, candidate = _validate_inputs(
        base_observations, candidate_observations, contract
    )
    repetitions = len(base)
    base_representative = base[0]
    candidate_representative = candidate[0]

    problem = _first_execution_problem(base, candidate)
    if problem is not None:
        return DifferentialEvidence(
            base=base_representative,
            candidate=candidate_representative,
            base_revision=base_representative.revision,
            candidate_revision=candidate_representative.revision,
            repetitions=repetitions,
            stable=False,
            status=WorkflowStatus.EXECUTION_INCONCLUSIVE,
            reason_code=problem,
            explanation=_INCONCLUSIVE_TEMPLATES[problem],
            base_differing_run_indexes=[],
            candidate_differing_run_indexes=[],
        )

    base_differences = _differing_run_indexes(base)
    candidate_differences = _differing_run_indexes(candidate)
    if base_differences or candidate_differences:
        return DifferentialEvidence(
            base=base_representative,
            candidate=candidate_representative,
            base_revision=base_representative.revision,
            candidate_revision=candidate_representative.revision,
            repetitions=repetitions,
            stable=False,
            status=WorkflowStatus.UNSTABLE_RESULT,
            reason_code=_UNSTABLE_REASON,
            explanation=_UNSTABLE_EXPLANATION,
            base_differing_run_indexes=base_differences,
            candidate_differing_run_indexes=candidate_differences,
        )

    base_behavior = _supported_behavior(base_representative)
    candidate_behavior = _supported_behavior(candidate_representative)
    if base_behavior is None or candidate_behavior is None:
        raise AssertionError("validated supported facts must have a behavior")
    status, reason_code, explanation = _MATRIX[
        (base_behavior, candidate_behavior)
    ]
    return DifferentialEvidence(
        base=base_representative,
        candidate=candidate_representative,
        base_revision=base_representative.revision,
        candidate_revision=candidate_representative.revision,
        repetitions=repetitions,
        stable=True,
        status=status,
        reason_code=reason_code,
        explanation=explanation,
        base_differing_run_indexes=[],
        candidate_differing_run_indexes=[],
    )


def _validate_inputs(
    base_observations: Sequence[RuntimeObservation],
    candidate_observations: Sequence[RuntimeObservation],
    contract: RiskContract,
) -> tuple[list[RuntimeObservation], list[RuntimeObservation]]:
    if not isinstance(contract, RiskContract):
        raise TypeError("contract must be a validated RiskContract")
    if (
        canonical_sha256(contract.model_dump(mode="json"))
        != _APPROVED_PATIENT_DELETE_CONTRACT_SHA256
    ):
        raise UnsupportedRiskContractError()
    if not isinstance(base_observations, Sequence) or not isinstance(
        candidate_observations, Sequence
    ):
        raise TypeError("base and candidate observations must be observation sequences")
    base = list(base_observations)
    candidate = list(candidate_observations)
    if not base or not candidate:
        raise ValueError("base and candidate observation lists must be nonempty")
    if len(base) != len(candidate):
        raise ValueError("base and candidate require equal positive repeat counts")
    for observation in (*base, *candidate):
        if not isinstance(observation, RuntimeObservation):
            raise TypeError("all observations must be validated RuntimeObservation values")
        if not observation.revision.strip():
            raise ValueError("all observations must have a nonempty revision")
    if len({observation.revision for observation in base}) != 1:
        raise ValueError("base observations must share one revision")
    if len({observation.revision for observation in candidate}) != 1:
        raise ValueError("candidate observations must share one revision")
    if base[0].revision == candidate[0].revision:
        raise ValueError("base and candidate must identify different revisions")
    return base, candidate


def _first_execution_problem(
    base: list[RuntimeObservation],
    candidate: list[RuntimeObservation],
) -> str | None:
    groups = (("base", base), ("candidate", candidate))
    for side, observations in groups:
        if any(not observation.setup_succeeded for observation in observations):
            return f"{side}_setup_failed"
    for side, observations in groups:
        if any(not observation.action_attempted for observation in observations):
            return f"{side}_action_not_attempted"
    for side, observations in groups:
        if any(
            not _has_exact_control_tuple(observation)
            for observation in observations
        ):
            return f"{side}_control_failed_or_missing"
    for side, observations in groups:
        if any(_supported_behavior(observation) is None for observation in observations):
            return f"{side}_raw_facts_missing_or_unsupported"
    return None


def _supported_behavior(observation: RuntimeObservation) -> _Behavior | None:
    if (
        not observation.setup_succeeded
        or not observation.action_attempted
        or not _has_exact_control_tuple(observation)
    ):
        return None
    if (
        observation.request_status == 403
        and observation.resource_exists_after is True
    ):
        return "secure"
    if (
        observation.request_status is not None
        and 200 <= observation.request_status < 300
        and observation.resource_exists_after is False
    ):
        return "vulnerable"
    return None


def _has_exact_control_tuple(observation: RuntimeObservation) -> bool:
    return (
        observation.control_succeeded is True
        and observation.control_resource_exists_before is True
        and observation.control_request_status == 204
        and observation.control_resource_exists_after is False
    )


def _differing_run_indexes(
    observations: list[RuntimeObservation],
) -> list[int]:
    representative = _security_relevant_tuple(observations[0])
    return [
        index
        for index, observation in enumerate(observations[1:], start=2)
        if _security_relevant_tuple(observation) != representative
    ]


def _security_relevant_tuple(
    observation: RuntimeObservation,
) -> _SecurityTuple:
    return (
        observation.setup_succeeded,
        observation.action_attempted,
        observation.request_status,
        observation.resource_exists_after,
        observation.control_succeeded,
        observation.control_request_status,
        observation.control_resource_exists_before,
        observation.control_resource_exists_after,
    )
