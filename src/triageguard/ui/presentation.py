"""Pure plain-language presentation rules for the guided prototype UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressStep:
    """One read-only guided-flow step and its availability state."""

    label: str
    state: str


@dataclass(frozen=True)
class ResultMessage:
    """Plain-language terminal copy separated from raw research evidence."""

    level: str
    title: str
    body: str


_STEP_LABELS = (
    "Demonstration loaded",
    "Risk reviewed",
    "Test ready",
    "Comparison complete",
)

_CVSS_SOURCE_LABELS = {
    "contract": "Risk contract",
    "runtime_design": "Runtime test design",
    "deployment_assumption": "Deployment assumption",
    "expert_judgment": "Expert judgment",
    "standard_interpretation": "CVSS standard interpretation",
}

_NOT_SCORED_REASONS = {
    "tested_vulnerability_not_observed": (
        "Tested vulnerability not observed in this version."
    ),
    "insufficient_evidence_for_severity": (
        "Evidence was insufficient or unstable, so no severity was calculated."
    ),
}

_RESULT_MESSAGES = {
    "candidate_regression_observed": ResultMessage(
        level="error",
        title="Potential security regression detected",
        body=(
            "The candidate allowed a clerk without the required permission "
            "to delete the patient."
        ),
    ),
    "candidate_fix_observed": ResultMessage(
        level="success",
        title="Candidate security improvement observed",
        body="The candidate blocked behavior that was vulnerable in the base.",
    ),
    "no_regression_observed": ResultMessage(
        level="success",
        title="No tested regression observed",
        body="Both versions preserved the tested authorization boundary.",
    ),
    "pre_existing_risk_observed": ResultMessage(
        level="error",
        title="Risk observed in both versions",
        body=(
            "The tested authorization weakness was present in both base and "
            "candidate."
        ),
    ),
    "unstable_result": ResultMessage(
        level="warning",
        title="Results were not repeatable",
        body=(
            "No vulnerability conclusion was reached because repeated "
            "observations differed."
        ),
    ),
    "execution_inconclusive": ResultMessage(
        level="warning",
        title="Comparison was inconclusive",
        body=(
            "No vulnerability conclusion was reached because execution "
            "evidence was incomplete."
        ),
    ),
    "generation_abstained": ResultMessage(
        level="warning",
        title="Test generation stopped",
        body=(
            "No vulnerability conclusion was reached because no validated "
            "test was produced."
        ),
    ),
    "validation_failed": ResultMessage(
        level="warning",
        title="Generated test was rejected",
        body=(
            "No vulnerability conclusion was reached because deterministic "
            "validation failed."
        ),
    ),
}


def guided_progress(
    *,
    current_page: int,
    risk_accepted: bool,
    test_ready: bool,
    terminal: bool,
) -> tuple[ProgressStep, ...]:
    """Derive progress without allowing a presentation-only gate to be skipped."""
    if not 1 <= current_page <= len(_STEP_LABELS):
        raise ValueError("current_page must identify one guided step")
    if current_page >= 3 and not risk_accepted:
        raise ValueError("risk selection is required for the requested page")
    if current_page == 4 and not test_ready:
        raise ValueError("validated test is required for the requested page")
    if terminal and current_page == 4:
        states = ("complete", "complete", "complete", "complete")
    else:
        states = tuple(
            "complete"
            if index < current_page
            else "current"
            if index == current_page
            else "locked"
            for index in range(1, len(_STEP_LABELS) + 1)
        )
    return tuple(
        ProgressStep(label=label, state=state)
        for label, state in zip(_STEP_LABELS, states, strict=True)
    )


def observation_row(
    label: str, observation: Mapping[str, object]
) -> dict[str, str]:
    """Translate one exact raw observation tuple without adding an inference."""
    request_status = observation.get("request_status")
    resource_exists = observation.get("resource_exists_after")
    control_succeeded = observation.get("control_succeeded")

    if request_status == 403:
        request = "Denied (HTTP 403)"
    elif request_status == 204:
        request = "Allowed (HTTP 204)"
    elif isinstance(request_status, int) and not isinstance(request_status, bool):
        request = f"HTTP {request_status}"
    else:
        request = "Unknown"

    if resource_exists is True:
        patient_state = "Still exists"
    elif resource_exists is False:
        patient_state = "Deleted"
    else:
        patient_state = "Unknown"

    if control_succeeded is True:
        control = "Passed"
    elif control_succeeded is False:
        control = "Failed"
    else:
        control = "Unknown"

    if (
        request_status == 403
        and resource_exists is True
        and control_succeeded is True
    ):
        meaning = "Protected"
    elif (
        request_status == 204
        and resource_exists is False
        and control_succeeded is True
    ):
        meaning = "Unauthorized deletion observed"
    else:
        meaning = "Observed but not classifiable"

    return {
        "Version": label,
        "Delete request": request,
        "Patient afterward": patient_state,
        "Authorized control": control,
        "Meaning": meaning,
    }


def severity_card_data(
    label: str,
    assessment: Mapping[str, object],
) -> dict[str, str]:
    """Format a persisted assessment without calculating or inferring severity."""
    status = assessment.get("status")
    if status == "not_scored":
        reason_code = assessment.get("reason_code")
        reason = _NOT_SCORED_REASONS.get(
            str(reason_code),
            "No defensible severity score is available for this version.",
        )
        return {
            "version": label,
            "headline": "Not scored",
            "label": "CVSS 4.0 not calculated",
            "reason": reason,
            "vector": "",
        }
    score = assessment.get("score")
    severity = assessment.get("severity")
    vector = assessment.get("vector")
    if (
        status != "provisional"
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isinstance(severity, str)
        or not severity
        or not isinstance(vector, str)
        or not vector
    ):
        raise ValueError("severity assessment is incomplete")
    return {
        "version": label,
        "headline": f"{float(score):.1f} {severity}",
        "label": "Provisional CVSS 4.0",
        "reason": "The tested vulnerability was observed in this version.",
        "vector": vector,
    }


def cvss_source_label(source_category: str) -> str:
    """Translate a persisted provenance category without hiding its meaning."""
    return _CVSS_SOURCE_LABELS.get(source_category, source_category.replace("_", " ").title())


def result_message(status: str) -> ResultMessage:
    """Return explicit terminal copy, defaulting to a non-claiming message."""
    return _RESULT_MESSAGES.get(
        status,
        ResultMessage(
            level="info",
            title="Run completed without a classified conclusion",
            body=(
                "No vulnerability conclusion was inferred from this terminal "
                "result. Review the research evidence for details."
            ),
        ),
    )
