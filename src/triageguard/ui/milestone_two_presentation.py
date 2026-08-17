"""Plain-language presentation rules for the five-page Milestone 2 UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

STEP_LABELS = (
    "Choose a pull request",
    "Understand the change",
    "Review possible risks",
    "Choose and edit one risk",
    "Create and approve the scenario",
)

COMMIT_ID_EXPLANATION = (
    "A commit ID is the unique serial number for one saved code photograph. "
    "The interface labels its first 12 characters as an abbreviation to keep "
    "the table readable. Full commit IDs remain in Technical evidence."
)

ModelStage = Literal[
    "risk_hypothesis",
    "testability_assessment",
    "gherkin_generation",
]

_STAGE_LABELS: dict[ModelStage, str] = {
    "risk_hypothesis": "Risk proposal",
    "testability_assessment": "Scenario testability assessment",
    "gherkin_generation": "Scenario generation",
}

_OMISSION_REASONS = {
    "request_budget": (
        "Omitted because the model request reached its declared byte budget."
    ),
    "stage_irrelevant": (
        "Omitted because this frozen anchor was not relevant to this model stage."
    ),
    "superseded": (
        "Omitted because a later frozen anchor replaced it for this model stage."
    ),
}


@dataclass(frozen=True)
class ProgressStep:
    """One visible wizard step and its availability state."""

    label: str
    state: str


@dataclass(frozen=True)
class ComparisonCard:
    """Plain-language description of one comparison between saved snapshots."""

    title: str
    comparison: str
    explanation: str


def abbreviated_commit_id(commit_sha: str) -> str:
    """Label a display-only Git commit abbreviation unambiguously."""
    if len(commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise ValueError("commit_sha must be a lowercase 40-character Git SHA")
    return f"{commit_sha[:12]} (abbreviated)"


def evidence_coverage_text(visible_count: int, total_count: int) -> str:
    """Describe the exact fraction of frozen anchors visible to one model call."""
    if (
        isinstance(visible_count, bool)
        or isinstance(total_count, bool)
        or not isinstance(visible_count, int)
        or not isinstance(total_count, int)
        or visible_count < 0
        or total_count < 0
        or visible_count > total_count
    ):
        raise ValueError("evidence coverage counts must satisfy 0 <= visible <= total")
    return (
        f"{visible_count} of {total_count} frozen anchors visible to this model call."
    )


def omitted_evidence_reason(reason: str) -> str:
    """Translate one closed omission code without inventing an explanation."""
    try:
        return _OMISSION_REASONS[reason]
    except KeyError as error:
        raise ValueError("unknown model-evidence omission reason") from error


def model_stage_outcome_message(
    *,
    stage: ModelStage,
    reason_code: str,
    provider: str,
    request_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> str:
    """Return one stage-specific, secret-free explanation of a stopped model stage."""
    stage_label = _STAGE_LABELS[stage]
    if reason_code == "model_request_too_large":
        if request_bytes is None or limit_bytes is None:
            raise ValueError("request and policy bytes are required for a size stop")
        return (
            f"{stage_label} stopped before contacting {provider}: the exact request "
            f"was {request_bytes:,} bytes, above the declared {limit_bytes:,}-byte "
            "policy. No conclusion was produced."
        )
    if reason_code in {
        "insufficient_context_to_assess",
        "insufficient_frozen_evidence",
        "analysis_limit_exceeded",
    }:
        return (
            f"{stage_label} stopped because the bounded frozen evidence was "
            "insufficient. This does not mean the pull request is safe."
        )
    if reason_code in {
        "model_output_invalid",
        "groq_invalid_output",
        "assessment_validation_failed",
        "candidate_validation_failed",
    }:
        return (
            f"{stage_label} returned content, but it did not pass local validation. "
            "No conclusion was produced."
        )
    return (
        f"{stage_label} stopped because {provider} rejected the request at the "
        "provider boundary. No conclusion was produced."
    )


def comparison_cards() -> tuple[ComparisonCard, ...]:
    """Return the three fixed comparisons without exposing raw Git commands."""
    return (
        ComparisonCard(
            title="Author change",
            comparison="M → H",
            explanation=(
                "M is the shared starting point and H is the pull-request branch. "
                "This shows exactly what the author changed."
            ),
        ),
        ComparisonCard(
            title="Merge impact",
            comparison="B → C",
            explanation=(
                "B is current main and C is the merge preview. This shows what "
                "merging the pull request would change now."
            ),
        ),
        ComparisonCard(
            title="Main-branch drift",
            comparison="M → B",
            explanation=(
                "M is the shared starting point and B is current main. This shows "
                "what changed in main while the pull request was waiting."
            ),
        ),
    )


def guided_progress(current_page: int) -> tuple[ProgressStep, ...]:
    """Describe page progress without allowing the UI to skip a step."""
    if isinstance(current_page, bool) or not isinstance(current_page, int):
        raise TypeError("current_page must be an integer")
    if not 1 <= current_page <= len(STEP_LABELS):
        raise ValueError("current_page must identify one of the five pages")

    states = tuple(
        "complete"
        if number < current_page
        else "current"
        if number == current_page
        else "locked"
        for number in range(1, len(STEP_LABELS) + 1)
    )
    return tuple(
        ProgressStep(label=label, state=state)
        for label, state in zip(STEP_LABELS, states, strict=True)
    )


def freshness_label(status: str) -> str:
    """Translate a freshness result into the short label shown to users."""
    labels = {
        "current": "Current",
        "stale": "Stale",
        "unknown": "Unable to recheck",
    }
    return labels.get(status, "Unable to recheck")
