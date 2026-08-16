"""Plain-language presentation rules for the five-page Milestone 2 UI."""

from __future__ import annotations

from dataclasses import dataclass

STEP_LABELS = (
    "Choose a pull request",
    "Understand the change",
    "Review possible risks",
    "Choose and edit one risk",
    "Create and approve the scenario",
)

COMMIT_ID_EXPLANATION = (
    "A commit ID is the unique serial number for one saved code photograph. "
    "The interface shows its first 12 characters to keep the table readable."
)


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


def comparison_cards() -> tuple[ComparisonCard, ...]:
    """Return the three fixed comparisons without exposing raw Git commands."""
    return (
        ComparisonCard(
            title="Author change",
            comparison="M → H",
            explanation=(
                "Shows what the pull-request author changed from the shared "
                "starting point."
            ),
        ),
        ComparisonCard(
            title="Merge impact",
            comparison="B → C",
            explanation=(
                "Shows what merging this pull request would change in the "
                "current main branch."
            ),
        ),
        ComparisonCard(
            title="Main-branch drift",
            comparison="M → B",
            explanation=(
                "Shows what changed in main while the pull request was waiting."
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
