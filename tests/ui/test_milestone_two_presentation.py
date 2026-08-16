"""Tests for plain-language Milestone 2 UI presentation rules."""

from triageguard.ui.milestone_two_presentation import (
    COMMIT_ID_EXPLANATION,
    STEP_LABELS,
    comparison_cards,
    freshness_label,
    guided_progress,
)


def test_five_page_progress_uses_plain_language_and_locks_later_pages() -> None:
    """The UI explains the process without exposing implementation details."""
    assert STEP_LABELS == (
        "Choose a pull request",
        "Understand the change",
        "Review possible risks",
        "Choose and edit one risk",
        "Create and approve the scenario",
    )

    progress = guided_progress(current_page=3)

    assert [step.state for step in progress] == [
        "complete",
        "complete",
        "current",
        "locked",
        "locked",
    ]
    assert freshness_label("current") == "Current"
    assert freshness_label("stale") == "Stale"
    assert freshness_label("unknown") == "Unable to recheck"


def test_comparison_cards_explain_each_frozen_code_difference_separately() -> None:
    """The UI must explain what each of the three comparisons actually means."""
    cards = comparison_cards()

    assert [card.title for card in cards] == [
        "Author change",
        "Merge impact",
        "Main-branch drift",
    ]
    assert [card.comparison for card in cards] == ["M → H", "B → C", "M → B"]
    assert "unique serial number" in COMMIT_ID_EXPLANATION
