"""Tests for plain-language Milestone 2 UI presentation rules."""

from triageguard.ui.milestone_two_presentation import (
    COMMIT_ID_EXPLANATION,
    STEP_LABELS,
    abbreviated_commit_id,
    comparison_cards,
    evidence_coverage_text,
    freshness_label,
    guided_progress,
    model_stage_outcome_message,
    omitted_evidence_reason,
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
    assert "M is the shared starting point" in cards[0].explanation
    assert "H is the pull-request branch" in cards[0].explanation
    assert "B is current main" in cards[1].explanation
    assert "C is the merge preview" in cards[1].explanation
    assert "M is the shared starting point" in cards[2].explanation
    assert "B is current main" in cards[2].explanation
    assert "unique serial number" in COMMIT_ID_EXPLANATION


def test_commit_abbreviation_and_evidence_coverage_are_explicit() -> None:
    """Short hashes and bounded model visibility must not look complete."""
    commit_sha = "1a128316857397d7df2a7d993910df79e6eeeba0"

    assert abbreviated_commit_id(commit_sha) == "1a1283168573 (abbreviated)"
    assert evidence_coverage_text(2, 5) == (
        "2 of 5 frozen anchors visible to this model call."
    )


def test_omitted_evidence_reasons_use_plain_language() -> None:
    """Every withheld anchor needs an explicit bounded-selection explanation."""
    assert omitted_evidence_reason("request_budget") == (
        "Omitted because the model request reached its declared byte budget."
    )
    assert omitted_evidence_reason("stage_irrelevant") == (
        "Omitted because this frozen anchor was not relevant to this model stage."
    )
    assert omitted_evidence_reason("superseded") == (
        "Omitted because a later frozen anchor replaced it for this model stage."
    )


def test_stage_outcome_messages_distinguish_policy_provider_and_validation() -> None:
    """Safe messages must identify the failing gate and deny a conclusion."""
    assert model_stage_outcome_message(
        stage="risk_hypothesis",
        reason_code="model_request_too_large",
        provider="groq",
        request_bytes=15_488,
        limit_bytes=12_000,
    ) == (
        "Risk proposal stopped before contacting groq: the exact request was "
        "15,488 bytes, above the declared 12,000-byte policy. No conclusion "
        "was produced."
    )
    assert "groq rejected the request at the provider boundary" in (
        model_stage_outcome_message(
            stage="testability_assessment",
            reason_code="groq_non_retryable_error",
            provider="groq",
        )
    )
    assert "did not pass local validation" in model_stage_outcome_message(
        stage="gherkin_generation",
        reason_code="model_output_invalid",
        provider="groq",
    )
    assert model_stage_outcome_message(
        stage="testability_assessment",
        reason_code="insufficient_frozen_evidence",
        provider="groq",
    ).endswith("This does not mean the pull request is safe.")
