from triageguard.ui.presentation import (
    cvss_source_label,
    guided_progress,
    observation_row,
    result_message,
    severity_card_data,
)


def test_guided_progress_advances_without_skipping_security_gates() -> None:
    """Removing a gate must not make a later step look available."""
    initial = guided_progress(
        current_page=1,
        risk_accepted=False,
        test_ready=False,
        terminal=False,
    )
    selected = guided_progress(
        current_page=2,
        risk_accepted=True,
        test_ready=False,
        terminal=False,
    )
    ready = guided_progress(
        current_page=3,
        risk_accepted=True,
        test_ready=True,
        terminal=False,
    )
    comparison = guided_progress(
        current_page=4,
        risk_accepted=True,
        test_ready=True,
        terminal=False,
    )
    complete = guided_progress(
        current_page=4,
        risk_accepted=True,
        test_ready=True,
        terminal=True,
    )
    revisited = guided_progress(
        current_page=3,
        risk_accepted=True,
        test_ready=True,
        terminal=True,
    )

    assert [step.state for step in initial] == [
        "current",
        "locked",
        "locked",
        "locked",
    ]
    assert [step.state for step in selected] == [
        "complete",
        "current",
        "locked",
        "locked",
    ]
    assert [step.state for step in ready] == [
        "complete",
        "complete",
        "current",
        "locked",
    ]
    assert [step.state for step in comparison] == [
        "complete",
        "complete",
        "complete",
        "current",
    ]
    assert [step.state for step in complete] == [
        "complete",
        "complete",
        "complete",
        "complete",
    ]
    assert [step.state for step in revisited] == [
        "complete",
        "complete",
        "current",
        "locked",
    ]


def test_observation_row_translates_security_facts_for_non_programmers() -> None:
    """Raw status/state tuples must not be the only visible explanation."""
    assert observation_row(
        "Secure base",
        {
            "request_status": 403,
            "resource_exists_after": True,
            "control_succeeded": True,
        },
    ) == {
        "Version": "Secure base",
        "Delete request": "Denied (HTTP 403)",
        "Patient afterward": "Still exists",
        "Authorized control": "Passed",
        "Meaning": "Protected",
    }
    assert observation_row(
        "Candidate",
        {
            "request_status": 204,
            "resource_exists_after": False,
            "control_succeeded": True,
        },
    )["Meaning"] == "Unauthorized deletion observed"


def test_observation_row_does_not_guess_from_incomplete_or_novel_facts() -> None:
    """Unknown evidence tuples must remain unknown instead of looking safe."""
    incomplete = observation_row(
        "Candidate",
        {
            "request_status": None,
            "resource_exists_after": None,
            "control_succeeded": None,
        },
    )
    novel = observation_row(
        "Candidate",
        {
            "request_status": 401,
            "resource_exists_after": True,
            "control_succeeded": True,
        },
    )

    assert incomplete == {
        "Version": "Candidate",
        "Delete request": "Unknown",
        "Patient afterward": "Unknown",
        "Authorized control": "Unknown",
        "Meaning": "Observed but not classifiable",
    }
    assert novel["Meaning"] == "Observed but not classifiable"


def test_result_message_never_turns_inconclusive_evidence_into_a_finding() -> None:
    """A missing-evidence status must never reuse vulnerability language."""
    regression = result_message("candidate_regression_observed")
    inconclusive = result_message("execution_inconclusive")
    unknown = result_message("new_future_terminal_state")

    assert regression.level == "error"
    assert regression.title == "Potential security regression detected"
    assert "without the required permission" in regression.body
    assert inconclusive.level == "warning"
    assert "No vulnerability conclusion" in inconclusive.body
    assert unknown.level == "info"
    assert "No vulnerability conclusion" in unknown.body


def test_severity_cards_distinguish_provisional_score_from_not_scored() -> None:
    """A secure side must never be formatted as zero severity."""
    base = severity_card_data(
        "Secure base",
        {
            "status": "not_scored",
            "reason_code": "tested_vulnerability_not_observed",
            "score": None,
            "severity": None,
            "vector": None,
        },
    )
    candidate = severity_card_data(
        "Candidate",
        {
            "status": "provisional",
            "reason_code": "tested_vulnerability_observed",
            "score": 7.1,
            "severity": "High",
            "vector": (
                "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:N/VI:H/VA:N/"
                "SC:N/SI:N/SA:N"
            ),
        },
    )

    assert base == {
        "version": "Secure base",
        "headline": "Not scored",
        "label": "CVSS 4.0 not calculated",
        "reason": "Tested vulnerability not observed in this version.",
        "vector": "",
    }
    assert candidate["headline"] == "7.1 High"
    assert candidate["label"] == "Provisional CVSS 4.0"
    assert candidate["vector"].startswith("CVSS:4.0/")
    assert "0.0" not in str(base)


def test_cvss_source_labels_expose_assumptions_and_judgment() -> None:
    """Contextual values must not look like facts measured by pytest."""
    assert cvss_source_label("deployment_assumption") == "Deployment assumption"
    assert cvss_source_label("expert_judgment") == "Expert judgment"
    assert cvss_source_label("runtime_design") == "Runtime test design"


def test_unscored_severity_cards_never_invent_zero_or_a_numeric_delta() -> None:
    """Secure, unstable, and inconclusive sides all remain explicitly unscored."""
    assessments = (
        {
            "status": "not_scored",
            "reason_code": "tested_vulnerability_not_observed",
        },
        {
            "status": "not_scored",
            "reason_code": "insufficient_evidence_for_severity",
        },
        {
            "status": "not_scored",
            "reason_code": "insufficient_evidence_for_severity",
        },
    )

    cards = [severity_card_data("Compared version", item) for item in assessments]

    assert all(card["headline"] == "Not scored" for card in cards)
    assert all(card["vector"] == "" for card in cards)
    assert "0.0" not in str(cards)
    assert "delta" not in str(cards).lower()
