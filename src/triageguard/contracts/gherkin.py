"""Deterministic Gherkin views of approved authorization contracts."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from triageguard.domain.models import RiskContract

_STEP_PATTERN = re.compile(r"^\s*(Given|When|Then|And|But|\*)\s+(.+?)\s*$")
_SCENARIO_PATTERN = re.compile(r"^\s*Scenario:\s*(.*?)\s*$")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class AlignmentReport:
    """Deterministic result of comparing required contract steps to Gherkin."""

    approved: bool
    reason_codes: list[str]
    matched_steps: dict[str, str]


@dataclass(frozen=True)
class _ParsedStep:
    keyword: str
    text: str
    phase: str | None
    before_when: bool


def render_gherkin(contract: RiskContract) -> str:
    """Render a fixed, locally inspectable Gherkin scenario for a contract."""
    denial_oracle, persistence_oracle = _security_oracles(contract)
    lines = [
        f"Feature: Authorization contract {_escape(contract.contract_id)}",
        "",
        f"  Scenario: {_escape(contract.contract_id)}",
    ]

    given_emitted = False
    for precondition in contract.preconditions:
        keyword = "And" if given_emitted else "Given"
        lines.append(f"    {keyword} {_escape(precondition)}")
        given_emitted = True

    lines.extend(
        [
            (
                f"    {'And' if given_emitted else 'Given'} the "
                f"{_escape(contract.actor)} has privileges: "
                f"{_render_values(contract.actor_privileges)}"
            ),
            (
                f"    And the {_escape(contract.actor)} lacks privileges: "
                f"{_render_values(contract.missing_privileges)}"
            ),
            f"    When {_escape(contract.action)}",
            f"    Then {_escape(denial_oracle)}",
            f"    And {_escape(persistence_oracle)}",
        ]
    )
    for evidence in contract.observable_evidence:
        lines.append(f"    And evidence is collected through {_escape(evidence)}")
    for cleanup in contract.cleanup:
        lines.append(f"    # Cleanup: {_escape(cleanup)}")
    return "\n".join(lines) + "\n"


def validate_gherkin_alignment(contract: RiskContract, text: str) -> AlignmentReport:
    """Require exact contract steps in their Gherkin semantic roles."""
    denial_oracle, persistence_oracle = _security_oracles(contract)
    steps = _parse_steps(text)
    expected_steps = _parse_steps(render_gherkin(contract))
    matched_steps: dict[str, str] = {}
    reason_codes: list[str] = []

    _validate_scenario_header(contract, text, matched_steps, reason_codes)

    for precondition in contract.preconditions:
        _record_exact_role(
            key="precondition",
            expected=precondition,
            steps=steps,
            valid_role=_is_precondition,
            matched_steps=matched_steps,
            reason_codes=reason_codes,
            capture_match=False,
        )

    action_step = _record_exact_role(
        key="action",
        expected=contract.action,
        steps=steps,
        valid_role=lambda step: step.keyword == "When",
        matched_steps=matched_steps,
        reason_codes=reason_codes,
    )
    _record_actor(
        actor=contract.actor,
        action=contract.action,
        steps=steps,
        action_step=action_step,
        matched_steps=matched_steps,
        reason_codes=reason_codes,
    )
    _record_exact_role(
        key="denial_oracle",
        expected=denial_oracle,
        steps=steps,
        valid_role=lambda step: step.keyword == "Then",
        matched_steps=matched_steps,
        reason_codes=reason_codes,
    )
    _record_exact_role(
        key="persistence_oracle",
        expected=persistence_oracle,
        steps=steps,
        valid_role=lambda step: step.keyword == "And" and step.phase == "then",
        matched_steps=matched_steps,
        reason_codes=reason_codes,
    )
    _validate_executable_sequence(expected_steps, steps, reason_codes)
    return AlignmentReport(
        approved=not reason_codes,
        reason_codes=reason_codes,
        matched_steps=matched_steps,
    )


def _validate_scenario_header(
    contract: RiskContract,
    text: str,
    matched_steps: dict[str, str],
    reason_codes: list[str],
) -> None:
    titles = [
        match.group(1)
        for line in text.splitlines()
        if (match := _SCENARIO_PATTERN.match(line)) is not None
    ]
    if not titles:
        _add_reason(reason_codes, "scenario_missing")
        return
    if len(titles) != 1:
        _add_reason(reason_codes, "scenario_duplicate")
        return
    if titles[0] != contract.contract_id:
        _add_reason(reason_codes, "scenario_title_changed")
        return
    matched_steps["scenario"] = titles[0]


def _security_oracles(contract: RiskContract) -> tuple[str, str]:
    """Split the approved expectation into its two required literal oracles."""
    clauses = [clause.strip() for clause in re.split(r"\s+and\s+", contract.secure_expectation, maxsplit=1)]
    if len(clauses) != 2 or not all(clauses):
        raise ValueError(
            "secure_expectation must contain denial and persistence clauses joined by 'and'"
        )
    return clauses[0], clauses[1]


def _parse_steps(text: str) -> list[_ParsedStep]:
    """Parse keywords while retaining the phase inherited by And steps."""
    steps: list[_ParsedStep] = []
    phase: str | None = None
    seen_when = False
    for line in text.splitlines():
        match = _STEP_PATTERN.match(line)
        if match is None:
            continue
        keyword = match.group(1)
        if keyword == "Given":
            phase = "given"
        elif keyword == "When":
            phase = "when"
            seen_when = True
        elif keyword == "Then":
            phase = "then"
        elif keyword == "*" and phase is None:
            phase = "unbound"
        steps.append(
            _ParsedStep(
                keyword=keyword,
                text=_normalize(match.group(2)),
                phase=phase,
                before_when=not seen_when,
            )
        )
    return steps


def _record_exact_role(
    *,
    key: str,
    expected: str,
    steps: list[_ParsedStep],
    valid_role: Callable[[_ParsedStep], bool],
    matched_steps: dict[str, str],
    reason_codes: list[str],
    capture_match: bool = True,
) -> _ParsedStep | None:
    normalized_expected = _normalize(expected)
    exact_matches = [step for step in steps if step.text == normalized_expected]
    role_matches = [step for step in exact_matches if valid_role(step)]

    if len(role_matches) == 1 and len(exact_matches) == 1:
        if capture_match:
            matched_steps[key] = role_matches[0].text
        return role_matches[0]
    if len(role_matches) > 1:
        _add_reason(reason_codes, f"duplicate_{key}")
    elif exact_matches:
        _add_reason(reason_codes, f"{key}_wrong_phase")
    elif _has_candidate_in_role(steps, valid_role):
        _add_reason(reason_codes, f"{key}_changed")
    else:
        _add_reason(reason_codes, f"{key}_missing")

    if role_matches and len(exact_matches) > len(role_matches):
        _add_reason(reason_codes, f"{key}_wrong_phase")
    return None


def _record_actor(
    *,
    actor: str,
    action: str,
    steps: list[_ParsedStep],
    action_step: _ParsedStep | None,
    matched_steps: dict[str, str],
    reason_codes: list[str],
) -> None:
    """Bind the actor to the actual When step when the action names the actor."""
    normalized_actor = _normalize(actor)
    actor_pattern = re.compile(rf"\b{re.escape(normalized_actor)}\b")
    normalized_action = _normalize(action)
    if actor_pattern.search(normalized_action):
        when_actor_steps = [
            step
            for step in steps
            if step.keyword == "When" and actor_pattern.search(step.text)
        ]
        if len(when_actor_steps) == 1 and action_step is not None:
            matched_steps["actor"] = when_actor_steps[0].text
            return
        if len(when_actor_steps) > 1:
            _add_reason(reason_codes, "duplicate_actor")
        else:
            _add_reason(reason_codes, "actor_changed")
        return

    setup_actor_steps = [
        step
        for step in steps
        if _is_precondition(step) and actor_pattern.search(step.text)
    ]
    if setup_actor_steps:
        matched_steps["actor"] = setup_actor_steps[0].text
    else:
        _add_reason(reason_codes, "actor_changed")


def _is_precondition(step: _ParsedStep) -> bool:
    return step.before_when and (
        step.keyword == "Given" or (step.keyword == "And" and step.phase == "given")
    )


def _has_candidate_in_role(
    steps: list[_ParsedStep], valid_role: Callable[[_ParsedStep], bool]
) -> bool:
    return any(valid_role(step) for step in steps)


def _add_reason(reason_codes: list[str], reason_code: str) -> None:
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)


def _validate_executable_sequence(
    expected_steps: list[_ParsedStep],
    observed_steps: list[_ParsedStep],
    reason_codes: list[str],
) -> None:
    """Reject every executable-step change outside the approved renderer output."""
    expected_signatures = [_step_signature(step) for step in expected_steps]
    observed_signatures = [_step_signature(step) for step in observed_steps]
    if observed_signatures == expected_signatures:
        return

    if len(observed_signatures) > len(expected_signatures):
        _add_reason(reason_codes, "executable_step_added")
        expected_counts = Counter(expected_signatures)
        observed_counts = Counter(observed_signatures)
        if any(
            observed_counts[signature] > expected_counts[signature]
            for signature in expected_counts
        ):
            _add_reason(reason_codes, "executable_step_repeated")
        return
    if len(observed_signatures) < len(expected_signatures):
        _add_reason(reason_codes, "executable_step_removed")
        return
    if Counter(observed_signatures) == Counter(expected_signatures):
        _add_reason(reason_codes, "executable_step_reordered")
        return
    if Counter(step.text for step in observed_steps) == Counter(
        step.text for step in expected_steps
    ):
        _add_reason(reason_codes, "executable_step_relocated")
        return
    _add_reason(reason_codes, "executable_step_changed")


def _step_signature(step: _ParsedStep) -> tuple[str, str, str | None, bool]:
    return step.keyword, step.text, step.phase, step.before_when


def _render_values(values: list[str]) -> str:
    return ", ".join(_escape(value) for value in values)


def _escape(value: str) -> str:
    """Keep interpolated text on one Gherkin line without altering its meaning."""
    return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def _normalize(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip().casefold()
