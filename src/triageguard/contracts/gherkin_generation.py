"""Build structured Gherkin-generation requests from human-reviewed risks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from triageguard.domain.pr_analysis import (
    GherkinApproval,
    GherkinCandidate,
    GherkinCandidateDraft,
    GherkinStep,
    HumanReviewedRisk,
)
from triageguard.llm.gateway import (
    ModelOutputInvalid,
    ModelRequest,
    ModelResponse,
    StructuredModelGateway,
)
from triageguard.provenance import canonical_sha256

GHERKIN_SYSTEM_PROMPT = (
    "Convert the approved unconfirmed OpenMRS security-risk hypothesis into one "
    "observable Gherkin scenario. Preserve the approved actor, preconditions, "
    "action, expected secure behavior, possible failure oracle, observables, "
    "evidence terms, risk hash, and snapshot key. Return no Python, "
    "implementation code, CVSS score, or claim that a vulnerability exists."
)


def _strict_schema(value: object) -> object:
    """Require every declared field and forbid extras in every schema object."""
    if isinstance(value, dict):
        strict_value = {key: _strict_schema(item) for key, item in value.items()}
        properties = strict_value.get("properties")
        if strict_value.get("type") == "object" and isinstance(properties, dict):
            strict_value["additionalProperties"] = False
            strict_value["required"] = sorted(properties)
        return strict_value
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    return value


_raw_output_schema = _strict_schema(GherkinCandidateDraft.model_json_schema())
if not isinstance(_raw_output_schema, dict):
    raise TypeError("Gherkin output schema must be a JSON object")

GHERKIN_OUTPUT_SCHEMA: dict[str, Any] = _raw_output_schema


def build_gherkin_request(human_review: HumanReviewedRisk) -> ModelRequest:
    """Build one strict request bound to a human-approved risk successor."""
    try:
        human_review = HumanReviewedRisk.model_validate(
            human_review.model_dump(mode="json")
        )
    except ValidationError as error:
        raise ValueError("human review failed immutable-content validation") from error

    approved_risk = human_review.reviewed_risk.model_dump(mode="json")

    return ModelRequest(
        purpose="gherkin_generation",
        system_prompt=GHERKIN_SYSTEM_PROMPT,
        payload={
            "snapshot_key": human_review.snapshot_key,
            "reviewed_risk_sha256": human_review.reviewed_content_sha256,
            "approved_risk": approved_risk,
            "output_rules": {
                "scenario_count": 1,
                "feature_count": 1,
                "citation_rule": (
                    "Use only the supplied approved-risk terms and identifiers."
                ),
                "prohibited_content": [
                    "Python or implementation code",
                    "CVSS scores",
                    "claims that a vulnerability exists",
                ],
            },
        },
        output_schema=GHERKIN_OUTPUT_SCHEMA,
        max_output_tokens=2048,
    )


def generate_gherkin(
    *,
    human_review: HumanReviewedRisk,
    gateway: StructuredModelGateway,
) -> tuple[GherkinCandidate, ModelResponse]:
    """Generate and locally validate one Gherkin candidate."""
    request = build_gherkin_request(human_review)
    response = gateway.generate(request)

    try:
        draft = GherkinCandidateDraft.model_validate(response.data)
    except ValidationError as error:
        raise ModelOutputInvalid(
            "model response does not form a coherent Gherkin candidate draft"
        ) from error

    if draft.snapshot_key != human_review.snapshot_key:
        raise ModelOutputInvalid(
            "model response snapshot key does not match the human review"
        )
    if draft.reviewed_risk_sha256 != human_review.reviewed_content_sha256:
        raise ModelOutputInvalid(
            "model response reviewed-risk hash does not match the human review"
        )
    if draft.approved_risk != human_review.reviewed_risk:
        raise ModelOutputInvalid(
            "model response approved risk does not match the human review"
        )

    return GherkinCandidate.from_draft(draft), response


_STEP_PATTERN = re.compile(r"^(Given|When|Then|And)\s+(.+)$")


class GherkinGenerationError(ValueError):
    """A generated or edited scenario failed local Gherkin safety checks."""


@dataclass(frozen=True)
class GherkinValidationReport:
    """The local decision about whether a candidate still matches its review."""

    approved: bool
    reason_codes: tuple[str, ...]


def _add_reason(reason_codes: list[str], reason_code: str) -> None:
    """Record each validation reason only once in deterministic order."""
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)


def _required_risk_terms(human_review: HumanReviewedRisk) -> tuple[str, ...]:
    """Return terms whose removal would erase the approved failure oracle."""
    risk = human_review.reviewed_risk
    return (
        risk.expected_secure_behavior,
        risk.possible_failure,
        *risk.observables,
        *risk.code_identifiers,
    )


def _parse_editable_text(
    *,
    text: str,
    feature_title: str,
    scenario_title: str,
) -> tuple[tuple[str, str], ...]:
    """Parse exactly one simple Feature/Scenario without executing anything."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if sum(line.startswith("Feature:") for line in lines) != 1:
        raise GherkinGenerationError("gherkin_feature_count_invalid")
    if sum(line.startswith("Scenario:") for line in lines) != 1:
        raise GherkinGenerationError("gherkin_scenario_count_invalid")
    if (
        len(lines) < 3
        or lines[0] != f"Feature: {feature_title}"
        or lines[1] != f"Scenario: {scenario_title}"
    ):
        raise GherkinGenerationError("gherkin_title_mismatch")

    parsed_steps: list[tuple[str, str]] = []
    for line in lines[2:]:
        match = _STEP_PATTERN.fullmatch(line)
        if match is None:
            raise GherkinGenerationError("gherkin_text_contains_non_step_content")
        parsed_steps.append((match.group(1), match.group(2)))

    if not parsed_steps:
        raise GherkinGenerationError("gherkin_steps_missing")
    return tuple(parsed_steps)


def _validate_candidate_identity(
    candidate: GherkinCandidate,
    human_review: HumanReviewedRisk,
    reason_codes: list[str],
) -> None:
    """Check that the scenario belongs to this exact reviewed risk."""
    try:
        GherkinCandidate.from_persisted(candidate.model_dump(mode="json"))
    except ValueError:
        _add_reason(reason_codes, "invalid_candidate_identity")

    if candidate.snapshot_key != human_review.snapshot_key:
        _add_reason(reason_codes, "candidate_snapshot_mismatch")
    if candidate.reviewed_risk_sha256 != human_review.reviewed_content_sha256:
        _add_reason(reason_codes, "candidate_reviewed_risk_mismatch")
    if canonical_sha256(candidate.approved_risk.model_dump(mode="json")) != (
        human_review.reviewed_content_sha256
    ):
        _add_reason(reason_codes, "candidate_approved_risk_mismatch")


def validate_gherkin_candidate(
    *,
    candidate: GherkinCandidate,
    human_review: HumanReviewedRisk,
) -> GherkinValidationReport:
    """Check a candidate against its exact human-reviewed risk."""
    reason_codes: list[str] = []
    _validate_candidate_identity(candidate, human_review, reason_codes)

    for term in _required_risk_terms(human_review):
        if term not in candidate.gherkin_text:
            _add_reason(reason_codes, "bound_risk_term_removed")

    return GherkinValidationReport(
        approved=not reason_codes,
        reason_codes=tuple(reason_codes),
    )


def apply_gherkin_text_edit(
    *,
    candidate: GherkinCandidate,
    text: str,
    human_review: HumanReviewedRisk,
) -> GherkinCandidate:
    """Create a new candidate after a structure-preserving human text edit."""
    original_validation = validate_gherkin_candidate(
        candidate=candidate,
        human_review=human_review,
    )
    if not original_validation.approved:
        raise GherkinGenerationError(", ".join(original_validation.reason_codes))

    for term in _required_risk_terms(human_review):
        if term not in text:
            raise GherkinGenerationError("bound_risk_term_removed")

    parsed_steps = _parse_editable_text(
        text=text,
        feature_title=candidate.feature_title,
        scenario_title=candidate.scenario_title,
    )
    original_keywords = tuple(step.keyword for step in candidate.steps)
    edited_keywords = tuple(keyword for keyword, _ in parsed_steps)
    if (
        len(parsed_steps) != len(candidate.steps)
        or edited_keywords != original_keywords
    ):
        raise GherkinGenerationError("gherkin_step_structure_changed")

    edited_steps = tuple(
        GherkinStep(
            number=index,
            keyword=keyword,
            text=step_text,
        )
        for index, (keyword, step_text) in enumerate(parsed_steps, start=1)
    )
    edited_values = {
        **candidate.model_dump(
            mode="python",
            exclude={"candidate_id"},
        ),
        "steps": edited_steps,
        "gherkin_text": text,
    }

    try:
        edited_draft = GherkinCandidateDraft.model_validate(edited_values)
    except ValidationError as error:
        raise GherkinGenerationError(
            "edited Gherkin does not satisfy the approved candidate contract"
        ) from error

    return GherkinCandidate.from_draft(edited_draft)


def approve_gherkin(
    *,
    candidate: GherkinCandidate,
    human_review: HumanReviewedRisk,
    approved_at: datetime,
) -> GherkinApproval:
    """Approve only a locally valid candidate tied to this human review."""
    validation = validate_gherkin_candidate(
        candidate=candidate,
        human_review=human_review,
    )
    if not validation.approved:
        raise GherkinGenerationError(", ".join(validation.reason_codes))

    return GherkinApproval(
        snapshot_key=human_review.snapshot_key,
        candidate_id=candidate.candidate_id,
        candidate_sha256=canonical_sha256(candidate.model_dump(mode="json")),
        reviewed_risk_sha256=human_review.reviewed_content_sha256,
        approved_at=approved_at,
    )
