"""Build structured Gherkin-generation requests from human-reviewed risks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import ValidationError

from triageguard.domain.pr_analysis import (
    ContextBundle,
    GherkinApproval,
    GherkinCandidate,
    GherkinCandidateDraft,
    GherkinStep,
    HumanReviewedRisk,
    TestabilityAssessment,
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


def _validate_gherkin_request_inputs(
    *,
    human_review: HumanReviewedRisk,
    testability_assessment: TestabilityAssessment,
    context: ContextBundle,
) -> tuple[HumanReviewedRisk, TestabilityAssessment, ContextBundle]:
    """Require one coherent reviewed risk, testability result, and frozen context."""
    try:
        reviewed = HumanReviewedRisk.model_validate(
            human_review.model_dump(mode="json")
        )
    except ValidationError as error:
        raise ValueError(
            "Gherkin generation requires a valid immutable human review"
        ) from error

    try:
        testability = TestabilityAssessment.model_validate(
            testability_assessment.model_dump(mode="json")
        )
    except ValidationError as error:
        raise ValueError(
            "Gherkin generation requires a valid immutable testability assessment"
        ) from error

    try:
        frozen_context = ContextBundle.model_validate(context.model_dump(mode="json"))
    except ValidationError as error:
        raise ValueError(
            "Gherkin generation requires valid immutable frozen context evidence"
        ) from error

    if reviewed.snapshot_key != frozen_context.snapshot_key:
        raise ValueError("human review snapshot key must match the frozen context")
    if testability.snapshot_key != reviewed.snapshot_key:
        raise ValueError(
            "testability assessment snapshot key must match the human review"
        )
    if testability.context_sha256 != frozen_context.context_sha256:
        raise ValueError(
            "testability assessment context hash must match the frozen context"
        )
    if testability.reviewed_risk_sha256 != reviewed.reviewed_content_sha256:
        raise ValueError("testability assessment must match the reviewed risk content")
    if testability.decision != "testable_from_frozen_evidence":
        raise ValueError(
            "Gherkin generation requires testable frozen-evidence assessment"
        )

    anchors_by_id = {anchor.anchor_id: anchor for anchor in frozen_context.anchors}
    assessment_anchor_ids = tuple(
        anchor_id
        for binding in testability.bindings
        for anchor_id in binding.anchor_ids
    )
    if any(anchor_id not in anchors_by_id for anchor_id in assessment_anchor_ids):
        raise ValueError(
            "testability assessment cited an anchor absent from the frozen context"
        )
    if not any(
        anchors_by_id[anchor_id].change_relation == "integration_change"
        for anchor_id in assessment_anchor_ids
    ):
        raise ValueError("testability assessment requires integration-change evidence")

    if (
        reviewed.reviewed_grounding is not None
        and reviewed.reviewed_grounding.context_sha256 != frozen_context.context_sha256
    ):
        raise ValueError("human-review grounding must match the frozen Gherkin context")

    return reviewed, testability, frozen_context


def build_gherkin_request(
    *,
    human_review: HumanReviewedRisk,
    testability_assessment: TestabilityAssessment,
    context: ContextBundle,
) -> ModelRequest:
    """Build one strict Gherkin request after local frozen-evidence approval."""
    reviewed, testability, frozen_context = _validate_gherkin_request_inputs(
        human_review=human_review,
        testability_assessment=testability_assessment,
        context=context,
    )

    approved_risk = reviewed.reviewed_risk.model_dump(mode="json")

    return ModelRequest(
        purpose="gherkin_generation",
        system_prompt=GHERKIN_SYSTEM_PROMPT,
        payload={
            "snapshot_key": reviewed.snapshot_key,
            "reviewed_risk_sha256": reviewed.reviewed_content_sha256,
            "testability_assessment_sha256": testability.assessment_sha256,
            "context_sha256": frozen_context.context_sha256,
            "approved_risk": approved_risk,
            "context_anchors": [
                anchor.model_dump(mode="json") for anchor in frozen_context.anchors
            ],
            "output_rules": {
                "scenario_count": 1,
                "feature_count": 1,
                "citation_rule": (
                    "Use only the approved-risk terms and the supplied frozen "
                    "context anchors."
                ),
                "testability_rule": (
                    "The scenario must remain within the locally approved setup, "
                    "action, and observable evidence roles."
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
    testability_assessment: TestabilityAssessment,
    context: ContextBundle,
    gateway: StructuredModelGateway,
) -> tuple[GherkinCandidate, ModelResponse]:
    """Generate one scenario only after local frozen-evidence testability approval."""
    reviewed, _, frozen_context = _validate_gherkin_request_inputs(
        human_review=human_review,
        testability_assessment=testability_assessment,
        context=context,
    )
    request = build_gherkin_request(
        human_review=reviewed,
        testability_assessment=testability_assessment,
        context=context,
    )
    response = gateway.generate(request)

    try:
        draft = GherkinCandidateDraft.model_validate(response.data)
    except ValidationError as error:
        raise ModelOutputInvalid(
            "model response does not form a coherent Gherkin candidate draft"
        ) from error

    if draft.snapshot_key != reviewed.snapshot_key:
        raise ModelOutputInvalid(
            "model response snapshot key does not match the human review"
        )
    if draft.context_sha256 != frozen_context.context_sha256:
        raise ModelOutputInvalid(
            "model response context hash does not match the frozen context"
        )
    if draft.reviewed_risk_sha256 != reviewed.reviewed_content_sha256:
        raise ModelOutputInvalid(
            "model response reviewed-risk hash does not match the human review"
        )
    if draft.approved_risk != reviewed.reviewed_risk:
        raise ModelOutputInvalid(
            "model response approved risk does not match the human review"
        )

    return GherkinCandidate.from_draft(draft), response


_STEP_PATTERN = re.compile(r"^(Given|When|Then|And)\s+(.+)$")
_PROHIBITED_GHERKIN_CONTENT = re.compile(
    r"```|#|[(){}\[\];|><$`]|"
    r"\b(?:def|class|import|os|sys|subprocess|python|bash|sh|curl|wget|rm|"
    r"chmod|sudo|eval|exec|cat|echo)\b|"
    r"(?:^|\s)/(?:\S+)",
    flags=re.IGNORECASE,
)
_CODE_SHAPED_IDENTIFIER = re.compile(r"\b[a-z][A-Za-z0-9_$]*[A-Z][A-Za-z0-9_$]*\b")


class GherkinGenerationError(ValueError):
    """A generated or edited scenario failed local Gherkin safety checks."""


GherkinValidationDecision = Literal[
    "valid_evidence_bound_gherkin",
    "needs_more_frozen_evidence",
    "hypothesis_changed",
    "invalid_gherkin",
]


@dataclass(frozen=True)
class GherkinValidationReport:
    """The local decision about whether edited Gherkin may advance."""

    decision: GherkinValidationDecision
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


def _code_shaped_identifiers(text: str) -> set[str]:
    """Return Java-like camel-case identifiers introduced in scenario prose."""
    return set(_CODE_SHAPED_IDENTIFIER.findall(text))


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


def _decision_from_reasons(
    reason_codes: list[str],
) -> GherkinValidationDecision:
    """Classify the reviewer-facing next action from deterministic checks."""
    if not reason_codes:
        return "valid_evidence_bound_gherkin"
    if any(
        reason_code
        in {
            "unknown_step_evidence_anchor",
            "missing_integration_step_evidence",
            "unbound_code_identifier",
        }
        for reason_code in reason_codes
    ):
        return "needs_more_frozen_evidence"
    if any(
        reason_code
        in {
            "candidate_reviewed_risk_mismatch",
            "candidate_approved_risk_mismatch",
            "bound_risk_term_removed",
        }
        for reason_code in reason_codes
    ):
        return "hypothesis_changed"
    return "invalid_gherkin"


def validate_edited_gherkin(
    *,
    candidate: GherkinCandidate,
    text: str,
    human_review: HumanReviewedRisk,
    context: ContextBundle,
) -> GherkinValidationReport:
    """Classify one edited scenario against its current frozen evidence context."""
    reason_codes: list[str] = []
    _validate_candidate_identity(candidate, human_review, reason_codes)

    try:
        frozen_context = ContextBundle.model_validate(context.model_dump(mode="json"))
    except ValidationError:
        _add_reason(reason_codes, "invalid_frozen_context")
        frozen_context = None

    if (
        frozen_context is not None
        and candidate.context_sha256 != frozen_context.context_sha256
    ):
        _add_reason(reason_codes, "candidate_context_mismatch")

    if frozen_context is not None:
        anchors_by_id = {anchor.anchor_id: anchor for anchor in frozen_context.anchors}
        evidence_anchor_ids = tuple(
            anchor_id
            for binding in candidate.step_evidence_bindings
            for anchor_id in binding.anchor_ids
        )
        if any(anchor_id not in anchors_by_id for anchor_id in evidence_anchor_ids):
            _add_reason(reason_codes, "unknown_step_evidence_anchor")
        elif not any(
            anchors_by_id[anchor_id].change_relation == "integration_change"
            for anchor_id in evidence_anchor_ids
        ):
            _add_reason(reason_codes, "missing_integration_step_evidence")

    try:
        parsed_steps = _parse_editable_text(
            text=text,
            feature_title=candidate.feature_title,
            scenario_title=candidate.scenario_title,
        )
    except GherkinGenerationError as error:
        _add_reason(reason_codes, str(error))
        parsed_steps = ()

    if parsed_steps:
        original_keywords = tuple(step.keyword for step in candidate.steps)
        edited_keywords = tuple(keyword for keyword, _ in parsed_steps)
        if (
            len(parsed_steps) != len(candidate.steps)
            or edited_keywords != original_keywords
        ):
            _add_reason(reason_codes, "gherkin_step_structure_changed")

    if _PROHIBITED_GHERKIN_CONTENT.search(text):
        _add_reason(reason_codes, "gherkin_text_contains_implementation_code")

    if frozen_context is not None:
        known_identifiers = _code_shaped_identifiers(candidate.gherkin_text)
        known_identifiers.update(human_review.reviewed_risk.code_identifiers)
        for anchor in frozen_context.anchors:
            known_identifiers.update(_code_shaped_identifiers(anchor.text))

        introduced_identifiers = _code_shaped_identifiers(text) - known_identifiers
        if introduced_identifiers:
            _add_reason(reason_codes, "unbound_code_identifier")

    for term in _required_risk_terms(human_review):
        if term not in text:
            _add_reason(reason_codes, "bound_risk_term_removed")

    decision = _decision_from_reasons(reason_codes)
    return GherkinValidationReport(
        decision=decision,
        approved=decision == "valid_evidence_bound_gherkin",
        reason_codes=tuple(reason_codes),
    )


def validate_gherkin_candidate(
    *,
    candidate: GherkinCandidate,
    human_review: HumanReviewedRisk,
    context: ContextBundle,
) -> GherkinValidationReport:
    """Validate the candidate's current text against its frozen evidence."""
    return validate_edited_gherkin(
        candidate=candidate,
        text=candidate.gherkin_text,
        human_review=human_review,
        context=context,
    )


def apply_gherkin_text_edit(
    *,
    candidate: GherkinCandidate,
    text: str,
    human_review: HumanReviewedRisk,
    context: ContextBundle,
) -> GherkinCandidate:
    """Create a new candidate after a structure-preserving human text edit."""
    original_validation = validate_edited_gherkin(
        candidate=candidate,
        text=text,
        human_review=human_review,
        context=context,
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
    context: ContextBundle,
    approved_at: datetime,
) -> GherkinApproval:
    """Approve only a locally valid candidate tied to this human review."""
    validation = validate_edited_gherkin(
        candidate=candidate,
        text=candidate.gherkin_text,
        human_review=human_review,
        context=context,
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
