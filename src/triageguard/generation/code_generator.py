"""Structured LLM boundary for constrained pytest-bdd source generation."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from triageguard.contracts.gherkin import validate_gherkin_alignment
from triageguard.domain.models import RiskContract, TestPlan
from triageguard.generation.planner import PlanValidationError, validate_test_plan
from triageguard.generation.primitives import primitive_catalog_prompt_data
from triageguard.llm.gateway import (
    ModelRequest,
    ModelResponse,
    StructuredModelGateway,
    canonical_json,
)

ALLOWED_IMPORTS = (
    "os",
    "pathlib",
    "pytest",
    "pytest_bdd",
    "triageguard.runtime",
)

_SYSTEM_PROMPT = "\n".join(  # noqa: FLY002 - stable line-oriented prompt
    (
        "You render one approved security TestPlan as pytest-bdd Python source.",
        "Return only JSON matching the supplied schema and preserve every code byte.",
        "Use exact Gherkin step text and only the supplied primitive-to-runtime mappings.",
        "Render every planned primitive occurrence exactly once in its planned phase and preserve capture-to-input dataflow.",
        "Use an independent control patient and record its pre-delete existence, administrator delete status, and post-delete absence.",
        "Use OpenMrsTestClient for every target interaction and ObservationWriter for observations.",
        "Do not import or invoke arbitrary HTTP, process, shell, evaluation, or network helpers.",
        "Read target URL and credentials from environment variables without fallback defaults.",
        "Do not skip tests, swallow failures, weaken assertions, or invent evidence.",
    )
)

_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 1},
        "implemented_steps": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "used_primitives": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "contract_id": {"type": "string", "minLength": 1},
    },
    "required": ["code", "implemented_steps", "used_primitives", "contract_id"],
    "additionalProperties": False,
}


class GeneratedCodeArtifact(BaseModel):
    """Model-rendered source retained alongside immutable call provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    implemented_steps: list[str]
    used_primitives: list[str]
    contract_id: str = Field(min_length=1)
    model_response: ModelResponse


def generate_pytest(
    contract: RiskContract,
    gherkin: str,
    plan: TestPlan,
    gateway: StructuredModelGateway,
) -> GeneratedCodeArtifact:
    """Request source only after deterministic contract and plan gates succeed."""
    alignment = validate_gherkin_alignment(contract, gherkin)
    if not alignment.approved:
        raise PlanValidationError(
            f"gherkin_{reason}" for reason in alignment.reason_codes
        )
    validate_test_plan(contract, plan)

    request = ModelRequest(
        purpose="pytest_generation",
        system_prompt=_SYSTEM_PROMPT,
        payload={
            "approved_contract": contract.model_dump(mode="json"),
            "contract_sha256": _sha256(
                canonical_json(contract.model_dump(mode="json"))
            ),
            "exact_gherkin": gherkin,
            "gherkin_sha256": _sha256(gherkin),
            "plan": plan.model_dump(mode="json"),
            "primitive_catalog": primitive_catalog_prompt_data(),
            "allowed_imports": list(ALLOWED_IMPORTS),
        },
        output_schema=_GENERATION_SCHEMA,
        max_output_tokens=5000,
    )
    response = gateway.generate(request)
    try:
        artifact = GeneratedCodeArtifact.model_validate(
            {**response.data, "model_response": response}
        )
    except ValidationError as error:
        raise PlanValidationError(["model_generated_code_invalid"]) from error
    if artifact.contract_id != contract.contract_id:
        raise PlanValidationError(["contract_id_changed"])
    return artifact


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
