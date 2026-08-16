"""Shared request, response, validation, and provenance types for model calls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Protocol

from jsonschema import SchemaError, ValidationError, validate
from pydantic import BaseModel, ConfigDict, Field

from triageguard.provenance import canonical_json


class ModelRequest(BaseModel):
    """One bounded structured-output request made by a named workflow operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: Literal[
        "risk_hypothesis",
        "test_plan",
        "pytest_generation",
        "mechanical_repair",
        "plain_explanation",
        "gherkin_generation",
        "testability_assessment",
    ]
    system_prompt: str = Field(min_length=1)
    payload: dict[str, Any]
    output_schema: dict[str, Any]
    max_output_tokens: int = Field(gt=0)


class ModelAttempt(BaseModel):
    """Per-attempt metadata retained for later durable transformation recording."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(ge=1)
    started_at: datetime
    finished_at: datetime
    latency_ms: int = Field(ge=0)
    outcome: Literal["succeeded", "transient_error", "invalid_output", "failed"]
    error_type: str | None = None


class ModelFailureProvenance(BaseModel):
    """Recorder-ready metadata retained when a live invocation cannot return data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    attempts: tuple[ModelAttempt, ...] = Field(min_length=1)
    final_outcome: Literal["transient_error", "failed", "invalid_output"]
    reason_code: str = Field(min_length=1)


class ModelResponse(BaseModel):
    """Parsed structured output plus immutable, recorder-ready call provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: dict[str, Any]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    latency_ms: int = Field(ge=0)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    attempts: list[ModelAttempt] = Field(min_length=1)


class StructuredModelGateway(Protocol):
    """Boundary implemented by explicit live and deterministic replay gateways."""

    def generate(self, request: ModelRequest) -> ModelResponse: ...


class ModelGatewayError(RuntimeError):
    """Base failure whose attempt metadata can be recorded by a later workflow."""

    def __init__(
        self,
        message: str,
        attempts: list[ModelAttempt] | None = None,
        *,
        provenance: ModelFailureProvenance | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts or [])
        self.provenance = provenance


class ModelOutputInvalid(ModelGatewayError):
    """The provider returned JSON that cannot satisfy the requested output schema."""


def prompt_sha256(request: ModelRequest) -> str:
    """Hash exactly the system and user content sent to a structured model."""
    prompt = canonical_json(
        {"payload": request.payload, "system_prompt": request.system_prompt}
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def response_sha256(content: str) -> str:
    """Hash the raw provider response before it is interpreted as structured data."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def request_sha256(request: ModelRequest) -> str:
    """Hash the complete canonical request without retaining its potentially sensitive data."""
    serialized_request = canonical_json(request.model_dump(mode="json"))
    return hashlib.sha256(serialized_request.encode("utf-8")).hexdigest()


def error_sha256(error: BaseException) -> str:
    """Hash raw provider error text for attribution without storing it in provenance."""
    return hashlib.sha256(str(error).encode("utf-8")).hexdigest()


def parse_and_validate_output(
    content: str, output_schema: dict[str, Any]
) -> dict[str, Any]:
    """Parse once and reject any result incompatible with the requested JSON Schema."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelOutputInvalid("model response is not valid JSON") from error
    if not isinstance(data, dict):
        raise ModelOutputInvalid("model response must be a JSON object")

    try:
        validate(data, output_schema)
    except SchemaError as error:
        raise ModelOutputInvalid(
            "model request contains an invalid JSON Schema"
        ) from error
    except ValidationError as error:
        raise ModelOutputInvalid(
            "model response does not match output schema"
        ) from error
    return data
