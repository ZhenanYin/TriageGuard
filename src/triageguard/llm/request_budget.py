"""Exact provider-body construction and local model-request size policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from triageguard.config import Settings
from triageguard.llm.gateway import ModelGatewayError, ModelRequest
from triageguard.provenance import canonical_json


class ModelRequestTooLarge(ModelGatewayError):
    """The exact provider body exceeds the locally declared safe budget."""


@dataclass(frozen=True)
class ProviderRequestBudget:
    """One explicit, reproducible byte policy for a provider and model."""

    provider: Literal["groq"]
    model: str
    max_body_bytes: int
    policy_version: Literal["groq-body-v1"] = "groq-body-v1"

    def __post_init__(self) -> None:
        if self.provider != "groq":
            raise ValueError("provider must be 'groq'")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if type(self.max_body_bytes) is not int or self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be a positive integer")

    @classmethod
    def from_settings(cls, settings: Settings) -> ProviderRequestBudget:
        """Copy only the public fields that determine the outbound byte boundary."""
        return cls(
            provider=settings.llm_provider,
            model=settings.llm_model,
            max_body_bytes=settings.max_model_request_bytes,
        )


def groq_request_body(*, request: ModelRequest, model: str) -> dict[str, Any]:
    """Build the one exact body used for both measurement and Groq transmission."""
    return {
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": canonical_json(request.payload)},
        ],
        "model": model,
        "max_tokens": request.max_output_tokens,
        "reasoning_effort": "medium",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.purpose,
                "strict": True,
                "schema": request.output_schema,
            },
        },
    }


def groq_request_body_bytes(*, request: ModelRequest, model: str) -> int:
    """Measure the canonical UTF-8 representation of the exact outbound body."""
    body = groq_request_body(request=request, model=model)
    return len(canonical_json(body).encode("utf-8"))
