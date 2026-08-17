"""Contracts for exact provider-body measurement and local request policy."""

import pytest

from triageguard.config import Settings
from triageguard.llm.gateway import ModelRequest, canonical_json
from triageguard.llm.request_budget import (
    ProviderRequestBudget,
    groq_request_body,
    groq_request_body_bytes,
)


def _request() -> ModelRequest:
    return ModelRequest(
        purpose="risk_hypothesis",
        system_prompt="Use only frozen evidence.",
        payload={"snapshot_key": "a" * 64, "anchors": ["anchor-1"]},
        output_schema={
            "type": "object",
            "properties": {"outcome": {"type": "string"}},
            "required": ["outcome"],
            "additionalProperties": False,
        },
        max_output_tokens=2_048,
    )


def test_provider_request_budget_copies_the_approved_public_policy() -> None:
    settings = Settings(max_model_request_bytes=6_500)

    budget = ProviderRequestBudget.from_settings(settings)

    assert budget.provider == "groq"
    assert budget.model == "openai/gpt-oss-120b"
    assert budget.max_body_bytes == 6_500
    assert budget.policy_version == "groq-body-v1"


@pytest.mark.parametrize("value", [0, -1, True])
def test_provider_request_budget_requires_a_positive_strict_integer(value) -> None:
    with pytest.raises(ValueError, match="max_body_bytes"):
        ProviderRequestBudget(
            provider="groq",
            model="openai/gpt-oss-120b",
            max_body_bytes=value,
        )


def test_groq_body_measurement_uses_the_exact_canonical_outbound_body() -> None:
    request = _request()

    body = groq_request_body(request=request, model="openai/gpt-oss-120b")

    assert body["messages"] == [
        {"role": "system", "content": "Use only frozen evidence."},
        {
            "role": "user",
            "content": canonical_json(request.payload),
        },
    ]
    assert body["response_format"]["json_schema"]["schema"] == request.output_schema
    assert groq_request_body_bytes(request=request, model="openai/gpt-oss-120b") == len(
        canonical_json(body).encode("utf-8")
    )
