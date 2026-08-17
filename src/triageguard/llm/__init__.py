"""Explicit structured model gateways for live and replay operation."""

from triageguard.llm.gateway import (
    ModelAttempt,
    ModelFailureProvenance,
    ModelGatewayError,
    ModelOutputInvalid,
    ModelRequest,
    ModelResponse,
    StructuredModelGateway,
)
from triageguard.llm.groq_gateway import GroqStructuredGateway
from triageguard.llm.replay_gateway import ReplayGateway, ReplayResponseMissing
from triageguard.llm.request_budget import (
    ModelRequestTooLarge,
    ProviderRequestBudget,
    groq_request_body,
    groq_request_body_bytes,
)

__all__ = [
    "GroqStructuredGateway",
    "ModelAttempt",
    "ModelFailureProvenance",
    "ModelGatewayError",
    "ModelOutputInvalid",
    "ModelRequest",
    "ModelRequestTooLarge",
    "ModelResponse",
    "ProviderRequestBudget",
    "ReplayGateway",
    "ReplayResponseMissing",
    "StructuredModelGateway",
    "groq_request_body",
    "groq_request_body_bytes",
]
