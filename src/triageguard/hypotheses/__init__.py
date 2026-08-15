"""Structured, evidence-bound risk-hypothesis operations."""

from triageguard.hypotheses.generator import (
    RISK_SYSTEM_PROMPT,
    build_risk_request,
    generate_risk_assessment,
)

__all__ = [
    "RISK_SYSTEM_PROMPT",
    "build_risk_request",
    "generate_risk_assessment",
]
