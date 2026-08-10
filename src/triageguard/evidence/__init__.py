"""Deterministic classification of repeated differential observations."""

from triageguard.evidence.classifier import (
    UnsupportedRiskContractError,
    classify_differential,
)

__all__ = ["UnsupportedRiskContractError", "classify_differential"]
