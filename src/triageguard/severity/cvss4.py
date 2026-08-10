"""Narrow adapter around the maintained Red Hat CVSS v4 calculator."""

from __future__ import annotations

from dataclasses import dataclass

import cvss
from cvss import CVSS4

from triageguard.domain import (
    CvssProfile,
    DifferentialEvidence,
    DifferentialSeverityAssessment,
    RuntimeObservation,
    VersionSeverityAssessment,
    WorkflowStatus,
)
from triageguard.provenance import canonical_sha256


class CvssAssessmentError(ValueError):
    """The prepared metric profile could not be calculated safely."""


@dataclass(frozen=True)
class CvssCalculation:
    """Normalized deterministic result with calculator provenance."""

    vector: str
    score: float
    severity: str
    calculator: str


def calculate_cvss4(profile: CvssProfile) -> CvssCalculation:
    """Calculate, but never infer, a score from a validated metric profile."""
    try:
        calculated = CVSS4(profile.vector)
        vector = calculated.clean_vector()
        score = round(float(calculated.scores()[0]), 1)
        severity = str(calculated.severities()[0])
    except Exception as error:
        raise CvssAssessmentError("CVSS 4.0 profile calculation failed") from error
    return CvssCalculation(
        vector=vector,
        score=score,
        severity=severity,
        calculator=f"cvss-python/{cvss.__version__}",
    )


def assess_differential_severity(
    evidence: DifferentialEvidence,
    profile: CvssProfile,
) -> DifferentialSeverityAssessment:
    """Apply a prepared profile only to repeatable observed vulnerability facts."""
    insufficient = evidence.status in {
        WorkflowStatus.UNSTABLE_RESULT,
        WorkflowStatus.EXECUTION_INCONCLUSIVE,
    }
    calculation = None
    if not insufficient and (
        evidence.base.security_behavior == "vulnerable"
        or evidence.candidate.security_behavior == "vulnerable"
    ):
        calculation = calculate_cvss4(profile)
    return DifferentialSeverityAssessment(
        base=_assess_version(
            evidence.base,
            profile,
            calculation,
            insufficient=insufficient,
        ),
        candidate=_assess_version(
            evidence.candidate,
            profile,
            calculation,
            insufficient=insufficient,
        ),
    )


def _assess_version(
    observation: RuntimeObservation,
    profile: CvssProfile,
    calculation: CvssCalculation | None,
    *,
    insufficient: bool,
) -> VersionSeverityAssessment:
    evidence_sha256 = canonical_sha256(observation.model_dump(mode="json"))
    if insufficient or observation.security_behavior != "vulnerable":
        return VersionSeverityAssessment(
            revision=observation.revision,
            status="not_scored",
            reason_code=(
                "insufficient_evidence_for_severity"
                if insufficient
                else "tested_vulnerability_not_observed"
            ),
            evidence_sha256=evidence_sha256,
            review_status="not_applicable",
        )
    if calculation is None:
        raise CvssAssessmentError(
            "vulnerable evidence has no deterministic CVSS calculation"
        )
    return VersionSeverityAssessment(
        revision=observation.revision,
        status="provisional",
        reason_code="tested_vulnerability_observed",
        profile_id=profile.profile_id,
        profile_sha256=canonical_sha256(profile.model_dump(mode="json")),
        evidence_sha256=evidence_sha256,
        vector=calculation.vector,
        score=calculation.score,
        severity=calculation.severity,
        metrics=list(profile.metrics),
        calculator=calculation.calculator,
        review_status="expert_authored_provisional",
    )
