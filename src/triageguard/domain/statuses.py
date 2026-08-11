"""Explicit user-visible workflow and execution contexts."""

from enum import Enum


class EnvironmentKind(str, Enum):
    """The environment that supplied the observations in a research run."""

    CONTROLLED_FIXTURE = "controlled_fixture"
    REAL_PR_ANALYSIS = "real_pr_analysis"


class MilestoneTwoStatus(str, Enum):
    """Terminal states for an immutable real pull-request analysis run."""

    APPROVED_GHERKIN = "approved_gherkin"
    NO_MEANINGFUL_SECURITY_RISK_FOUND = "no_meaningful_security_risk_found"
    INSUFFICIENT_CONTEXT_TO_ASSESS = "insufficient_context_to_assess"
    STALE = "stale"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    """The explicit states shown by the V2 UI and API."""

    SUPPORTED_RISK_PROPOSED = "supported_risk_proposed"
    NO_SUPPORTED_RISK_FOUND = "no_supported_risk_found"
    AWAITING_HUMAN_APPROVAL = "awaiting_human_approval"
    GENERATION_ABSTAINED = "generation_abstained"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_INCONCLUSIVE = "execution_inconclusive"
    UNSTABLE_RESULT = "unstable_result"
    NO_REGRESSION_OBSERVED = "no_regression_observed"
    PRE_EXISTING_RISK_OBSERVED = "pre_existing_risk_observed"
    CANDIDATE_FIX_OBSERVED = "candidate_fix_observed"
    CANDIDATE_REGRESSION_OBSERVED = "candidate_regression_observed"
    VALIDATED_EVIDENCE = "validated_evidence"
