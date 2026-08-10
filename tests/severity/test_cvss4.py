import hashlib
import json
from pathlib import Path

import cvss
import pytest
from pydantic import ValidationError

from triageguard.domain import (
    CvssProfile,
    DifferentialSeverityAssessment,
    RiskContract,
    RuntimeObservation,
    VersionSeverityAssessment,
)
from triageguard.evidence import classify_differential
from triageguard.severity import (
    CvssAssessmentError,
    assess_differential_severity,
    calculate_cvss4,
)

FIXTURE_ROOT = (
    Path(__file__).parents[2] / "fixtures" / "patient_delete_authorization"
)


def _profile() -> CvssProfile:
    payload = json.loads(
        (FIXTURE_ROOT / "cvss_profile.json").read_text(encoding="utf-8")
    )
    return CvssProfile.model_validate(payload)


def _contract() -> RiskContract:
    return RiskContract.model_validate_json(
        (FIXTURE_ROOT / "approved_contract.json").read_bytes()
    )


def _observation(revision: str, behavior: str) -> RuntimeObservation:
    if behavior == "secure":
        request_status, resource_exists_after, exit_code = 403, True, 0
    elif behavior == "vulnerable":
        request_status, resource_exists_after, exit_code = 204, False, 1
    else:
        raise ValueError(f"unsupported test behavior: {behavior}")
    return RuntimeObservation(
        revision=revision,
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=True,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=request_status,
        resource_exists_after=resource_exists_after,
        pytest_exit_code=exit_code,
        reason_code="raw_execution_complete",
    )


def _independent_digest(value: dict) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def test_controlled_profile_calculates_with_cvss4_library() -> None:
    """Changing the vector or bypassing the maintained calculator must fail."""
    calculation = calculate_cvss4(_profile())

    assert calculation.vector == (
        "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:N/VI:H/VA:N/"
        "SC:N/SI:N/SA:N"
    )
    assert calculation.score == 7.1
    assert calculation.severity == "High"
    assert calculation.calculator == f"cvss-python/{cvss.__version__}"


def _provisional_payload(*, revision: str = "candidate") -> dict:
    profile = _profile()
    return {
        "revision": revision,
        "status": "provisional",
        "reason_code": "tested_vulnerability_observed",
        "profile_id": profile.profile_id,
        "profile_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
        "vector": profile.vector,
        "score": 7.1,
        "severity": "High",
        "metrics": profile.model_dump(mode="json")["metrics"],
        "calculator": f"cvss-python/{cvss.__version__}",
        "review_status": "expert_authored_provisional",
    }


def _not_scored_payload(*, revision: str = "base") -> dict:
    return {
        "revision": revision,
        "status": "not_scored",
        "reason_code": "tested_vulnerability_not_observed",
        "profile_id": None,
        "profile_sha256": None,
        "evidence_sha256": "c" * 64,
        "vector": None,
        "score": None,
        "severity": None,
        "metrics": [],
        "calculator": None,
        "review_status": "not_applicable",
    }


def test_provisional_assessment_model_requires_complete_scoring_provenance() -> None:
    """Dropping any scoring provenance must invalidate a numeric assessment."""
    valid = VersionSeverityAssessment.model_validate(_provisional_payload())

    assert valid.score == 7.1
    assert len(valid.metrics) == 11

    for field in (
        "profile_id",
        "profile_sha256",
        "vector",
        "score",
        "severity",
        "calculator",
    ):
        payload = _provisional_payload()
        payload[field] = None
        with pytest.raises(ValidationError, match="provisional"):
            VersionSeverityAssessment.model_validate(payload)

    boolean_score = _provisional_payload()
    boolean_score["score"] = True
    with pytest.raises(ValidationError):
        VersionSeverityAssessment.model_validate(boolean_score)


def test_not_scored_assessment_model_forbids_numeric_or_profile_claims() -> None:
    """A not-scored version must never carry a hidden zero or vector."""
    valid = VersionSeverityAssessment.model_validate(_not_scored_payload())

    assert valid.score is None
    assert valid.vector is None

    for field, value in (
        ("score", 0.0),
        ("vector", _profile().vector),
        ("profile_id", _profile().profile_id),
        ("profile_sha256", "d" * 64),
        ("severity", "None"),
        ("calculator", f"cvss-python/{cvss.__version__}"),
        ("metrics", _profile().model_dump(mode="json")["metrics"]),
    ):
        payload = _not_scored_payload()
        payload[field] = value
        with pytest.raises(ValidationError, match="not-scored"):
            VersionSeverityAssessment.model_validate(payload)


def test_differential_assessment_model_requires_distinct_revisions() -> None:
    """Base and candidate severity must not accidentally bind one revision twice."""
    base = VersionSeverityAssessment.model_validate(_not_scored_payload())
    candidate = VersionSeverityAssessment.model_validate(_provisional_payload())

    assessment = DifferentialSeverityAssessment(base=base, candidate=candidate)
    assert assessment.base.revision == "base"
    assert assessment.candidate.revision == "candidate"

    with pytest.raises(ValidationError, match="distinct"):
        DifferentialSeverityAssessment(
            base=base,
            candidate=VersionSeverityAssessment.model_validate(
                _provisional_payload(revision="base")
            ),
        )


@pytest.mark.parametrize(
    ("base_behavior", "candidate_behavior", "base_status", "candidate_status"),
    [
        ("secure", "vulnerable", "not_scored", "provisional"),
        ("vulnerable", "secure", "provisional", "not_scored"),
        ("secure", "secure", "not_scored", "not_scored"),
        ("vulnerable", "vulnerable", "provisional", "provisional"),
    ],
)
def test_differential_severity_scores_only_observed_vulnerable_sides(
    base_behavior: str,
    candidate_behavior: str,
    base_status: str,
    candidate_status: str,
) -> None:
    """Applying the profile to a secure side must fail this outcome matrix."""
    base = _observation("base-sha", base_behavior)
    candidate = _observation("candidate-sha", candidate_behavior)
    evidence = classify_differential([base], [candidate], _contract())

    assessment = assess_differential_severity(evidence, _profile())

    assert assessment.base.status == base_status
    assert assessment.candidate.status == candidate_status
    assert assessment.base.evidence_sha256 == _independent_digest(
        base.model_dump(mode="json")
    )
    assert assessment.candidate.evidence_sha256 == _independent_digest(
        candidate.model_dump(mode="json")
    )
    expected_profile_hash = _independent_digest(
        _profile().model_dump(mode="json")
    )
    for side in (assessment.base, assessment.candidate):
        if side.status == "provisional":
            assert side.score == 7.1
            assert side.severity == "High"
            assert side.profile_sha256 == expected_profile_hash
        else:
            assert side.reason_code == "tested_vulnerability_not_observed"
            assert side.score is None


def test_unstable_and_inconclusive_evidence_never_receive_a_numeric_score() -> None:
    """Representative vulnerable facts cannot override incomplete repeatability."""
    unstable = classify_differential(
        [
            _observation("base-sha", "secure"),
            _observation("base-sha", "vulnerable"),
        ],
        [
            _observation("candidate-sha", "vulnerable"),
            _observation("candidate-sha", "vulnerable"),
        ],
        _contract(),
    )
    broken_base = _observation("base-sha", "secure").model_copy(
        update={"control_succeeded": False}
    )
    inconclusive = classify_differential(
        [broken_base],
        [_observation("candidate-sha", "vulnerable")],
        _contract(),
    )

    for evidence in (unstable, inconclusive):
        assessment = assess_differential_severity(evidence, _profile())
        for side in (assessment.base, assessment.candidate):
            assert side.status == "not_scored"
            assert side.reason_code == "insufficient_evidence_for_severity"
            assert side.score is None
            assert side.vector is None


def test_invalid_library_vector_abstains_instead_of_returning_partial_severity() -> None:
    """An unsupported metric value must stop scoring before a claim is built."""
    payload = _profile().model_dump(mode="json")
    payload["vector"] = payload["vector"].replace("/AV:N/", "/AV:X/")
    payload["metrics"][0]["value"] = "X"
    invalid_profile = CvssProfile.model_validate(payload)
    evidence = classify_differential(
        [_observation("base-sha", "secure")],
        [_observation("candidate-sha", "vulnerable")],
        _contract(),
    )

    with pytest.raises(CvssAssessmentError, match="calculation failed"):
        assess_differential_severity(evidence, invalid_profile)
