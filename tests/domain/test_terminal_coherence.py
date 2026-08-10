from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from triageguard.domain import (
    CvssProfile,
    DifferentialEvidence,
    EnvironmentKind,
    RunRecord,
    RuntimeObservation,
    WorkflowStatus,
)
from triageguard.severity import assess_differential_severity

FIXTURE_ROOT = (
    Path(__file__).parents[2] / "fixtures" / "patient_delete_authorization"
)


def _observation(revision: str, *, secure: bool) -> RuntimeObservation:
    return RuntimeObservation(
        revision=revision,
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=True,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=403 if secure else 204,
        resource_exists_after=secure,
        pytest_exit_code=0 if secure else 1,
        reason_code="raw_execution_complete",
    )


def _evidence_payload() -> dict[str, object]:
    return {
        "base": _observation("base-revision", secure=True),
        "candidate": _observation("candidate-revision", secure=False),
        "base_revision": "base-revision",
        "candidate_revision": "candidate-revision",
        "repetitions": 1,
        "stable": True,
        "status": WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
        "reason_code": "candidate_regression_observed",
        "explanation": (
            "The base denied unauthorized deletion and preserved the patient, "
            "while the candidate allowed deletion and removed the patient."
        ),
        "base_differing_run_indexes": [],
        "candidate_differing_run_indexes": [],
    }


def _profile() -> CvssProfile:
    return CvssProfile.model_validate_json(
        (FIXTURE_ROOT / "cvss_profile.json").read_bytes()
    )


def _scored_record_payload() -> dict[str, object]:
    evidence = DifferentialEvidence.model_validate(
        {
            **_evidence_payload(),
            "execution_manifest_sha256s": ["1" * 64, "2" * 64],
        }
    )
    return {
        "run_id": "run-scored",
        "environment_kind": EnvironmentKind.CONTROLLED_FIXTURE,
        "base_revision": "base-revision",
        "candidate_revision": "candidate-revision",
        "status": WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
        "reason_code": "candidate_regression_observed",
        "explanation": (
            "The base denied unauthorized deletion and preserved the patient, "
            "while the candidate allowed deletion and removed the patient."
        ),
        "started_at": datetime(2026, 8, 7, 12, tzinfo=UTC),
        "finished_at": datetime(2026, 8, 7, 12, 0, 1, tzinfo=UTC),
        "differential_evidence": evidence.model_dump(mode="json"),
        "execution_manifest_sha256s": ["1" * 64, "2" * 64],
        "severity_assessment": assess_differential_severity(
            evidence, _profile()
        ).model_dump(mode="json"),
    }


def test_run_record_binds_severity_to_exact_revisions_and_observations() -> None:
    """Changing the revision or evidence digest must invalidate terminal severity."""
    valid = RunRecord.model_validate(_scored_record_payload())
    assert valid.severity_assessment is not None
    assert valid.severity_assessment.candidate.score == 7.1

    wrong_revision = _scored_record_payload()
    wrong_revision["severity_assessment"]["candidate"]["revision"] = (
        "candidate-other"
    )
    wrong_evidence = _scored_record_payload()
    wrong_evidence["severity_assessment"]["candidate"]["evidence_sha256"] = (
        "f" * 64
    )

    with pytest.raises(ValidationError, match="severity revisions"):
        RunRecord.model_validate(wrong_revision)
    with pytest.raises(ValidationError, match="observation hash"):
        RunRecord.model_validate(wrong_evidence)


def test_run_record_without_differential_evidence_forbids_severity() -> None:
    """A stage failure with no runtime comparison must not retain a score."""
    payload = _scored_record_payload()
    payload.update(
        {
            "status": WorkflowStatus.EXECUTION_INCONCLUSIVE,
            "reason_code": "execution_failed",
            "explanation": "Execution failed.",
            "differential_evidence": None,
        }
    )

    with pytest.raises(ValidationError, match="without differential evidence"):
        RunRecord.model_validate(payload)


def test_run_record_with_differential_evidence_requires_severity() -> None:
    """A classified runtime result must not finalize without two severity decisions."""
    payload = _scored_record_payload()
    payload.pop("severity_assessment")

    with pytest.raises(ValidationError, match="requires severity assessment"):
        RunRecord.model_validate(payload)


def test_run_record_rejects_structural_severity_tampering() -> None:
    """Profile, behavior, and reason changes must break terminal coherence."""
    mutations: list[dict[str, object]] = []

    wrong_profile_hash = _scored_record_payload()
    wrong_profile_hash["severity_assessment"]["candidate"]["profile_sha256"] = (
        "f" * 64
    )
    mutations.append(wrong_profile_hash)

    wrong_vector = _scored_record_payload()
    wrong_vector["severity_assessment"]["candidate"]["vector"] = (
        wrong_vector["severity_assessment"]["candidate"]["vector"].replace(
            "/VI:H/", "/VI:L/"
        )
    )
    mutations.append(wrong_vector)

    wrong_metric = _scored_record_payload()
    wrong_metric["severity_assessment"]["candidate"]["metrics"][6]["value"] = "L"
    mutations.append(wrong_metric)

    wrong_reason = _scored_record_payload()
    wrong_reason["severity_assessment"]["candidate"]["reason_code"] = (
        "tested_vulnerability_not_observed"
    )
    mutations.append(wrong_reason)

    vulnerable_not_scored = _scored_record_payload()
    original_candidate = vulnerable_not_scored["severity_assessment"]["candidate"]
    vulnerable_not_scored["severity_assessment"]["candidate"] = {
        "revision": original_candidate["revision"],
        "status": "not_scored",
        "reason_code": "tested_vulnerability_not_observed",
        "profile_id": None,
        "profile_sha256": None,
        "evidence_sha256": original_candidate["evidence_sha256"],
        "vector": None,
        "score": None,
        "severity": None,
        "metrics": [],
        "calculator": None,
        "review_status": "not_applicable",
    }
    mutations.append(vulnerable_not_scored)

    for payload in mutations:
        with pytest.raises(ValidationError):
            RunRecord.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"repetitions": 0},
        {
            "candidate": _observation("base-revision", secure=False),
        },
        {
            "candidate": _observation("candidate-revision", secure=True),
        },
        {
            "repetitions": 2,
            "candidate_differing_run_indexes": [2],
        },
        {
            "status": WorkflowStatus.NO_REGRESSION_OBSERVED,
            "reason_code": "candidate_regression_observed",
        },
        {
            "status": WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
            "reason_code": "candidate_regression_observed",
            "explanation": "A contradictory explanation.",
        },
        {
            "repetitions": 2,
            "stable": False,
            "status": WorkflowStatus.UNSTABLE_RESULT,
            "reason_code": "security_relevant_tuple_unstable",
            "explanation": (
                "Repeated security-relevant facts differed from run 1; one-based "
                "differing run indexes are reported separately for base and candidate."
            ),
            "base_differing_run_indexes": [1],
        },
    ],
)
def test_differential_evidence_rejects_incoherent_terminal_claims(
    changes: dict[str, object],
) -> None:
    payload = {**_evidence_payload(), **changes}

    with pytest.raises(ValidationError):
        DifferentialEvidence.model_validate(payload)


@pytest.mark.parametrize(
    ("started_at", "finished_at"),
    [
        (
            datetime(2026, 8, 7, 12, tzinfo=UTC).replace(tzinfo=None),
            datetime(2026, 8, 7, 13, tzinfo=UTC),
        ),
        (datetime(2026, 8, 7, 12, tzinfo=UTC), datetime(2026, 8, 7, 11, tzinfo=UTC)),
    ],
)
def test_run_record_requires_ordered_utc_timestamps(
    started_at: datetime,
    finished_at: datetime,
) -> None:
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="run-incoherent-time",
            environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
            base_revision="base-revision",
            candidate_revision="candidate-revision",
            status=WorkflowStatus.EXECUTION_INCONCLUSIVE,
            reason_code="execution_failed",
            explanation="Execution failed.",
            started_at=started_at,
            finished_at=finished_at,
            differential_evidence=None,
        )


def test_run_record_must_repeat_its_differential_conclusion_exactly() -> None:
    evidence = DifferentialEvidence.model_validate(_evidence_payload())
    started_at = datetime(2026, 8, 7, 12, tzinfo=UTC)

    with pytest.raises(ValidationError):
        RunRecord(
            run_id="run-incoherent-conclusion",
            environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
            base_revision="base-revision",
            candidate_revision="candidate-revision",
            status=WorkflowStatus.NO_REGRESSION_OBSERVED,
            reason_code="no_regression_observed",
            explanation="Both revisions were secure.",
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            differential_evidence=evidence,
        )


@pytest.mark.parametrize(
    ("status", "reason_code", "explanation"),
    [
        (
            WorkflowStatus.CANDIDATE_REGRESSION_OBSERVED,
            "candidate_regression_observed",
            (
                "The base denied unauthorized deletion and preserved the patient, "
                "while the candidate allowed deletion and removed the patient."
            ),
        ),
        (
            WorkflowStatus.CANDIDATE_FIX_OBSERVED,
            "candidate_fix_observed",
            (
                "The base allowed unauthorized deletion and removed the patient, "
                "while the candidate denied deletion and preserved the patient."
            ),
        ),
        (
            WorkflowStatus.NO_REGRESSION_OBSERVED,
            "no_regression_observed",
            "Both revisions denied unauthorized deletion and preserved the patient.",
        ),
        (
            WorkflowStatus.PRE_EXISTING_RISK_OBSERVED,
            "pre_existing_risk_observed",
            (
                "Both revisions allowed unauthorized deletion and removed the "
                "patient."
            ),
        ),
        (
            WorkflowStatus.UNSTABLE_RESULT,
            "security_relevant_tuple_unstable",
            (
                "Repeated security-relevant facts differed from run 1; one-based "
                "differing run indexes are reported separately for base and candidate."
            ),
        ),
    ],
)
def test_differential_terminal_status_requires_differential_evidence(
    status: WorkflowStatus,
    reason_code: str,
    explanation: str,
) -> None:
    started_at = datetime(2026, 8, 7, 12, tzinfo=UTC)

    with pytest.raises(ValidationError, match="differential evidence"):
        RunRecord(
            run_id="run-evidence-omitted",
            environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
            base_revision="base-revision",
            candidate_revision="candidate-revision",
            status=status,
            reason_code=reason_code,
            explanation=explanation,
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            differential_evidence=None,
        )


def test_classifier_inconclusive_terminal_reason_requires_evidence() -> None:
    started_at = datetime(2026, 8, 7, 12, tzinfo=UTC)

    with pytest.raises(ValidationError, match="differential evidence"):
        RunRecord(
            run_id="run-inconclusive-evidence-omitted",
            environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
            base_revision="base-revision",
            candidate_revision="candidate-revision",
            status=WorkflowStatus.EXECUTION_INCONCLUSIVE,
            reason_code="base_setup_failed",
            explanation=(
                "At least one base run did not complete setup; differential "
                "evidence is inconclusive."
            ),
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            differential_evidence=None,
        )


def test_run_record_requires_distinct_explicit_revision_identifiers() -> None:
    started_at = datetime(2026, 8, 7, 12, tzinfo=UTC)
    payload = {
        "run_id": "run-missing-revisions",
        "environment_kind": EnvironmentKind.CONTROLLED_FIXTURE,
        "status": WorkflowStatus.EXECUTION_INCONCLUSIVE,
        "reason_code": "execution_failed",
        "explanation": "Execution failed.",
        "started_at": started_at,
        "finished_at": started_at + timedelta(seconds=1),
        "differential_evidence": None,
    }

    with pytest.raises(ValidationError):
        RunRecord.model_validate(payload)

    with pytest.raises(ValidationError):
        RunRecord.model_validate(
            {
                **payload,
                "base_revision": "same-revision",
                "candidate_revision": "same-revision",
            }
        )


def test_terminal_run_record_requires_a_finished_timestamp() -> None:
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="run-unfinished",
            environment_kind=EnvironmentKind.CONTROLLED_FIXTURE,
            base_revision="base-revision",
            candidate_revision="candidate-revision",
            status=WorkflowStatus.EXECUTION_INCONCLUSIVE,
            reason_code="execution_failed",
            explanation="Execution failed.",
            started_at=datetime(2026, 8, 7, 12, tzinfo=UTC),
            differential_evidence=None,
        )
