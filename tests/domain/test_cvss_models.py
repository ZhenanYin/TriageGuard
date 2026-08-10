import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from triageguard.domain import CvssProfile

FIXTURE_ROOT = (
    Path(__file__).parents[2] / "fixtures" / "patient_delete_authorization"
)
EXPECTED_VECTOR = (
    "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:N/VI:H/VA:N/"
    "SC:N/SI:N/SA:N"
)
EXPECTED_METRICS = [
    "AV",
    "AC",
    "AT",
    "PR",
    "UI",
    "VC",
    "VI",
    "VA",
    "SC",
    "SI",
    "SA",
]


def _valid_payload() -> dict:
    return json.loads(
        (FIXTURE_ROOT / "cvss_profile.json").read_text(encoding="utf-8")
    )


def test_controlled_profile_is_complete_immutable_and_excludes_a_score() -> None:
    """A missing metric or handwritten score must invalidate the fixture."""
    profile = CvssProfile.model_validate(_valid_payload())

    assert profile.cvss_version == "4.0"
    assert profile.assessment_label == "expert_authored_provisional"
    assert profile.vector == EXPECTED_VECTOR
    assert [item.metric for item in profile.metrics] == EXPECTED_METRICS
    assert all(item.rationale for item in profile.metrics)
    assert all(item.source_references for item in profile.metrics)
    assert "score" not in profile.model_dump(mode="json")

    with pytest.raises(ValidationError, match="frozen"):
        profile.vector = "CVSS:4.0/AV:L"  # type: ignore[misc]


def test_profile_rejects_missing_or_duplicate_base_metrics() -> None:
    """Dropping or duplicating one Base metric must not produce a valid profile."""
    missing = _valid_payload()
    missing["metrics"] = [
        item for item in missing["metrics"] if item["metric"] != "VI"
    ]
    duplicate = _valid_payload()
    duplicate["metrics"].append(dict(duplicate["metrics"][3]))

    with pytest.raises(ValidationError, match="exactly the CVSS v4.0 Base metrics"):
        CvssProfile.model_validate(missing)
    with pytest.raises(ValidationError, match="exactly the CVSS v4.0 Base metrics"):
        CvssProfile.model_validate(duplicate)


def test_profile_rejects_vector_metric_mismatch_and_non_base_segments() -> None:
    """A vector that contradicts its evidence rows must not be accepted."""
    mismatch = _valid_payload()
    mismatch["vector"] = mismatch["vector"].replace("/PR:L/", "/PR:H/")
    threat = _valid_payload()
    threat["vector"] = f"{threat['vector']}/E:P"

    with pytest.raises(ValidationError, match="vector values must match"):
        CvssProfile.model_validate(mismatch)
    with pytest.raises(ValidationError, match="Base metrics only"):
        CvssProfile.model_validate(threat)


def test_profile_rejects_empty_provenance_and_unknown_fields() -> None:
    """Every judgment needs reviewable provenance and no hidden score input."""
    empty_rationale = _valid_payload()
    empty_rationale["metrics"][0]["rationale"] = ""
    empty_references = _valid_payload()
    empty_references["metrics"][0]["source_references"] = []
    unknown_category = _valid_payload()
    unknown_category["metrics"][0]["source_category"] = "llm_guess"
    handwritten_score = _valid_payload()
    handwritten_score["score"] = 7.1

    for payload in (empty_rationale, empty_references, unknown_category):
        with pytest.raises(ValidationError):
            CvssProfile.model_validate(payload)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CvssProfile.model_validate(handwritten_score)
