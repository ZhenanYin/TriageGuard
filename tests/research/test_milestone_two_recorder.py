"""Tests for sealing terminal Milestone 2 research evidence."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from triageguard.domain import (
    MilestoneTwoRunRecord,
    MilestoneTwoStatus,
    PullRequestSnapshot,
)
from triageguard.research import ArtifactRecorder, RunOwnership, RunSealedError
from triageguard.research.recorder import LifecycleEventType


class UnrelatedModel(BaseModel):
    """A valid Pydantic model that must never become a terminal run record."""

    value: str


def _snapshot() -> PullRequestSnapshot:
    """Create one frozen, internally consistent OpenMRS Core snapshot."""
    return PullRequestSnapshot.from_identity(
        repository="openmrs/openmrs-core",
        pull_number=7312,
        pull_url="https://github.com/openmrs/openmrs-core/pull/7312",
        state="open",
        default_branch="master",
        base_branch="master",
        merge_base_sha="a" * 40,
        base_sha="b" * 40,
        head_sha="c" * 40,
        candidate_sha="d" * 40,
        merge_base_tree_sha="e" * 40,
        base_tree_sha="f" * 40,
        head_tree_sha="0" * 40,
        candidate_tree_sha="1" * 40,
        acquired_at=datetime(2026, 8, 14, tzinfo=UTC),
        github_api_version="2022-11-28",
        git_version="git version 2.45.0",
        acquisition_tool_version="0.1.0",
        analysis_config_sha256="2" * 64,
    )


def _failed_milestone_two_record() -> MilestoneTwoRunRecord:
    """Use a valid terminal failure record without inventing risk evidence."""
    return MilestoneTwoRunRecord(
        run_id="m2-run-1",
        snapshot=_snapshot(),
        status=MilestoneTwoStatus.FAILED,
        reason_code="model_output_invalid",
        explanation="The model response did not satisfy the required risk schema.",
        started_at=datetime(2026, 8, 14, tzinfo=UTC),
        finished_at=datetime(2026, 8, 14, 0, 0, 1, tzinfo=UTC),
    )


def test_recorder_finalizes_and_seals_milestone_two_record(tmp_path) -> None:
    """A terminal Milestone 2 record is durable and seals later lifecycle changes."""
    record = _failed_milestone_two_record()
    recorder = ArtifactRecorder(tmp_path)
    ownership = RunOwnership.issue(record.run_id)
    handle = recorder.start_run(record.run_id, ownership)

    terminal = recorder.finalize_run(handle, record)
    stored = recorder.read_artifact(handle, "run_record.json")

    assert stored.endswith(b"\n")
    assert terminal.sha256 == hashlib.sha256(stored).hexdigest()
    assert terminal.byte_count == len(stored)

    with pytest.raises(RunSealedError, match="sealed"):
        recorder.record_event(
            handle,
            LifecycleEventType.RISK_APPROVED,
            {"id": "risk-1", "risk_sha256": "a" * 64},
        )


def test_recorder_rejects_an_unrelated_pydantic_terminal_model(tmp_path) -> None:
    """The recorder accepts only the two explicitly supported terminal record types."""
    recorder = ArtifactRecorder(tmp_path)
    ownership = RunOwnership.issue("m2-run-1")
    handle = recorder.start_run("m2-run-1", ownership)

    with pytest.raises(TypeError, match="supported terminal record"):
        recorder.finalize_run(handle, UnrelatedModel(value="unsafe"))


def test_recorder_records_only_exact_milestone_two_approval_payloads(tmp_path) -> None:
    """Approval events carry only an ID and the exact hash of approved evidence."""
    recorder = ArtifactRecorder(tmp_path)
    ownership = RunOwnership.issue("m2-run-1")
    handle = recorder.start_run("m2-run-1", ownership)

    risk_event = recorder.record_event(
        handle,
        LifecycleEventType.RISK_APPROVED,
        {"id": "risk-1", "risk_sha256": "a" * 64},
    )
    gherkin_event = recorder.record_event(
        handle,
        LifecycleEventType.GHERKIN_APPROVED,
        {"id": "gherkin-1", "gherkin_sha256": "b" * 64},
    )

    assert risk_event.payload == {"id": "risk-1", "risk_sha256": "a" * 64}
    assert gherkin_event.payload == {
        "id": "gherkin-1",
        "gherkin_sha256": "b" * 64,
    }

    with pytest.raises(ValidationError, match="requires payload keys"):
        recorder.record_event(
            handle,
            LifecycleEventType.RISK_APPROVED,
            {
                "id": "risk-2",
                "risk_sha256": "c" * 64,
                "extra": "not-allowed",
            },
        )
