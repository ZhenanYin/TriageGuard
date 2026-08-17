"""Cross-stage tests for durable, secret-safe structured model execution."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from triageguard.evidence import ModelEvidenceEnvelope
from triageguard.llm import (
    ModelAttempt,
    ModelFailureProvenance,
    ModelGatewayError,
    ModelRequest,
    ModelResponse,
    ModelStageRunner,
)
from triageguard.llm.gateway import prompt_sha256, request_sha256
from triageguard.llm.request_budget import groq_request_body_bytes
from triageguard.provenance import canonical_json, canonical_sha256
from triageguard.research import ArtifactRecorder, RunOwnership

STAGES = (
    "risk_hypothesis",
    "testability_assessment",
    "gherkin_generation",
)
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _request_and_envelope(stage: str) -> tuple[ModelRequest, ModelEvidenceEnvelope]:
    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": False,
    }
    envelope = ModelEvidenceEnvelope.from_content(
        stage=stage,
        snapshot_key="a" * 64,
        context_sha256="b" * 64,
        comparison_bindings=(),
        input_bindings=(),
        visible_anchors=(),
        omitted_anchors=(),
        catalog_anchor_ids=(),
        max_request_body_bytes=8_192,
        selection_policy_version="stage-runner-test-v1",
        output_schema_sha256=canonical_sha256(schema),
    )
    request = ModelRequest(
        purpose=stage,
        system_prompt="Return one bounded structured result.",
        payload={
            "evidence_envelope": envelope.model_dump(mode="json"),
        },
        output_schema=schema,
        max_output_tokens=128,
    )
    return request, envelope


class _SuccessfulGateway:
    provider = "groq"
    model = "openai/gpt-oss-120b"

    def __init__(self, recorder: ArtifactRecorder, handle, stage: str) -> None:
        self._recorder = recorder
        self._handle = handle
        self._stage = stage
        self.call_count = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        # This read is the observable ordering contract: invocation cannot begin
        # until the exact visibility boundary is durable.
        saved = json.loads(
            self._recorder.read_artifact(
                self._handle,
                f"artifacts/model_evidence/{self._stage}.json",
            )
        )
        assert (
            saved["envelope_sha256"]
            == (request.payload["evidence_envelope"]["envelope_sha256"])
        )
        self.call_count += 1
        content = canonical_json({"result": "ok"})
        return ModelResponse(
            data={"result": "ok"},
            provider=self.provider,
            model=self.model,
            latency_ms=1,
            prompt_sha256=prompt_sha256(request),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            input_tokens=10,
            output_tokens=2,
            attempts=[
                ModelAttempt(
                    number=1,
                    started_at=NOW,
                    finished_at=NOW,
                    latency_ms=1,
                    outcome="succeeded",
                )
            ],
        )


class _FailingGateway:
    provider = "groq"
    model = "openai/gpt-oss-120b"

    def __init__(self, raw_error: str) -> None:
        self.raw_error = raw_error
        self.call_count = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        attempt = ModelAttempt(
            number=1,
            started_at=NOW,
            finished_at=NOW,
            latency_ms=2,
            outcome="failed",
            error_type="ProviderRejected",
            status_code=413,
        )
        provenance = ModelFailureProvenance(
            provider=self.provider,
            model=self.model,
            purpose=request.purpose,
            prompt_sha256=prompt_sha256(request),
            request_sha256=request_sha256(request),
            error_sha256=hashlib.sha256(self.raw_error.encode("utf-8")).hexdigest(),
            latency_ms=2,
            attempts=(attempt,),
            final_outcome="failed",
            reason_code="provider_rejected",
        )
        raise ModelGatewayError(
            self.raw_error,
            [attempt],
            provenance=provenance,
        )


class _InterruptAfterResponseRecorder(ArtifactRecorder):
    def __init__(self, root_directory) -> None:
        super().__init__(root_directory)
        self.interrupt = True

    def write_artifact(self, handle, name, content, provenance):
        result = super().write_artifact(handle, name, content, provenance)
        if self.interrupt and "/model_responses/" in name:
            self.interrupt = False
            raise OSError("simulated interruption after durable response")
        return result


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runner_persists_envelope_before_call_and_response_after(
    tmp_path,
    stage: str,
) -> None:
    """Moving the invocation before envelope storage breaks evidence ordering."""
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(
        f"runner-success-{stage}", RunOwnership.issue(f"runner-success-{stage}")
    )
    request, envelope = _request_and_envelope(stage)
    gateway = _SuccessfulGateway(recorder, handle, stage)

    result = ModelStageRunner(recorder).run(
        run_handle=handle,
        envelope=envelope,
        request=request,
        gateway=gateway,
    )

    assert result.envelope == envelope
    assert result.request == request
    assert result.response.data == {"result": "ok"}
    assert gateway.call_count == 1
    saved = json.loads(
        recorder.read_artifact(
            handle,
            f"artifacts/model_responses/{stage}.json",
        )
    )
    assert saved["stage"] == stage
    assert saved["evidence_envelope_sha256"] == envelope.envelope_sha256
    assert saved["request_sha256"] == request_sha256(request)
    assert 0 < saved["request_body_bytes"] <= saved["max_request_body_bytes"]
    assert saved["max_request_body_bytes"] == envelope.max_request_body_bytes


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runner_persists_safe_stage_keyed_failure_and_allows_retry(
    tmp_path,
    stage: str,
) -> None:
    """A stage-specific failure must be attributable without retaining secrets."""
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(
        f"runner-failure-{stage}", RunOwnership.issue(f"runner-failure-{stage}")
    )
    request, envelope = _request_and_envelope(stage)
    raw_error = "raw provider error with secret-token and prompt payload"
    failing_gateway = _FailingGateway(raw_error)
    runner = ModelStageRunner(recorder)

    with pytest.raises(ModelGatewayError, match="raw provider error"):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=request,
            gateway=failing_gateway,
        )

    saved_bytes = recorder.read_artifact(
        handle,
        f"artifacts/model_failures/{stage}.json",
    )
    saved = json.loads(saved_bytes)
    assert saved["stage"] == stage
    assert saved["failure"]["purpose"] == stage
    assert saved["request_sha256"] == request_sha256(request)
    assert 0 < saved["request_body_bytes"] <= saved["max_request_body_bytes"]
    assert saved["max_request_body_bytes"] == envelope.max_request_body_bytes
    assert (
        saved["failure"]["attempts"][0]["request_body_bytes"]
        == (saved["request_body_bytes"])
    )
    assert saved["failure"]["attempts"][0]["provider_body_limit_bytes"] is None
    assert raw_error.encode("utf-8") not in saved_bytes
    assert b"secret-token" not in saved_bytes
    assert b"prompt payload" not in saved_bytes
    assert not (
        tmp_path
        / f"runner-failure-{stage}"
        / "artifacts"
        / "model_responses"
        / f"{stage}.json"
    ).exists()

    successful_gateway = _SuccessfulGateway(recorder, handle, stage)
    result = runner.run(
        run_handle=handle,
        envelope=envelope,
        request=request,
        gateway=successful_gateway,
    )

    assert result.response.data == {"result": "ok"}
    assert successful_gateway.call_count == 1


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runner_reuses_response_written_before_interruption(
    tmp_path,
    stage: str,
) -> None:
    """A durable successful response must prevent a duplicate provider call."""
    recorder = _InterruptAfterResponseRecorder(tmp_path)
    handle = recorder.start_run(
        f"runner-resume-{stage}", RunOwnership.issue(f"runner-resume-{stage}")
    )
    request, envelope = _request_and_envelope(stage)
    gateway = _SuccessfulGateway(recorder, handle, stage)
    runner = ModelStageRunner(recorder)

    with pytest.raises(OSError, match="simulated interruption"):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=request,
            gateway=gateway,
        )

    resumed = runner.run(
        run_handle=handle,
        envelope=envelope,
        request=request,
        gateway=gateway,
    )

    assert resumed.response.data == {"result": "ok"}
    assert gateway.call_count == 1


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runner_rejects_a_saved_response_for_a_different_request(
    tmp_path,
    stage: str,
) -> None:
    """Changing request bytes while reusing a response would break attribution."""
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(
        f"runner-mismatch-{stage}", RunOwnership.issue(f"runner-mismatch-{stage}")
    )
    request, envelope = _request_and_envelope(stage)
    gateway = _SuccessfulGateway(recorder, handle, stage)
    runner = ModelStageRunner(recorder)
    runner.run(
        run_handle=handle,
        envelope=envelope,
        request=request,
        gateway=gateway,
    )
    changed_request = request.model_copy(
        update={
            "payload": {
                **request.payload,
                "output_rule": "A changed request cannot reuse prior output.",
            }
        }
    )

    with pytest.raises(ValueError, match="does not match the current request"):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=changed_request,
            gateway=gateway,
        )

    assert gateway.call_count == 1


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runner_keeps_first_failure_without_masking_a_failed_retry(
    tmp_path,
    stage: str,
) -> None:
    """Immutable provenance must not replace the actual error from a later retry."""
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(
        f"runner-repeat-failure-{stage}",
        RunOwnership.issue(f"runner-repeat-failure-{stage}"),
    )
    request, envelope = _request_and_envelope(stage)
    runner = ModelStageRunner(recorder)
    first_gateway = _FailingGateway("first provider rejection")
    second_gateway = _FailingGateway("second provider rejection")

    with pytest.raises(ModelGatewayError, match="first provider rejection"):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=request,
            gateway=first_gateway,
        )
    first_saved = recorder.read_artifact(
        handle,
        f"artifacts/model_failures/{stage}.json",
    )

    with pytest.raises(ModelGatewayError, match="second provider rejection"):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=request,
            gateway=second_gateway,
        )

    assert (
        recorder.read_artifact(
            handle,
            f"artifacts/model_failures/{stage}.json",
        )
        == first_saved
    )
    assert first_gateway.call_count == 1
    assert second_gateway.call_count == 1


@pytest.mark.parametrize("stage", STAGES)
def test_stage_runner_rejects_changed_request_after_a_saved_failure(
    tmp_path,
    stage: str,
) -> None:
    """A retry cannot silently change the request bound to durable failure evidence."""
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(
        f"runner-failure-mismatch-{stage}",
        RunOwnership.issue(f"runner-failure-mismatch-{stage}"),
    )
    request, envelope = _request_and_envelope(stage)
    runner = ModelStageRunner(recorder)
    with pytest.raises(ModelGatewayError):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=request,
            gateway=_FailingGateway("first provider rejection"),
        )
    changed_request = request.model_copy(
        update={
            "payload": {
                **request.payload,
                "output_rule": "This retry changed immutable request bytes.",
            }
        }
    )
    gateway = _SuccessfulGateway(recorder, handle, stage)

    with pytest.raises(ValueError, match="failure does not match"):
        runner.run(
            run_handle=handle,
            envelope=envelope,
            request=changed_request,
            gateway=gateway,
        )

    assert gateway.call_count == 0


def test_stage_runner_measures_replay_against_the_configured_provider_model(
    tmp_path,
) -> None:
    """Replay identity must not change the exact body size selected for Groq."""
    stage = "risk_hypothesis"
    recorder = ArtifactRecorder(tmp_path)
    handle = recorder.start_run(
        "runner-replay-measurement",
        RunOwnership.issue("runner-replay-measurement"),
    )
    request, envelope = _request_and_envelope(stage)
    gateway = _SuccessfulGateway(recorder, handle, stage)
    gateway.model = "replay/a-deliberately-long-provider-model-identity"
    configured_model = "openai/gpt-oss-120b"

    ModelStageRunner(recorder, request_model=configured_model).run(
        run_handle=handle,
        envelope=envelope,
        request=request,
        gateway=gateway,
    )

    saved = json.loads(
        recorder.read_artifact(
            handle,
            "artifacts/model_responses/risk_hypothesis.json",
        )
    )
    assert saved["request_body_bytes"] == groq_request_body_bytes(
        request=request,
        model=configured_model,
    )
