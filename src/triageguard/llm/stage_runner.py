"""Durable, stage-agnostic execution of one evidence-bound model request."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from triageguard.evidence import ModelEvidenceEnvelope, ModelEvidenceStage
from triageguard.llm.gateway import (
    ModelAttempt,
    ModelFailureProvenance,
    ModelGatewayError,
    ModelRequest,
    ModelResponse,
    StructuredModelGateway,
    error_sha256,
    prompt_sha256,
    request_sha256,
)
from triageguard.llm.request_budget import (
    ModelRequestTooLarge,
    groq_request_body_bytes,
)
from triageguard.provenance import canonical_json, canonical_sha256
from triageguard.research import ArtifactRecorder, RunHandle
from triageguard.research.recorder import ArtifactWriteJournal, TransformationEvent


@dataclass(frozen=True)
class ModelStageResult:
    """One bound request and its durable structured response."""

    envelope: ModelEvidenceEnvelope
    request: ModelRequest
    response: ModelResponse


class _ModelStageResponseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ModelEvidenceStage
    evidence_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_body_bytes: int = Field(gt=0)
    max_request_body_bytes: int = Field(gt=0)
    response: ModelResponse

    @model_validator(mode="after")
    def validate_size(self) -> _ModelStageResponseRecord:
        if self.request_body_bytes > self.max_request_body_bytes:
            raise ValueError("saved model response exceeds its declared byte limit")
        return self


class _ModelStageFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ModelEvidenceStage
    evidence_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_body_bytes: int = Field(gt=0)
    max_request_body_bytes: int = Field(gt=0)
    failure: ModelFailureProvenance

    @model_validator(mode="after")
    def validate_failure_binding(self) -> _ModelStageFailureRecord:
        if (
            self.failure.purpose != self.stage
            or self.failure.request_sha256 != self.request_sha256
        ):
            raise ValueError("saved model failure does not bind its stage request")
        return self


class ModelStageRunner:
    """Persist, invoke, and recover a model call without interpreting its data."""

    def __init__(
        self,
        recorder: ArtifactRecorder,
        *,
        artifact_name: Callable[[str], str] | None = None,
        request_model: str | None = None,
    ) -> None:
        self._recorder = recorder
        self._artifact_name = artifact_name or (lambda name: name)
        self._request_model = request_model

    def run(
        self,
        *,
        run_handle: RunHandle,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        gateway: StructuredModelGateway,
    ) -> ModelStageResult:
        """Return one durable response after enforcing exact binding and ordering."""
        request_body_bytes = self._validate_request(
            envelope=envelope,
            request=request,
            gateway=gateway,
        )
        self._ensure_envelope(
            run_handle=run_handle,
            envelope=envelope,
        )

        saved_response = self._load_response(
            run_handle=run_handle,
            envelope=envelope,
            request=request,
            gateway=gateway,
            request_body_bytes=request_body_bytes,
        )
        if saved_response is not None:
            return ModelStageResult(
                envelope=envelope,
                request=request,
                response=saved_response,
            )

        self._validate_existing_failure(
            run_handle=run_handle,
            envelope=envelope,
            request=request,
            request_body_bytes=request_body_bytes,
        )

        if request_body_bytes > envelope.max_request_body_bytes:
            error = self._oversized_error(
                envelope=envelope,
                request=request,
                gateway=gateway,
                request_body_bytes=request_body_bytes,
            )
            self._ensure_failure(
                run_handle=run_handle,
                envelope=envelope,
                request=request,
                request_body_bytes=request_body_bytes,
                failure=error.provenance,
            )
            raise error

        try:
            response = gateway.generate(request)
        except ModelGatewayError as error:
            if error.provenance is not None:
                failure = self._normalize_failure(
                    failure=error.provenance,
                    envelope=envelope,
                    request=request,
                    gateway=gateway,
                    request_body_bytes=request_body_bytes,
                )
                self._ensure_failure(
                    run_handle=run_handle,
                    envelope=envelope,
                    request=request,
                    request_body_bytes=request_body_bytes,
                    failure=failure,
                )
                error.provenance = failure
                error.attempts = failure.attempts
            raise

        self._validate_response(
            response=response,
            request=request,
            gateway=gateway,
        )
        record = _ModelStageResponseRecord(
            stage=envelope.stage,
            evidence_envelope_sha256=envelope.envelope_sha256,
            request_sha256=request_sha256(request),
            request_body_bytes=request_body_bytes,
            max_request_body_bytes=envelope.max_request_body_bytes,
            response=response,
        )
        self._ensure_record(
            run_handle=run_handle,
            artifact_name=f"artifacts/model_responses/{envelope.stage}.json",
            event_type=f"model_stage_{envelope.stage}_response_recorded",
            payload=record.model_dump(mode="json"),
            input_hashes={
                "snapshot": envelope.snapshot_key,
                "context": envelope.context_sha256,
                "evidence_envelope": envelope.envelope_sha256,
                "request": request_sha256(request),
                "response": response.response_sha256,
            },
            reason_code="model_stage_response_recorded",
        )
        return ModelStageResult(
            envelope=envelope,
            request=request,
            response=response,
        )

    def _validate_request(
        self,
        *,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        gateway: StructuredModelGateway,
    ) -> int:
        if request.purpose != envelope.stage:
            raise ValueError("model request purpose must match the evidence stage")
        if canonical_sha256(request.output_schema) != envelope.output_schema_sha256:
            raise ValueError("model request schema must match the evidence envelope")
        request_envelope = request.payload.get("evidence_envelope")
        if not isinstance(request_envelope, Mapping):
            raise TypeError("model request must contain its evidence envelope")
        normalized = ModelEvidenceEnvelope.model_validate(dict(request_envelope))
        if normalized != envelope:
            raise ValueError("model request evidence must match the durable envelope")
        return groq_request_body_bytes(
            request=request,
            model=self._request_model or gateway.model,
        )

    def _ensure_envelope(
        self,
        *,
        run_handle: RunHandle,
        envelope: ModelEvidenceEnvelope,
    ) -> None:
        self._ensure_record(
            run_handle=run_handle,
            artifact_name=f"artifacts/model_evidence/{envelope.stage}.json",
            event_type=f"model_stage_{envelope.stage}_evidence_recorded",
            payload=envelope.model_dump(mode="json"),
            input_hashes={
                "snapshot": envelope.snapshot_key,
                "context": envelope.context_sha256,
                "evidence_envelope": envelope.envelope_sha256,
            },
            reason_code="model_stage_evidence_recorded",
        )

    def _ensure_failure(
        self,
        *,
        run_handle: RunHandle,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        request_body_bytes: int,
        failure: ModelFailureProvenance | None,
    ) -> None:
        if failure is None:
            return
        artifact_name = f"artifacts/model_failures/{envelope.stage}.json"
        if self._validate_existing_failure(
            run_handle=run_handle,
            envelope=envelope,
            request=request,
            request_body_bytes=request_body_bytes,
        ):
            return
        record = _ModelStageFailureRecord(
            stage=envelope.stage,
            evidence_envelope_sha256=envelope.envelope_sha256,
            request_sha256=request_sha256(request),
            request_body_bytes=request_body_bytes,
            max_request_body_bytes=envelope.max_request_body_bytes,
            failure=failure,
        )
        self._ensure_record(
            run_handle=run_handle,
            artifact_name=artifact_name,
            event_type=f"model_stage_{envelope.stage}_failure_recorded",
            payload=record.model_dump(mode="json"),
            input_hashes={
                "snapshot": envelope.snapshot_key,
                "context": envelope.context_sha256,
                "evidence_envelope": envelope.envelope_sha256,
                "request": request_sha256(request),
                "failure": canonical_sha256(failure.model_dump(mode="json")),
            },
            reason_code=failure.reason_code,
        )

    def _validate_existing_failure(
        self,
        *,
        run_handle: RunHandle,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        request_body_bytes: int,
    ) -> bool:
        artifact_name = f"artifacts/model_failures/{envelope.stage}.json"
        existing = self._load_verified_record(run_handle, artifact_name)
        if existing is None:
            return False
        saved = _ModelStageFailureRecord.model_validate(existing)
        if (
            saved.stage != envelope.stage
            or saved.evidence_envelope_sha256 != envelope.envelope_sha256
            or saved.request_sha256 != request_sha256(request)
            or saved.request_body_bytes != request_body_bytes
            or saved.max_request_body_bytes != envelope.max_request_body_bytes
        ):
            raise ValueError("saved model failure does not match the current request")
        return True

    def _load_response(
        self,
        *,
        run_handle: RunHandle,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        gateway: StructuredModelGateway,
        request_body_bytes: int,
    ) -> ModelResponse | None:
        payload = self._load_verified_record(
            run_handle,
            f"artifacts/model_responses/{envelope.stage}.json",
        )
        if payload is None:
            return None
        record = _ModelStageResponseRecord.model_validate(payload)
        if (
            record.stage != envelope.stage
            or record.evidence_envelope_sha256 != envelope.envelope_sha256
            or record.request_sha256 != request_sha256(request)
            or record.request_body_bytes != request_body_bytes
            or record.max_request_body_bytes != envelope.max_request_body_bytes
        ):
            raise ValueError("saved model response does not match the current request")
        self._validate_response(
            response=record.response,
            request=request,
            gateway=gateway,
        )
        return record.response

    @staticmethod
    def _validate_response(
        *,
        response: ModelResponse,
        request: ModelRequest,
        gateway: StructuredModelGateway,
    ) -> None:
        if (
            response.provider != gateway.provider
            or response.model != gateway.model
            or response.prompt_sha256 != prompt_sha256(request)
        ):
            raise ValueError("model response provenance does not match the request")

    @staticmethod
    def _normalize_failure(
        *,
        failure: ModelFailureProvenance,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        gateway: StructuredModelGateway,
        request_body_bytes: int,
    ) -> ModelFailureProvenance:
        if (
            failure.provider != gateway.provider
            or failure.model != gateway.model
            or failure.purpose != envelope.stage
            or failure.prompt_sha256 != prompt_sha256(request)
            or failure.request_sha256 != request_sha256(request)
        ):
            raise ValueError("model failure provenance does not match the request")
        if any(
            attempt.request_body_bytes is not None
            and attempt.request_body_bytes != request_body_bytes
            for attempt in failure.attempts
        ):
            raise ValueError("model failure attempt size does not match the request")
        attempts = tuple(
            attempt
            if attempt.request_body_bytes is not None
            else attempt.model_copy(update={"request_body_bytes": request_body_bytes})
            for attempt in failure.attempts
        )
        return failure.model_copy(update={"attempts": attempts})

    @staticmethod
    def _oversized_error(
        *,
        envelope: ModelEvidenceEnvelope,
        request: ModelRequest,
        gateway: StructuredModelGateway,
        request_body_bytes: int,
    ) -> ModelRequestTooLarge:
        observed_at = datetime.now(UTC)
        local_error = ModelRequestTooLarge(
            "Model request exceeded the declared provider budget"
        )
        attempt = ModelAttempt(
            number=1,
            started_at=observed_at,
            finished_at=observed_at,
            latency_ms=0,
            outcome="failed",
            error_type=type(local_error).__name__,
            request_body_bytes=request_body_bytes,
        )
        provenance = ModelFailureProvenance(
            provider=gateway.provider,
            model=gateway.model,
            purpose=envelope.stage,
            prompt_sha256=prompt_sha256(request),
            request_sha256=request_sha256(request),
            error_sha256=error_sha256(local_error),
            latency_ms=0,
            attempts=(attempt,),
            final_outcome="failed",
            reason_code="model_request_too_large",
        )
        return ModelRequestTooLarge(
            str(local_error),
            [attempt],
            provenance=provenance,
        )

    def _ensure_record(
        self,
        *,
        run_handle: RunHandle,
        artifact_name: str,
        event_type: str,
        payload: Mapping[str, object],
        input_hashes: Mapping[str, str],
        reason_code: str,
    ) -> None:
        resolved_name = self._artifact_name(artifact_name)
        existing = self._load_verified_record(run_handle, artifact_name)
        normalized_payload = dict(payload)
        if existing is not None:
            if existing != normalized_payload:
                raise ValueError(
                    f"saved model-stage artifact conflicts: {resolved_name}"
                )
            return

        content = (canonical_json(normalized_payload) + "\n").encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        started_at = datetime.now(UTC)
        resolved_event_type = (
            f"{event_type}_"
            f"{hashlib.sha256(resolved_name.encode('utf-8')).hexdigest()[:12]}"
        )
        event = TransformationEvent(
            event_type=resolved_event_type,
            inputs={name: name for name in input_hashes},
            outputs={resolved_name: resolved_name},
            input_hashes=dict(input_hashes),
            output_hashes={resolved_name: digest},
            versions={
                "triageguard": "2.0.0",
                "model_stage_runner": "v1",
            },
            started_at=started_at,
            finished_at=started_at + timedelta(microseconds=1),
            reason_code=reason_code,
        )
        self._recorder.write_artifact(
            run_handle,
            resolved_name,
            content,
            event,
        )
        self._recorder.record_transformation(run_handle, event)

    def _load_verified_record(
        self,
        run_handle: RunHandle,
        artifact_name: str,
    ) -> dict[str, object] | None:
        resolved_name = self._artifact_name(artifact_name)
        try:
            content = self._recorder.read_artifact(run_handle, resolved_name)
        except FileNotFoundError:
            return None

        digest = hashlib.sha256(content).hexdigest()
        events = self._recorder.read_events(run_handle)
        started_payloads = [
            event.payload
            for event in events
            if event.event_type == "artifact_write_started"
            and event.payload.get("artifact_name") == resolved_name
        ]
        completed_payloads = [
            event.payload
            for event in events
            if event.event_type == "artifact_write_completed"
            and event.payload.get("artifact_name") == resolved_name
        ]
        if len(started_payloads) != 1 or len(completed_payloads) != 1:
            raise ValueError("model-stage artifact lacks one exact journal pair")
        started = ArtifactWriteJournal.model_validate(started_payloads[0])
        completed = ArtifactWriteJournal.model_validate(completed_payloads[0])
        if (
            started != completed
            or started.artifact_sha256 != digest
            or started.artifact_byte_count != len(content)
        ):
            raise ValueError("model-stage artifact does not match its journal")

        matching = [
            event
            for event in events
            if event.event_type == started.provenance.event_type
        ]
        if len(matching) > 1:
            raise ValueError("model-stage transformation is duplicated")
        if matching:
            transformation = TransformationEvent.model_validate(matching[0].payload)
            if transformation != started.provenance:
                raise ValueError("model-stage transformation contradicts its artifact")
        else:
            self._recorder.record_transformation(run_handle, started.provenance)

        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise TypeError("model-stage artifact must contain a JSON object")
        return payload
