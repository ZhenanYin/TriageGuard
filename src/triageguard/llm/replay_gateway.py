"""Deterministic, fixture-only implementation of the structured model boundary."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from triageguard.llm.gateway import (
    ModelAttempt,
    ModelFailureProvenance,
    ModelGatewayError,
    ModelOutputInvalid,
    ModelRequest,
    ModelResponse,
    canonical_json,
    error_sha256,
    parse_and_validate_output,
    prompt_sha256,
    request_sha256,
    response_sha256,
)


class ReplayResponseMissing(ModelGatewayError):
    """No recorded fixture exists for the exact requested model purpose."""


class ReplayGateway:
    """Return only named prerecorded fixtures; never infer or synthesize output."""

    def __init__(
        self,
        responses: Mapping[str, dict[str, Any]],
        model: str = "replay/openai-gpt-oss-120b",
    ) -> None:
        self._responses = dict(responses)
        self._model = model

    @property
    def provider(self) -> str:
        return "replay"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Validate and return the fixture assigned to this exact request purpose."""
        started_at = datetime.now(UTC)
        started_clock = perf_counter()
        try:
            fixture = self._responses[request.purpose]
        except KeyError as error:
            finished_at = datetime.now(UTC)
            latency_ms = _elapsed_milliseconds(started_clock)
            attempt = ModelAttempt(
                number=1,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
                outcome="failed",
                error_type=type(error).__name__,
            )
            raise ReplayResponseMissing(
                f"no replay response is configured for purpose: {request.purpose}",
                [attempt],
                provenance=_failure_provenance(
                    request=request,
                    model=self._model,
                    attempt=attempt,
                    final_outcome="failed",
                    reason_code="replay_response_missing",
                    error_hash=error_sha256(error),
                ),
            ) from error

        content: str | None = None
        try:
            content = canonical_json(fixture)
            data = parse_and_validate_output(content, request.output_schema)
        except (ModelOutputInvalid, TypeError, ValueError) as error:
            finished_at = datetime.now(UTC)
            latency_ms = _elapsed_milliseconds(started_clock)
            attempt = ModelAttempt(
                number=1,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
                outcome="invalid_output",
                error_type=type(error).__name__,
            )
            raise ModelOutputInvalid(
                (
                    str(error)
                    if isinstance(error, ModelOutputInvalid)
                    else "replay response is not valid canonical JSON"
                ),
                [attempt],
                provenance=_failure_provenance(
                    request=request,
                    model=self._model,
                    attempt=attempt,
                    final_outcome="invalid_output",
                    reason_code="replay_invalid_output",
                    response_hash=(
                        response_sha256(content) if content is not None else None
                    ),
                    error_hash=error_sha256(error),
                ),
            ) from error

        finished_at = datetime.now(UTC)
        latency_ms = _elapsed_milliseconds(started_clock)
        return ModelResponse(
            data=data,
            provider="replay",
            model=self._model,
            latency_ms=latency_ms,
            prompt_sha256=prompt_sha256(request),
            response_sha256=response_sha256(content),
            input_tokens=0,
            output_tokens=0,
            attempts=[
                ModelAttempt(
                    number=1,
                    started_at=started_at,
                    finished_at=finished_at,
                    latency_ms=latency_ms,
                    outcome="succeeded",
                )
            ],
        )


def _elapsed_milliseconds(started_clock: float) -> int:
    return max(0, round((perf_counter() - started_clock) * 1000))


def _failure_provenance(
    *,
    request: ModelRequest,
    model: str,
    attempt: ModelAttempt,
    final_outcome: str,
    reason_code: str,
    response_hash: str | None = None,
    error_hash: str | None = None,
) -> ModelFailureProvenance:
    return ModelFailureProvenance(
        provider="replay",
        model=model,
        purpose=request.purpose,
        prompt_sha256=prompt_sha256(request),
        request_sha256=request_sha256(request),
        response_sha256=response_hash,
        error_sha256=error_hash,
        input_tokens=0,
        output_tokens=0,
        latency_ms=attempt.latency_ms,
        attempts=(attempt,),
        final_outcome=final_outcome,
        reason_code=reason_code,
    )
