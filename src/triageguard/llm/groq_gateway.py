"""Live Groq implementation with strict JSON Schema outputs and bounded retries."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from time import sleep as default_sleep
from typing import Any

from triageguard.config import Settings
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


class GroqRequestFailed(ModelGatewayError):
    """The live provider did not return a valid structured response."""


class GroqStructuredGateway:
    """Make a single strict structured-output request, retrying only transient failures."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = default_sleep,
    ) -> None:
        if settings.llm_provider != "groq":
            raise ValueError("GroqStructuredGateway requires llm_provider='groq'")
        if settings.llm_mode != "live":
            raise ValueError("GroqStructuredGateway requires live Settings")
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if client is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required for a live Groq gateway")
            from groq import Groq

            client = Groq(api_key=settings.groq_api_key)

        self._client = client
        self._model = settings.llm_model
        self._max_attempts = max_attempts
        self._sleep = sleep

    @property
    def provider(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Call Groq with strict schema enforcement and no content fallback path."""
        attempts: list[ModelAttempt] = []
        request_hash = prompt_sha256(request)
        canonical_request_hash = request_sha256(request)
        call_kwargs = {
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": canonical_json(request.payload)},
            ],
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "reasoning_effort": "medium",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.purpose,
                    "strict": True,
                    "schema": request.output_schema,
                },
            },
        }

        for number in range(1, self._max_attempts + 1):
            started_at = datetime.now(UTC)
            started_clock = perf_counter()
            try:
                completion = self._client.chat.completions.create(**call_kwargs)
            except Exception as error:
                finished_at = datetime.now(UTC)
                is_transient = _is_transient_groq_error(error)
                attempt = ModelAttempt(
                    number=number,
                    started_at=started_at,
                    finished_at=finished_at,
                    latency_ms=_elapsed_milliseconds(started_clock),
                    outcome="transient_error" if is_transient else "failed",
                    error_type=type(error).__name__,
                )
                attempts.append(attempt)
                if is_transient and number < self._max_attempts:
                    self._sleep(0.25 * number)
                    continue
                final_outcome = "transient_error" if is_transient else "failed"
                reason_code = (
                    "groq_transient_retries_exhausted"
                    if is_transient
                    else "groq_non_retryable_error"
                )
                raise GroqRequestFailed(
                    "Groq structured request failed",
                    attempts,
                    provenance=_failure_provenance(
                        model=self._model,
                        request=request,
                        prompt_hash=request_hash,
                        request_hash=canonical_request_hash,
                        attempts=attempts,
                        final_outcome=final_outcome,
                        reason_code=reason_code,
                        error_hash=error_sha256(error),
                    ),
                ) from error

            finished_at = datetime.now(UTC)
            latency_ms = _elapsed_milliseconds(started_clock)
            usage = getattr(completion, "usage", None)
            content: str | None = None
            try:
                content = completion.choices[0].message.content
                if not isinstance(content, str):
                    raise ModelOutputInvalid("Groq response did not contain text content")
                data = parse_and_validate_output(content, request.output_schema)
            except (AttributeError, IndexError, ModelOutputInvalid) as error:
                attempts.append(
                    ModelAttempt(
                        number=number,
                        started_at=started_at,
                        finished_at=finished_at,
                        latency_ms=latency_ms,
                        outcome="invalid_output",
                        error_type=type(error).__name__,
                    )
                )
                raise ModelOutputInvalid(
                    str(error),
                    attempts,
                    provenance=_failure_provenance(
                        model=self._model,
                        request=request,
                        prompt_hash=request_hash,
                        request_hash=canonical_request_hash,
                        attempts=attempts,
                        final_outcome="invalid_output",
                        reason_code="groq_invalid_output",
                        response_hash=response_sha256(content) if content is not None else None,
                        error_hash=error_sha256(error),
                        input_tokens=_nullable_usage_count(usage, "prompt_tokens"),
                        output_tokens=_nullable_usage_count(usage, "completion_tokens"),
                    ),
                ) from error

            attempts.append(
                ModelAttempt(
                    number=number,
                    started_at=started_at,
                    finished_at=finished_at,
                    latency_ms=latency_ms,
                    outcome="succeeded",
                )
            )
            return ModelResponse(
                data=data,
                provider="groq",
                model=self._model,
                latency_ms=sum(attempt.latency_ms for attempt in attempts),
                prompt_sha256=request_hash,
                response_sha256=response_sha256(content),
                input_tokens=_usage_count(usage, "prompt_tokens"),
                output_tokens=_usage_count(usage, "completion_tokens"),
                attempts=attempts,
            )

        raise AssertionError("bounded retry loop must return or raise")


def _usage_count(usage: Any, field: str) -> int:
    value = getattr(usage, field, 0) if usage is not None else 0
    return int(value or 0)


def _nullable_usage_count(usage: Any, field: str) -> int | None:
    return _usage_count(usage, field) if usage is not None else None


def _failure_provenance(
    *,
    model: str,
    request: ModelRequest,
    prompt_hash: str,
    request_hash: str,
    attempts: list[ModelAttempt],
    final_outcome: str,
    reason_code: str,
    response_hash: str | None = None,
    error_hash: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> ModelFailureProvenance:
    """Construct immutable, secret-free terminal failure provenance."""
    return ModelFailureProvenance(
        provider="groq",
        model=model,
        purpose=request.purpose,
        prompt_sha256=prompt_hash,
        request_sha256=request_hash,
        response_sha256=response_hash,
        error_sha256=error_hash,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=sum(attempt.latency_ms for attempt in attempts),
        attempts=tuple(attempts),
        final_outcome=final_outcome,
        reason_code=reason_code,
    )


def _elapsed_milliseconds(started_clock: float) -> int:
    return max(0, round((perf_counter() - started_clock) * 1000))


def _is_transient_groq_error(error: Exception) -> bool:
    """Match Groq's documented rate-limit and server-status retry conditions only."""
    status_code = getattr(error, "status_code", None)
    return status_code == 429 or isinstance(status_code, int) and 500 <= status_code <= 599
