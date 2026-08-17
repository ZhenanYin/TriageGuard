import hashlib
from types import SimpleNamespace

import pytest

from triageguard.config import Settings
from triageguard.llm.gateway import ModelOutputInvalid, ModelRequest, canonical_json
from triageguard.llm.groq_gateway import GroqRequestFailed, GroqStructuredGateway
from triageguard.llm.replay_gateway import ReplayGateway, ReplayResponseMissing
from triageguard.llm.request_budget import ModelRequestTooLarge


def _request() -> ModelRequest:
    return ModelRequest(
        purpose="test_plan",
        system_prompt="Return a plan",
        payload={"contract_id": "c1"},
        output_schema={
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
            "additionalProperties": False,
        },
        max_output_tokens=1500,
    )


def test_replay_gateway_returns_named_fixture_and_records_model_identity():
    """A missing fixture or model identity would make offline evidence ambiguous."""
    gateway = ReplayGateway(
        responses={"test_plan": {"plan_id": "plan-1"}},
        model="replay/openai-gpt-oss-120b",
    )

    response = gateway.generate(_request())

    assert response.data == {"plan_id": "plan-1"}
    assert response.provider == "replay"
    assert response.model == "replay/openai-gpt-oss-120b"
    assert len(response.prompt_sha256) == 64
    assert (
        response.response_sha256 == hashlib.sha256(b'{"plan_id":"plan-1"}').hexdigest()
    )
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.attempts[0].outcome == "succeeded"


def test_model_request_accepts_gherkin_generation() -> None:
    """The second structured Milestone 2 call needs its own auditable purpose."""
    request = ModelRequest(
        purpose="gherkin_generation",
        system_prompt="Return one structured security scenario.",
        payload={"risk_sha256": "a" * 64},
        output_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        max_output_tokens=2048,
    )

    assert request.purpose == "gherkin_generation"


def test_model_request_accepts_testability_assessment() -> None:
    """The frozen-evidence testability call needs its own auditable purpose."""
    request = ModelRequest(
        purpose="testability_assessment",
        system_prompt="Assess testability using only frozen code evidence.",
        payload={"reviewed_risk_sha256": "a" * 64},
        output_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        max_output_tokens=2048,
    )

    assert request.purpose == "testability_assessment"


def test_replay_gateway_refuses_to_synthesize_a_missing_fixture():
    """Returning a guessed response would conceal an unrecorded replay dependency."""
    gateway = ReplayGateway(responses={})

    with pytest.raises(ReplayResponseMissing, match="test_plan") as error:
        gateway.generate(_request())

    provenance = error.value.provenance
    assert provenance.provider == "replay"
    assert provenance.model == "replay/openai-gpt-oss-120b"
    assert provenance.purpose == "test_plan"
    assert len(provenance.prompt_sha256) == 64
    assert len(provenance.request_sha256) == 64
    assert len(provenance.error_sha256) == 64
    assert provenance.response_sha256 is None
    assert provenance.final_outcome == "failed"
    assert provenance.reason_code == "replay_response_missing"
    assert [(attempt.number, attempt.outcome) for attempt in provenance.attempts] == [
        (1, "failed")
    ]


def test_replay_gateway_retains_attempt_metadata_for_an_invalid_fixture():
    """An invalid replay fixture is still a model attempt that later evidence must retain."""
    gateway = ReplayGateway(responses={"test_plan": {"wrong": "shape"}})

    with pytest.raises(ModelOutputInvalid) as error:
        gateway.generate(_request())

    assert [(attempt.number, attempt.outcome) for attempt in error.value.attempts] == [
        (1, "invalid_output")
    ]
    provenance = error.value.provenance
    assert provenance.provider == "replay"
    assert provenance.model == "replay/openai-gpt-oss-120b"
    assert provenance.purpose == "test_plan"
    assert len(provenance.prompt_sha256) == 64
    assert len(provenance.request_sha256) == 64
    assert len(provenance.response_sha256) == 64
    assert len(provenance.error_sha256) == 64
    assert provenance.final_outcome == "invalid_output"
    assert provenance.reason_code == "replay_invalid_output"
    assert provenance.attempts == error.value.attempts


def test_replay_gateway_attributes_a_nonserializable_recorded_fixture() -> None:
    """Even malformed in-memory replay data must fail with typed provenance."""
    gateway = ReplayGateway(  # type: ignore[arg-type]
        responses={"test_plan": {"plan_id": object()}}
    )

    with pytest.raises(ModelOutputInvalid) as error:
        gateway.generate(_request())

    provenance = error.value.provenance
    assert provenance.provider == "replay"
    assert provenance.purpose == "test_plan"
    assert provenance.response_sha256 is None
    assert provenance.error_sha256 is not None
    assert provenance.final_outcome == "invalid_output"
    assert provenance.reason_code == "replay_invalid_output"
    assert provenance.attempts[0].outcome == "invalid_output"


def test_live_gateway_uses_strict_groq_schema_and_returns_call_metadata():
    """Dropping strict schema arguments or response provenance breaks the live contract."""
    client = _FakeGroqClient([_completion('{"plan_id":"plan-1"}', 12, 5)])
    gateway = GroqStructuredGateway(_live_settings(), client=client)

    response = gateway.generate(_request())

    assert client.calls == [
        {
            "messages": [
                {"role": "system", "content": "Return a plan"},
                {"role": "user", "content": '{"contract_id":"c1"}'},
            ],
            "model": "openai/gpt-oss-120b",
            "max_tokens": 1500,
            "reasoning_effort": "medium",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_plan",
                    "strict": True,
                    "schema": _request().output_schema,
                },
            },
        }
    ]
    assert response.data == {"plan_id": "plan-1"}
    assert response.provider == "groq"
    assert response.model == "openai/gpt-oss-120b"
    assert response.input_tokens == 12
    assert response.output_tokens == 5
    assert response.latency_ms >= 0
    assert len(response.response_sha256) == 64
    assert response.attempts[0].outcome == "succeeded"


@pytest.mark.parametrize("content", ["not json", '{"wrong":"shape"}'])
def test_live_gateway_fails_explicitly_for_invalid_or_schema_incompatible_output(
    content,
):
    """An invalid model result must not enter the workflow through a fallback path."""
    gateway = GroqStructuredGateway(
        _live_settings(), client=_FakeGroqClient([_completion(content, 1, 1)])
    )

    with pytest.raises(ModelOutputInvalid):
        gateway.generate(_request())


def test_live_gateway_rejects_a_response_violating_any_json_schema_constraint():
    """Local validation must enforce more than object shape after a provider response."""
    request = _request().model_copy(
        update={
            "output_schema": {
                "type": "object",
                "properties": {"plan_id": {"type": "string", "minLength": 6}},
                "required": ["plan_id"],
            }
        }
    )
    gateway = GroqStructuredGateway(
        _live_settings(),
        client=_FakeGroqClient([_completion('{"plan_id":"id"}', 1, 1)]),
    )

    with pytest.raises(ModelOutputInvalid):
        gateway.generate(request)


def test_live_gateway_retries_only_a_bounded_transient_failure():
    """A rate limit may be retried, but it must retain evidence of every attempt."""
    client = _FakeGroqClient(
        [
            _TransientGroqError(429),
            _completion('{"plan_id":"plan-1"}', 12, 5),
        ]
    )
    gateway = GroqStructuredGateway(
        _live_settings(), client=client, max_attempts=2, sleep=lambda _: None
    )

    response = gateway.generate(_request())

    assert len(client.calls) == 2
    assert [(attempt.number, attempt.outcome) for attempt in response.attempts] == [
        (1, "transient_error"),
        (2, "succeeded"),
    ]


def test_live_gateway_exposes_immutable_provenance_after_exhausted_retries():
    """Retry exhaustion must leave a recorder-ready, secret-free invocation record."""
    gateway = GroqStructuredGateway(
        _live_settings(),
        client=_FakeGroqClient([_TransientGroqError(429), _TransientGroqError(500)]),
        max_attempts=2,
        sleep=lambda _: None,
    )

    with pytest.raises(GroqRequestFailed) as error:
        gateway.generate(_request())

    provenance = error.value.provenance
    assert provenance.provider == "groq"
    assert provenance.model == "openai/gpt-oss-120b"
    assert len(provenance.prompt_sha256) == 64
    assert len(provenance.request_sha256) == 64
    assert provenance.response_sha256 is None
    assert len(provenance.error_sha256) == 64
    assert provenance.input_tokens is None
    assert provenance.output_tokens is None
    assert provenance.final_outcome == "transient_error"
    assert provenance.reason_code == "groq_transient_retries_exhausted"
    assert [(attempt.number, attempt.outcome) for attempt in provenance.attempts] == [
        (1, "transient_error"),
        (2, "transient_error"),
    ]
    assert "test-key" not in provenance.model_dump_json()


def test_live_gateway_exposes_provenance_for_a_non_retryable_error():
    """A 4xx failure must retain provenance while stopping after its first attempt."""
    client = _FakeGroqClient([_TransientGroqError(400)])
    gateway = GroqStructuredGateway(
        _live_settings(), client=client, max_attempts=3, sleep=lambda _: None
    )

    with pytest.raises(GroqRequestFailed) as error:
        gateway.generate(_request())

    provenance = error.value.provenance
    assert len(client.calls) == 1
    assert provenance.final_outcome == "failed"
    assert provenance.reason_code == "groq_non_retryable_error"
    assert [(attempt.number, attempt.outcome) for attempt in provenance.attempts] == [
        (1, "failed")
    ]
    assert provenance.attempts[0].status_code == 400
    assert provenance.attempts[0].request_body_bytes == len(
        canonical_json(client.calls[0]).encode("utf-8")
    )
    assert provenance.error_sha256 is not None


def test_live_gateway_records_only_a_numeric_provider_body_limit() -> None:
    """A request-size failure may retain its numeric cap but never raw error text."""
    client = _FakeGroqClient([_GroqErrorWithBodyLimit(413, "8 KB")])
    gateway = GroqStructuredGateway(
        _live_settings(), client=client, max_attempts=1, sleep=lambda _: None
    )

    with pytest.raises(GroqRequestFailed) as error:
        gateway.generate(_request())

    attempt = error.value.provenance.attempts[0]
    assert attempt.status_code == 413
    assert attempt.provider_body_limit_bytes == 8_192
    assert "8 KB" not in error.value.provenance.model_dump_json()


def test_live_gateway_rejects_an_oversized_body_before_calling_groq() -> None:
    """The local byte boundary must prevent known-oversized provider calls."""
    client = _FakeGroqClient([_completion('{"plan_id":"plan-1"}', 1, 1)])
    gateway = GroqStructuredGateway(
        _live_settings(max_model_request_bytes=300),
        client=client,
        max_attempts=3,
    )

    with pytest.raises(ModelRequestTooLarge, match="declared provider budget") as error:
        gateway.generate(_request())

    assert client.calls == []
    provenance = error.value.provenance
    assert provenance.reason_code == "model_request_too_large"
    assert provenance.final_outcome == "failed"
    assert len(provenance.attempts) == 1
    attempt = provenance.attempts[0]
    assert attempt.error_type == "ModelRequestTooLarge"
    assert attempt.status_code is None
    assert attempt.request_body_bytes > 300
    assert attempt.provider_body_limit_bytes == 300
    serialized = provenance.model_dump_json()
    assert "test-key" not in serialized
    assert "contract_id" not in serialized


def test_live_gateway_exposes_response_provenance_for_invalid_model_output():
    """A terminal parse failure must retain the raw-response hash and token usage."""
    gateway = GroqStructuredGateway(
        _live_settings(), client=_FakeGroqClient([_completion("not json", 7, 3)])
    )

    with pytest.raises(ModelOutputInvalid) as error:
        gateway.generate(_request())

    provenance = error.value.provenance
    assert len(provenance.response_sha256) == 64
    assert len(provenance.error_sha256) == 64
    assert provenance.input_tokens == 7
    assert provenance.output_tokens == 3
    assert provenance.final_outcome == "invalid_output"
    assert provenance.reason_code == "groq_invalid_output"


def test_live_gateway_keeps_missing_usage_nullable_in_failure_provenance():
    """Absent provider usage must not be reported as a measured zero-token request."""
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
    )
    gateway = GroqStructuredGateway(
        _live_settings(), client=_FakeGroqClient([completion])
    )

    with pytest.raises(ModelOutputInvalid) as error:
        gateway.generate(_request())

    assert error.value.provenance.input_tokens is None
    assert error.value.provenance.output_tokens is None


def _live_settings(**changes: object) -> Settings:
    return Settings(llm_mode="live", groq_api_key="test-key", **changes)


class _FakeGroqClient:
    def __init__(self, outcomes):
        self._outcomes = iter(outcomes)
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _TransientGroqError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"transient status {status_code}")


class _GroqErrorWithBodyLimit(_TransientGroqError):
    def __init__(self, status_code: int, limit: str):
        super().__init__(status_code)
        self.response = SimpleNamespace(
            json=lambda: {"error": {"message": f"Maximum request size is {limit}"}}
        )


def _completion(content: str, prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )
