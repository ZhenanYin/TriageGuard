from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from triageguard.domain.models import RuntimeObservation
from triageguard.runtime import (
    ObservationWriter,
    OpenMrsTestClient,
    RuntimeObservationEnvelope,
    TargetUnavailable,
)


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
        if self.payload is None:
            raise ValueError("response has no JSON body")
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _client(fake_session: FakeSession) -> OpenMrsTestClient:
    return OpenMrsTestClient(
        base_url="http://fixture.test/openmrs/",
        username="administrator",
        password="test-password",
        _session=fake_session,
    )


def _assert_finite_timeout(call: dict[str, Any]) -> None:
    timeout = call["timeout"]
    assert isinstance(timeout, (int, float))
    assert not isinstance(timeout, bool)
    assert math.isfinite(timeout)
    assert timeout > 0


def test_client_uses_only_bounded_openmrs_patient_paths() -> None:
    """A wrong verb, path, actor, or missing timeout would escape the test boundary."""
    fake_session = FakeSession(
        [
            FakeResponse(201, {"uuid": "patient/id"}),
            FakeResponse(200, {"authenticated": True}),
            FakeResponse(403),
            FakeResponse(200, {"uuid": "patient/id"}),
            FakeResponse(204),
        ]
    )
    client = _client(fake_session)

    patient_id = client.create_patient()
    actor_session = client.login_as_actor("clerk")
    delete_status = client.delete_patient(patient_id, actor_session)
    patient_exists = client.read_patient(patient_id, actor_session)
    cleanup_complete = client.authorized_cleanup_patient(patient_id)

    assert patient_id == "patient/id"
    assert delete_status == 403
    assert patient_exists is True
    assert cleanup_complete is True
    assert [(call["method"], call["url"]) for call in fake_session.calls] == [
        ("POST", "http://fixture.test/openmrs/ws/rest/v1/patient"),
        ("GET", "http://fixture.test/openmrs/ws/rest/v1/session"),
        ("DELETE", "http://fixture.test/openmrs/ws/rest/v1/patient/patient%2Fid"),
        ("GET", "http://fixture.test/openmrs/ws/rest/v1/patient/patient%2Fid"),
        ("DELETE", "http://fixture.test/openmrs/ws/rest/v1/patient/patient%2Fid"),
    ]
    assert fake_session.calls[1]["auth"] == ("clerk", "test-password")
    assert fake_session.calls[2]["auth"] == ("clerk", "test-password")
    assert fake_session.calls[3]["auth"] == ("clerk", "test-password")
    assert fake_session.calls[4]["auth"] == ("administrator", "test-password")
    assert fake_session.calls[4]["params"] == {"purge": "true"}
    for call in fake_session.calls:
        _assert_finite_timeout(call)
        assert call["allow_redirects"] is False


def test_administrator_actor_uses_configured_credential_name() -> None:
    """The control actor must use the harness-supplied administrator identity."""
    fake_session = FakeSession([FakeResponse(200, {"authenticated": True})])
    client = OpenMrsTestClient(
        base_url="http://fixture.test/openmrs",
        username="fixture-admin",
        password="test-password",
        _session=fake_session,
    )

    actor_session = client.login_as_actor("administrator")

    assert isinstance(actor_session, str)
    assert actor_session == "fixture-admin"
    assert fake_session.calls[0]["auth"] == ("fixture-admin", "test-password")


def test_read_patient_maps_only_not_found_to_nonexistence() -> None:
    """A post-delete 404 is an observed absence, not a transport failure."""
    fake_session = FakeSession(
        [
            FakeResponse(200, {"authenticated": True}),
            FakeResponse(404),
        ]
    )
    client = _client(fake_session)
    actor_session = client.login_as_actor("clerk")

    assert client.read_patient("missing", actor_session) is False


def test_transport_failure_raises_target_unavailable() -> None:
    """A connection failure must not be transformed into secure behavior."""
    client = _client(FakeSession([requests.ConnectionError("fixture refused")]))

    with pytest.raises(TargetUnavailable, match="OpenMRS target request failed"):
        client.create_patient()


def test_observation_writer_appends_raw_facts_without_classification(
    tmp_path: Path,
) -> None:
    """Generated record calls must survive subprocess exit as unclassified events."""
    observation_path = tmp_path / "observation.json"
    writer = ObservationWriter(observation_path)

    writer.record_http_status(403)
    writer.record_patient_exists(True)

    assert not observation_path.exists()
    events = [
        json.loads(line)
        for line in writer.draft_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == [
        {"field": "request_status", "value": 403},
        {"field": "resource_exists_after", "value": True},
    ]
    assert writer.read_draft() == {
        "request_status": 403,
        "resource_exists_after": True,
    }


def test_observation_writer_persists_the_complete_independent_control_tuple(
    tmp_path: Path,
) -> None:
    """Dropping any raw control fact must prevent a derived non-vacuity result."""
    required_methods = {
        "record_control_http_status",
        "record_control_patient_exists_before",
        "record_control_patient_exists_after",
    }
    assert required_methods <= set(dir(ObservationWriter))
    writer = ObservationWriter(tmp_path / "observation.json")

    writer.record_control_patient_exists_before(True)
    writer.record_control_http_status(204)
    writer.record_control_patient_exists_after(False)

    assert writer.read_events() == [
        ("control_resource_exists_before", True),
        ("control_request_status", 204),
        ("control_resource_exists_after", False),
    ]


@pytest.mark.parametrize(
    "serialized_event",
    [
        '{"field":"request_status","value":403,"unexpected":"data"}\n',
        '{"field":"unknown","value":403}\n',
        'not-json\n',
    ],
)
def test_observation_writer_rejects_extra_unknown_or_malformed_events(
    tmp_path: Path,
    serialized_event: str,
) -> None:
    """Unrecognized event structure must not enter final evidence."""
    writer = ObservationWriter(tmp_path / "observation.json")
    writer.draft_path.write_text(serialized_event, encoding="utf-8")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        writer.read_events()


def test_observation_writer_exclusively_writes_validated_observation(
    tmp_path: Path,
) -> None:
    """The final file must preserve all runtime facts and bind the approved contract."""
    contract_bytes = b'{"contract_id":"patient-delete-authz-001"}'
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    observation = RuntimeObservation(
        revision="base-revision",
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=True,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=403,
        resource_exists_after=True,
        pytest_exit_code=0,
        reason_code="execution_completed",
    )
    observation_path = tmp_path / "observation.json"
    writer = ObservationWriter(observation_path)

    writer.write(observation, contract_sha256=contract_sha256)

    serialized_observation = observation_path.read_text(encoding="utf-8")
    assert json.loads(serialized_observation) == {
        "action_attempted": True,
        "contract_sha256": contract_sha256,
        "control_request_status": 204,
        "control_resource_exists_after": False,
        "control_resource_exists_before": True,
        "control_succeeded": True,
        "pytest_exit_code": 0,
        "reason_code": "execution_completed",
        "request_status": 403,
        "resource_exists_after": True,
        "revision": "base-revision",
        "setup_succeeded": True,
    }
    envelope = RuntimeObservationEnvelope.model_validate_json(
        serialized_observation
    )
    assert envelope.contract_sha256 == contract_sha256
    assert envelope.model_dump(exclude={"contract_sha256"}) == (
        observation.model_dump()
    )
    with pytest.raises(FileExistsError):
        writer.write(observation, contract_sha256=contract_sha256)


def test_observation_writer_rejects_unvalidated_or_unbound_final_data(
    tmp_path: Path,
) -> None:
    """Arbitrary dictionaries and malformed provenance digests must not become evidence."""
    writer = ObservationWriter(tmp_path / "observation.json")
    observation = RuntimeObservation(
        revision="candidate-revision",
        setup_succeeded=True,
        action_attempted=True,
        control_succeeded=True,
        control_request_status=204,
        control_resource_exists_before=True,
        control_resource_exists_after=False,
        request_status=204,
        resource_exists_after=False,
        pytest_exit_code=1,
        reason_code="security_assertion_failed",
    )

    with pytest.raises(TypeError, match="RuntimeObservation"):
        writer.write(  # type: ignore[arg-type]
            observation.model_dump(),
            contract_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        writer.write(observation, contract_sha256="not-a-digest")


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("setup_succeeded", "true"),
        ("action_attempted", 1),
        ("control_succeeded", "true"),
        ("control_request_status", False),
        ("control_resource_exists_before", "true"),
        ("control_resource_exists_after", "false"),
        ("request_status", False),
        ("resource_exists_after", "false"),
        ("pytest_exit_code", "0"),
        ("revision", " base-revision"),
        ("reason_code", 123),
    ],
)
def test_persisted_observation_envelope_rejects_coerced_raw_facts(
    field: str,
    invalid_value: object,
) -> None:
    """Malformed JSON evidence must fail strict envelope validation on reload."""
    payload = {
        "revision": "base-revision",
        "setup_succeeded": True,
        "action_attempted": True,
        "control_request_status": 204,
        "control_resource_exists_before": True,
        "control_resource_exists_after": False,
        "control_succeeded": True,
        "request_status": 403,
        "resource_exists_after": True,
        "pytest_exit_code": 0,
        "reason_code": "execution_completed",
        "contract_sha256": "0" * 64,
    }
    payload[field] = invalid_value

    with pytest.raises(ValidationError):
        RuntimeObservationEnvelope.model_validate_json(json.dumps(payload))
