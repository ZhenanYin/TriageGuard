from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest
import requests

from triageguard.execution import ControlledAuthorizationServer
from triageguard.runtime import OpenMrsTestClient

ADMINISTRATOR = "fixture-administrator"
CLERK = "clerk"
PASSWORD = "fixture-password-not-for-logs"


def _client(server: ControlledAuthorizationServer) -> OpenMrsTestClient:
    return OpenMrsTestClient(
        base_url=server.base_url,
        username=ADMINISTRATOR,
        password=PASSWORD,
    )


def test_secure_fixture_denies_clerk_delete_and_keeps_patient() -> None:
    """Changing secure clerk DELETE to success must fail this boundary test."""
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        client = _client(server)
        patient_id = client.create_patient()
        clerk_session = client.login_as_actor(CLERK)

        assert client.delete_patient(patient_id, clerk_session) == 403
        assert client.read_patient(patient_id, clerk_session) is True
        assert client.authorized_cleanup_patient(patient_id) is True


def test_vulnerable_fixture_allows_clerk_delete_and_removes_patient() -> None:
    """Fail if vulnerable behavior stops exposing the approved regression."""
    with ControlledAuthorizationServer(
        behavior="vulnerable",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        client = _client(server)
        patient_id = client.create_patient()
        clerk_session = client.login_as_actor(CLERK)

        assert client.delete_patient(patient_id, clerk_session) == 204
        assert client.read_patient(patient_id, clerk_session) is False
        assert client.authorized_cleanup_patient(patient_id) is True


@pytest.mark.parametrize("behavior", ["secure", "vulnerable"])
def test_ordinary_delete_of_an_absent_patient_is_not_reported_as_success(
    behavior: str,
) -> None:
    """Only explicit purge cleanup may be idempotent for an absent resource."""
    with ControlledAuthorizationServer(
        behavior=behavior,
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        client = _client(server)
        patient_id = client.create_patient()
        administrator = client.login_as_actor("administrator")

        assert client.delete_patient(patient_id, administrator) == 204
        assert client.delete_patient(patient_id, administrator) == 404
        assert client.authorized_cleanup_patient(patient_id) is True


def test_fixture_binds_loopback_and_rejects_any_inexact_credential() -> None:
    """A wildcard bind or permissive credential check would escape the fixture."""
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as server:
        parsed = urlparse(server.base_url)
        assert parsed.hostname == "127.0.0.1"
        assert parsed.port is not None and parsed.port > 0

        wrong_password = requests.get(
            f"{server.base_url}/ws/rest/v1/session",
            auth=(CLERK, f"{PASSWORD}-wrong"),
            timeout=2,
        )
        unknown_actor = requests.get(
            f"{server.base_url}/ws/rest/v1/session",
            auth=("not-the-clerk", PASSWORD),
            timeout=2,
        )

        assert wrong_password.status_code == 401
        assert unknown_actor.status_code == 401
        assert PASSWORD not in wrong_password.text
        assert PASSWORD not in unknown_actor.text


def test_patient_state_is_not_shared_between_server_instances() -> None:
    """Moving patient state to a class/global dictionary must fail this test."""
    with ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    ) as first_server:
        first_client = _client(first_server)
        patient_id = first_client.create_patient()

        with ControlledAuthorizationServer(
            behavior="secure",
            administrator_username=ADMINISTRATOR,
            clerk_username=CLERK,
            password=PASSWORD,
        ) as second_server:
            second_client = _client(second_server)
            second_clerk = second_client.login_as_actor(CLERK)

            assert second_client.read_patient(patient_id, second_clerk) is False

        assert first_client.authorized_cleanup_patient(patient_id) is True


def test_server_lifecycle_closes_bound_socket_and_forbids_restart() -> None:
    """Stopping before start must release the port and permanently close state."""
    unstarted = ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    )
    bound_address = unstarted._server.server_address

    unstarted.stop()
    unstarted.stop()
    with socket.socket() as probe:
        probe.bind(bound_address)
    with pytest.raises(RuntimeError, match="closed"):
        unstarted.start()

    running = ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    )
    running.start()
    with pytest.raises(RuntimeError, match="already running"):
        running.start()
    running.stop()
    running.stop()
    with pytest.raises(RuntimeError, match="not running"):
        _ = running.base_url
    with pytest.raises(RuntimeError, match="closed"):
        running.start()


def test_cleanup_failure_does_not_replace_the_primary_body_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown failure must remain secondary to the execution failure."""
    server = ControlledAuthorizationServer(
        behavior="secure",
        administrator_username=ADMINISTRATOR,
        clerk_username=CLERK,
        password=PASSWORD,
    )
    real_stop = server.stop

    def fail_cleanup() -> None:
        raise OSError("controlled cleanup failed")

    monkeypatch.setattr(server, "stop", fail_cleanup)
    try:
        with pytest.raises(ValueError, match="primary execution failure"), server:
            raise ValueError("primary execution failure")
        assert isinstance(server.cleanup_error, OSError)
        assert "controlled cleanup failed" in str(server.cleanup_error)
    finally:
        real_stop()
