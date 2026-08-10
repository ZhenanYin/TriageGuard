"""Bounded OpenMRS HTTP operations available to generated tests."""

from __future__ import annotations

from typing import Any, NewType
from urllib.parse import quote

import requests

_REQUEST_TIMEOUT_SECONDS = 5.0
_PATIENT_PATH = "/ws/rest/v1/patient"
_SESSION_PATH = "/ws/rest/v1/session"


class TargetUnavailable(RuntimeError):
    """The controlled target could not be reached at its HTTP boundary."""


# Milestone 1 selects an actor-scoped Basic-auth identity. A later real-target
# executor may replace this controlled-fixture credential mapping.
ActorSession = NewType("ActorSession", str)


class OpenMrsTestClient:
    """A small, allowlisted client for the patient authorization experiment."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        _session: requests.Session | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not username:
            raise ValueError("username must not be empty")
        if not password:
            raise ValueError("password must not be empty")
        self._base_url = base_url.rstrip("/")
        self._administrator_username = username
        self._password = password
        self._session = _session if _session is not None else requests.Session()

    def create_patient(self) -> str:
        """Create one isolated patient with the authorized harness identity."""
        response = self._request(
            "POST",
            _PATIENT_PATH,
            auth=(self._administrator_username, self._password),
            json={
                "person": {
                    "names": [
                        {"givenName": "TriageGuard", "familyName": "Fixture"}
                    ],
                    "gender": "U",
                }
            },
        )
        self._require_success(response)
        patient_id = response.json().get("uuid")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("OpenMRS patient response did not contain a UUID")
        return patient_id

    def login_as_actor(self, actor: str) -> ActorSession:
        """Select and verify the actor identity used by later patient calls."""
        if not actor:
            raise ValueError("actor must not be empty")
        username = (
            self._administrator_username if actor == "administrator" else actor
        )
        response = self._request(
            "GET",
            _SESSION_PATH,
            auth=(username, self._password),
        )
        self._require_success(response)
        if response.json().get("authenticated") is not True:
            raise PermissionError(f"OpenMRS did not authenticate actor {actor!r}")
        return ActorSession(username)

    def delete_patient(
        self, patient_id: str, actor_session: ActorSession
    ) -> int:
        """Attempt deletion and return its raw HTTP status for observation."""
        response = self._request(
            "DELETE",
            self._patient_path(patient_id),
            auth=self._auth_for(actor_session),
        )
        return response.status_code

    def read_patient(
        self, patient_id: str, actor_session: ActorSession
    ) -> bool:
        """Return whether the target reports the patient as present."""
        response = self._request(
            "GET",
            self._patient_path(patient_id),
            auth=self._auth_for(actor_session),
        )
        if response.status_code == 404:
            return False
        self._require_success(response)
        return True

    def authorized_cleanup_patient(self, patient_id: str) -> bool:
        """Purge the isolated patient with the harness administrator identity."""
        response = self._request(
            "DELETE",
            self._patient_path(patient_id),
            auth=(self._administrator_username, self._password),
            params={"purge": "true"},
        )
        if response.status_code == 404:
            return True
        self._require_success(response)
        return True

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            return self._session.request(
                method,
                f"{self._base_url}{path}",
                timeout=_REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
                **kwargs,
            )
        except requests.RequestException as error:
            raise TargetUnavailable("OpenMRS target request failed") from error

    @staticmethod
    def _require_success(response: requests.Response) -> None:
        if not 200 <= response.status_code < 300:
            raise requests.HTTPError(
                f"OpenMRS returned HTTP {response.status_code}",
                response=response,
            )

    def _auth_for(self, actor_session: ActorSession) -> tuple[str, str]:
        if not isinstance(actor_session, str) or not actor_session:
            raise TypeError("actor_session must come from login_as_actor")
        return actor_session, self._password

    @staticmethod
    def _patient_path(patient_id: str) -> str:
        if not patient_id:
            raise ValueError("patient_id must not be empty")
        return f"{_PATIENT_PATH}/{quote(patient_id, safe='')}"
