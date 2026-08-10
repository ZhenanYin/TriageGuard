"""Loopback-only OpenMRS-shaped authorization fixture."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Literal, Self
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

Behavior = Literal["secure", "vulnerable"]
_PATIENT_PATH = "/ws/rest/v1/patient"
_SESSION_PATH = "/ws/rest/v1/session"


class _FixtureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        *,
        behavior: Behavior,
        administrator_username: str,
        clerk_username: str,
        password: str,
    ) -> None:
        self.behavior = behavior
        self._administrator_credential = (administrator_username, password)
        self._clerk_credential = (clerk_username, password)
        self._patients: dict[str, dict[str, str]] = {}
        self._state_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _FixtureRequestHandler)

    def role_for_authorization(self, value: str | None) -> str | None:
        credential = _decode_basic_authorization(value)
        if credential is None:
            return None
        if _credential_matches(credential, self._administrator_credential):
            return "administrator"
        if _credential_matches(credential, self._clerk_credential):
            return "clerk"
        return None

    def create_patient(self) -> str:
        patient_id = str(uuid4())
        with self._state_lock:
            self._patients[patient_id] = {"uuid": patient_id}
        return patient_id

    def patient_exists(self, patient_id: str) -> bool:
        with self._state_lock:
            return patient_id in self._patients

    def remove_patient(self, patient_id: str) -> bool:
        with self._state_lock:
            return self._patients.pop(patient_id, None) is not None


class _FixtureRequestHandler(BaseHTTPRequestHandler):
    server: _FixtureHTTPServer

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        role = self._role_or_reject()
        if role is None:
            return
        if parsed.path == _SESSION_PATH and not parsed.query:
            self._json_response(HTTPStatus.OK, {"authenticated": True})
            return
        patient_id = _patient_id_from_path(parsed.path)
        if patient_id is not None and not parsed.query:
            if self.server.patient_exists(patient_id):
                self._json_response(HTTPStatus.OK, {"uuid": patient_id})
            else:
                self._empty_response(HTTPStatus.NOT_FOUND)
            return
        self._empty_response(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        role = self._role_or_reject()
        if role is None:
            return
        if parsed.path != _PATIENT_PATH or parsed.query or role != "administrator":
            self._empty_response(HTTPStatus.FORBIDDEN)
            return
        self._discard_request_body()
        patient_id = self.server.create_patient()
        self._json_response(HTTPStatus.CREATED, {"uuid": patient_id})

    def do_DELETE(self) -> None:
        parsed = urlsplit(self.path)
        role = self._role_or_reject()
        if role is None:
            return
        patient_id = _patient_id_from_path(parsed.path)
        if patient_id is None:
            self._empty_response(HTTPStatus.NOT_FOUND)
            return

        query = parse_qs(parsed.query, keep_blank_values=True)
        if query:
            if query != {"purge": ["true"]} or role != "administrator":
                self._empty_response(HTTPStatus.FORBIDDEN)
                return
            removed = self.server.remove_patient(patient_id)
            self._empty_response(
                HTTPStatus.NO_CONTENT if removed else HTTPStatus.NOT_FOUND
            )
            return

        if role == "administrator":
            removed = self.server.remove_patient(patient_id)
            self._empty_response(
                HTTPStatus.NO_CONTENT if removed else HTTPStatus.NOT_FOUND
            )
            return
        if self.server.behavior == "secure":
            self._empty_response(HTTPStatus.FORBIDDEN)
            return
        removed = self.server.remove_patient(patient_id)
        self._empty_response(
            HTTPStatus.NO_CONTENT if removed else HTTPStatus.NOT_FOUND
        )

    def log_message(self, format: str, *args: object) -> None:
        """Keep request and credential material out of process logs."""

    def _role_or_reject(self) -> str | None:
        role = self.server.role_for_authorization(self.headers.get("Authorization"))
        if role is None:
            self._empty_response(HTTPStatus.UNAUTHORIZED)
        return role

    def _discard_request_body(self) -> None:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            length = 0
        if length > 0:
            self.rfile.read(length)

    def _json_response(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty_response(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()


class ControlledAuthorizationServer:
    """Manage one isolated loopback authorization target."""

    def __init__(
        self,
        *,
        behavior: Behavior,
        administrator_username: str,
        clerk_username: str,
        password: str,
    ) -> None:
        if behavior not in {"secure", "vulnerable"}:
            raise ValueError("behavior must be 'secure' or 'vulnerable'")
        for label, value in (
            ("administrator_username", administrator_username),
            ("clerk_username", clerk_username),
            ("password", password),
        ):
            if not value:
                raise ValueError(f"{label} must not be empty")
        if ":" in administrator_username or ":" in clerk_username:
            raise ValueError("fixture usernames must not contain ':'")
        if administrator_username == clerk_username:
            raise ValueError("fixture actor usernames must be distinct")
        self._server = _FixtureHTTPServer(
            behavior=behavior,
            administrator_username=administrator_username,
            clerk_username=clerk_username,
            password=password,
        )
        self._thread: threading.Thread | None = None
        self._state: Literal["bound", "running", "closed"] = "bound"
        self._cleanup_error: BaseException | None = None

    @property
    def base_url(self) -> str:
        if (
            self._state != "running"
            or self._thread is None
            or not self._thread.is_alive()
        ):
            raise RuntimeError("controlled authorization server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def cleanup_error(self) -> BaseException | None:
        """Expose a secondary shutdown failure without replacing a body error."""
        return self._cleanup_error

    def start(self) -> ControlledAuthorizationServer:
        if self._state == "running":
            raise RuntimeError("controlled authorization server is already running")
        if self._state == "closed":
            raise RuntimeError("controlled authorization server is closed")
        thread = threading.Thread(
            target=self._server.serve_forever,
            name="triageguard-controlled-authorization",
            daemon=True,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._server.server_close()
            self._thread = None
            self._state = "closed"
            raise
        self._state = "running"
        return self

    def stop(self) -> None:
        if self._state == "closed":
            return
        thread = self._thread
        errors: list[Exception] = []
        try:
            if self._state == "running" and thread is not None:
                if thread.is_alive():
                    try:
                        self._server.shutdown()
                    except Exception as error:  # noqa: BLE001 - aggregate cleanup
                        errors.append(error)
                try:
                    thread.join(timeout=5)
                except Exception as error:  # noqa: BLE001 - aggregate cleanup
                    errors.append(error)
        finally:
            try:
                self._server.server_close()
            except Exception as error:  # noqa: BLE001 - aggregate cleanup
                errors.append(error)
            self._thread = None
            self._state = "closed"
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("controlled server cleanup failed", errors)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.stop()
        except Exception as cleanup_error:
            self._cleanup_error = cleanup_error
            if exc_value is None:
                raise
            exc_value.add_note(f"Secondary controlled-server cleanup failure: {cleanup_error}")


def _decode_basic_authorization(value: str | None) -> tuple[str, str] | None:
    if value is None or not value.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(value[6:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    if ":" not in decoded:
        return None
    username, password = decoded.split(":", 1)
    return username, password


def _credential_matches(
    supplied: tuple[str, str], expected: tuple[str, str]
) -> bool:
    return hmac.compare_digest(supplied[0], expected[0]) and hmac.compare_digest(
        supplied[1], expected[1]
    )


def _patient_id_from_path(path: str) -> str | None:
    prefix = f"{_PATIENT_PATH}/"
    if not path.startswith(prefix):
        return None
    encoded_patient_id = path[len(prefix) :]
    if not encoded_patient_id or "/" in encoded_patient_id:
        return None
    patient_id = unquote(encoded_patient_id)
    return patient_id or None
