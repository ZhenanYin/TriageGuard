"""Durable raw-fact capture and exclusive runtime-observation writing."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from triageguard.domain.models import RuntimeObservation

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DRAFT_SUFFIX = ".events.jsonl"


class RuntimeObservationEnvelope(RuntimeObservation):
    """A strict, provenance-bound schema for one final observation file."""

    contract_sha256: str = Field(pattern=_SHA256_PATTERN.pattern)


class ObservationWriter:
    """Persist raw generated-test facts separately from final observations."""

    def __init__(self, observation_path: str | Path) -> None:
        self._observation_path = Path(observation_path)
        self._draft_path = Path(f"{self._observation_path}{_DRAFT_SUFFIX}")

    @property
    def draft_path(self) -> Path:
        """Return the deterministic sidecar path used by the execution runner."""
        return self._draft_path

    def record_http_status(self, status: int) -> None:
        """Append the deletion response status without interpreting it."""
        if isinstance(status, bool) or not isinstance(status, int):
            raise TypeError("HTTP status must be an integer")
        self._append_event("request_status", status)

    def record_patient_exists(self, patient_exists: bool) -> None:
        """Append the observed resource state without interpreting it."""
        if not isinstance(patient_exists, bool):
            raise TypeError("patient_exists must be a boolean")
        self._append_event("resource_exists_after", patient_exists)

    def record_control_http_status(self, status: int) -> None:
        """Append the independent administrator deletion status."""
        if isinstance(status, bool) or not isinstance(status, int):
            raise TypeError("control HTTP status must be an integer")
        self._append_event("control_request_status", status)

    def record_control_patient_exists_before(self, patient_exists: bool) -> None:
        """Append whether the independent control patient existed before delete."""
        if not isinstance(patient_exists, bool):
            raise TypeError("control patient_exists before must be a boolean")
        self._append_event("control_resource_exists_before", patient_exists)

    def record_control_patient_exists_after(self, patient_exists: bool) -> None:
        """Append whether the independent control patient existed after delete."""
        if not isinstance(patient_exists, bool):
            raise TypeError("control patient_exists after must be a boolean")
        self._append_event("control_resource_exists_after", patient_exists)

    def read_draft(self) -> dict[str, int | bool]:
        """Read the latest raw value for each recorded field for runner assembly."""
        facts: dict[str, int | bool] = {}
        for field, value in self.read_events():
            facts[field] = value
        return facts

    def read_events(self) -> list[tuple[str, int | bool]]:
        """Read every validated raw event without collapsing duplicates."""
        events: list[tuple[str, int | bool]] = []
        with self._draft_path.open("r", encoding="utf-8") as draft:
            for line in draft:
                event = json.loads(line)
                if not isinstance(event, dict) or set(event) != {"field", "value"}:
                    raise ValueError(
                        "observation draft events require exactly field and value"
                    )
                field = event.get("field")
                value = event.get("value")
                self._validate_event(field, value)
                events.append((field, value))
        return events

    def write(
        self,
        observation: RuntimeObservation,
        *,
        contract_sha256: str,
    ) -> None:
        """Create the final digest-bound observation without overwriting evidence."""
        if not isinstance(observation, RuntimeObservation):
            raise TypeError("observation must be a validated RuntimeObservation")
        if _SHA256_PATTERN.fullmatch(contract_sha256) is None:
            raise ValueError("contract_sha256 must be a lowercase SHA-256 digest")

        envelope = RuntimeObservationEnvelope.model_validate(
            {
                **observation.model_dump(mode="json"),
                "contract_sha256": contract_sha256,
            }
        )
        serialized = (
            json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n"
        )
        self._observation_path.parent.mkdir(parents=True, exist_ok=True)
        with self._observation_path.open("x", encoding="utf-8") as destination:
            destination.write(serialized)
            destination.flush()
            os.fsync(destination.fileno())

    def _append_event(self, field: str, value: int | bool) -> None:
        event = json.dumps(
            {"field": field, "value": value},
            separators=(",", ":"),
            sort_keys=True,
        )
        self._draft_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self._draft_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, f"{event}\n".encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_event(field: Any, value: Any) -> None:
        if field == "request_status":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("draft request_status must be an integer")
            return
        if field == "resource_exists_after":
            if not isinstance(value, bool):
                raise ValueError("draft resource_exists_after must be a boolean")
            return
        if field == "control_request_status":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("draft control_request_status must be an integer")
            return
        if field in {
            "control_resource_exists_before",
            "control_resource_exists_after",
        }:
            if not isinstance(value, bool):
                raise ValueError(f"draft {field} must be a boolean")
            return
        raise ValueError(f"unknown observation draft field: {field!r}")
