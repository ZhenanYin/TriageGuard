"""The complete, explicit operation vocabulary available to the test planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrimitiveDefinition:
    """One runtime operation the planner may select, without executable code."""

    name: str
    purpose: str
    input_types: Mapping[str, type[str]]
    output_names: tuple[str, ...]
    allowed_phases: frozenset[str]
    required_captures: Mapping[str, tuple[str, ...]]
    runtime_helper: str

    def as_prompt_data(self) -> dict[str, Any]:
        """Return JSON-safe primitive metadata for the structured model request."""
        return {
            "name": self.name,
            "purpose": self.purpose,
            "input_types": {
                name: value_type.__name__
                for name, value_type in self.input_types.items()
            },
            "output_names": list(self.output_names),
            "allowed_phases": sorted(self.allowed_phases),
            "required_captures": {
                phase: list(captures)
                for phase, captures in self.required_captures.items()
            },
            "runtime_helper": self.runtime_helper,
        }


PRIMITIVE_CATALOG: dict[str, PrimitiveDefinition] = {
    "create_patient": PrimitiveDefinition(
        name="create_patient",
        purpose="Create an isolated patient record for the controlled experiment.",
        input_types={},
        output_names=("$patient_id", "$control_patient_id"),
        allowed_phases=frozenset({"setup", "control"}),
        required_captures={
            "setup": ("$patient_id",),
            "control": ("$control_patient_id",),
        },
        runtime_helper="OpenMrsTestClient.create_patient",
    ),
    "login_as_actor": PrimitiveDefinition(
        name="login_as_actor",
        purpose="Authenticate the approved primary actor or authorized control actor.",
        input_types={"actor": str},
        output_names=("$actor_session", "$control_actor_session"),
        allowed_phases=frozenset({"setup", "control"}),
        required_captures={
            "setup": ("$actor_session",),
            "control": ("$control_actor_session",),
        },
        runtime_helper="OpenMrsTestClient.login_as_actor",
    ),
    "delete_patient": PrimitiveDefinition(
        name="delete_patient",
        purpose="Attempt the bounded patient-deletion action.",
        input_types={"patient_id": str, "actor_session": str},
        output_names=("$delete_status", "$control_delete_status"),
        allowed_phases=frozenset({"action", "control"}),
        required_captures={
            "action": ("$delete_status",),
            "control": ("$control_delete_status",),
        },
        runtime_helper="OpenMrsTestClient.delete_patient",
    ),
    "read_patient": PrimitiveDefinition(
        name="read_patient",
        purpose="Read the controlled patient record to support a state observation.",
        input_types={"patient_id": str, "actor_session": str},
        output_names=(
            "$patient_exists",
            "$control_patient_exists_before",
            "$control_patient_exists",
        ),
        allowed_phases=frozenset({"post_action", "control"}),
        required_captures={
            "post_action": ("$patient_exists",),
        },
        runtime_helper="OpenMrsTestClient.read_patient",
    ),
    "record_http_status": PrimitiveDefinition(
        name="record_http_status",
        purpose="Record the HTTP status produced by the deletion attempt.",
        input_types={},
        output_names=("$delete_status", "$control_delete_status"),
        allowed_phases=frozenset({"assertion"}),
        required_captures={},
        runtime_helper="ObservationWriter.record_http_status",
    ),
    "record_patient_exists": PrimitiveDefinition(
        name="record_patient_exists",
        purpose="Record whether the patient persists after the deletion attempt.",
        input_types={},
        output_names=("$patient_exists", "$control_patient_exists"),
        allowed_phases=frozenset({"assertion"}),
        required_captures={},
        runtime_helper="ObservationWriter.record_patient_exists",
    ),
    "record_control_http_status": PrimitiveDefinition(
        name="record_control_http_status",
        purpose="Record the administrator control deletion HTTP status.",
        input_types={},
        output_names=("$control_delete_status",),
        allowed_phases=frozenset({"assertion"}),
        required_captures={},
        runtime_helper="ObservationWriter.record_control_http_status",
    ),
    "record_control_patient_exists_before": PrimitiveDefinition(
        name="record_control_patient_exists_before",
        purpose="Record that the independent control patient exists before deletion.",
        input_types={},
        output_names=("$control_patient_exists_before",),
        allowed_phases=frozenset({"assertion"}),
        required_captures={},
        runtime_helper="ObservationWriter.record_control_patient_exists_before",
    ),
    "record_control_patient_exists_after": PrimitiveDefinition(
        name="record_control_patient_exists_after",
        purpose="Record that the independent control patient is absent after deletion.",
        input_types={},
        output_names=("$control_patient_exists",),
        allowed_phases=frozenset({"assertion"}),
        required_captures={},
        runtime_helper="ObservationWriter.record_control_patient_exists_after",
    ),
    "authorized_cleanup_patient": PrimitiveDefinition(
        name="authorized_cleanup_patient",
        purpose="Remove the isolated patient with the fixture's authorized cleanup path.",
        input_types={"patient_id": str},
        output_names=("$cleanup_complete", "$control_cleanup_complete"),
        allowed_phases=frozenset({"cleanup"}),
        required_captures={},
        runtime_helper="OpenMrsTestClient.authorized_cleanup_patient",
    ),
}


def primitive_catalog_prompt_data() -> list[dict[str, Any]]:
    """Return the entire fixed catalog in a stable prompt representation."""
    return [PRIMITIVE_CATALOG[name].as_prompt_data() for name in sorted(PRIMITIVE_CATALOG)]
