"""Tests for replayable, structured pytest-bdd generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from triageguard.contracts.gherkin import render_gherkin
from triageguard.domain.models import RiskContract, TestPlan
from triageguard.generation.code_generator import (
    ALLOWED_IMPORTS,
    GeneratedCodeArtifact,
    generate_pytest,
)
from triageguard.llm.gateway import ModelRequest, canonical_json
from triageguard.llm.replay_gateway import ReplayGateway


class _RecordingReplayGateway(ReplayGateway):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__({"pytest_generation": response})
        self.request: ModelRequest | None = None

    def generate(self, request: ModelRequest):
        self.request = request
        return super().generate(request)


def test_generation_is_hash_bound_and_preserves_model_code_byte_for_byte():
    """Dropping hashes, constraints, or code bytes would make replay unauditable."""
    contract = RiskContract.model_validate(_read_fixture("approved_contract.json"))
    plan = TestPlan.model_validate(_read_fixture("planner_response.json"))
    response = _read_fixture("generator_response.json")
    gherkin = render_gherkin(contract)
    gateway = _RecordingReplayGateway(response)

    artifact = generate_pytest(contract, gherkin, plan, gateway)

    assert isinstance(artifact, GeneratedCodeArtifact)
    assert gateway.request is not None
    assert gateway.request.purpose == "pytest_generation"
    assert gateway.request.payload["contract_sha256"] == hashlib.sha256(
        canonical_json(contract.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    assert gateway.request.payload["gherkin_sha256"] == hashlib.sha256(
        gherkin.encode("utf-8")
    ).hexdigest()
    assert gateway.request.payload["plan"] == plan.model_dump(mode="json")
    assert gateway.request.payload["allowed_imports"] == list(ALLOWED_IMPORTS)
    assert artifact.code == response["code"]
    assert artifact.contract_id == contract.contract_id
    assert artifact.model_response.response_sha256 == hashlib.sha256(
        canonical_json(response).encode("utf-8")
    ).hexdigest()


def _read_fixture(name: str) -> dict[str, object]:
    fixture_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "patient_delete_authorization"
        / name
    )
    return json.loads(fixture_path.read_text())
