"""Offline synthetic fixture adapter for the Milestone 2 workflow."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from triageguard.analysis import (
    ContextBuilder,
    DiffBuilder,
    FrozenContextRefiner,
    SnapshotAcquirer,
)
from triageguard.config import Settings
from triageguard.domain import EnvironmentKind
from triageguard.llm import ModelRequest, ModelResponse, ReplayGateway
from triageguard.research import ArtifactRecorder
from triageguard.sources.git import GitCommandRunner, GitObjectStore
from triageguard.sources.github import GitHubClient
from triageguard.workflow.milestone_two import MilestoneTwoWorkflow

_FIXTURE_PULL_NUMBER = 900000001
_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "milestone_two"
    / "openmrs_shaped_pr"
)

FixtureOutcome = Literal[
    "risks_proposed",
    "no_meaningful_security_risk_found",
    "insufficient_context_to_assess",
]

_SUPPORTED_OUTCOMES = frozenset(
    {
        "risks_proposed",
        "no_meaningful_security_risk_found",
        "insufficient_context_to_assess",
    }
)


class _FixtureResponse:
    """Minimal local response object accepted by the GitHub metadata client."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.status_code = 200
        self.headers = {
            "ETag": '"triageguard-synthetic-fixture"',
            "X-RateLimit-Remaining": "5000",
            "X-RateLimit-Reset": "0",
        }
        self._payload = dict(payload)

    def json(self) -> dict[str, object]:
        """Return an isolated copy of one recorded synthetic response."""
        return copy.deepcopy(self._payload)


class _FixtureSession:
    """Route only two fixed GitHub metadata reads to local fixture JSON."""

    def __init__(
        self,
        *,
        repository_metadata: Mapping[str, object],
        pull_metadata: Mapping[str, object],
    ) -> None:
        self._repository_metadata = dict(repository_metadata)
        self._pull_metadata = dict(pull_metadata)

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: object,
    ) -> _FixtureResponse:
        """Return recorded metadata without opening a network connection."""
        del headers, timeout

        if url == "https://api.github.com/repos/openmrs/openmrs-core":
            return _FixtureResponse(self._repository_metadata)
        if url == (
            "https://api.github.com/repos/openmrs/openmrs-core"
            f"/pulls/{_FIXTURE_PULL_NUMBER}"
        ):
            return _FixtureResponse(self._pull_metadata)

        raise AssertionError("The replay fixture attempted an unsupported URL.")


class _FixtureGitObjectStore(GitObjectStore):
    """Import only the checked-in synthetic Git bundle into a private store."""

    def __init__(
        self,
        *,
        root: Path,
        bundle_path: Path,
        base_sha: str,
        head_sha: str,
        candidate_sha: str,
    ) -> None:
        runner = GitCommandRunner()
        super().__init__(root=root, runner=runner)
        self._fixture_runner = runner
        self._bundle_path = bundle_path
        self._base_sha = base_sha
        self._head_sha = head_sha
        self._candidate_sha = candidate_sha
        self._loaded = False

    def fetch_snapshot(self, base_branch: str, pull_number: int) -> None:
        """Load exactly M/B/H/C from the local fixture, never from GitHub."""
        if base_branch != "main":
            raise ValueError("The synthetic fixture supports only the main branch.")
        if pull_number != _FIXTURE_PULL_NUMBER:
            raise ValueError("The synthetic fixture supports one reserved pull number.")
        if self._loaded:
            return

        self._fixture_runner.run(
            [
                "--git-dir",
                str(self.root),
                "bundle",
                "unbundle",
                str(self._bundle_path),
            ]
        )
        for reference, commit_sha in (
            ("refs/triageguard/base", self._base_sha),
            ("refs/triageguard/head", self._head_sha),
            ("refs/triageguard/candidate", self._candidate_sha),
        ):
            self._fixture_runner.run(
                [
                    "--git-dir",
                    str(self.root),
                    "update-ref",
                    reference,
                    commit_sha,
                ]
            )
        self._loaded = True

    def remote_snapshot_refs(
        self,
        base_branch: str,
        pull_number: int,
    ) -> tuple[str, str]:
        """Return fixture B/C identities without reading a live Git remote."""
        if base_branch != "main":
            raise ValueError("The synthetic fixture supports only the main branch.")
        if pull_number != _FIXTURE_PULL_NUMBER:
            raise ValueError("The synthetic fixture supports one reserved pull number.")
        return self._base_sha, self._candidate_sha


class _TemplateReplayGateway:
    """Bind replay templates to this exact workflow request before validation."""

    def __init__(
        self,
        *,
        risk_templates: Mapping[str, object],
        gherkin_template: Mapping[str, object],
        testability_template: Mapping[str, object],
    ) -> None:
        self._risk_templates = dict(risk_templates)
        self._gherkin_template = dict(gherkin_template)
        self._testability_template = dict(testability_template)

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return only a fixture response bound to the given request fields."""
        if request.purpose == "risk_hypothesis":
            template = self._risk_template(request)
        elif request.purpose == "gherkin_generation":
            template = self._gherkin_response(request)
        elif request.purpose == "testability_assessment":
            template = self._testability_response(request)
        else:
            raise ValueError("The synthetic replay supports only Milestone 2 requests.")

        return ReplayGateway(
            {request.purpose: template},
            model="replay/openai-gpt-oss-120b",
        ).generate(request)

    def _risk_template(self, request: ModelRequest) -> dict[str, object]:
        outcome = request.payload.get("fixture_outcome")
        if not isinstance(outcome, str) or outcome not in _SUPPORTED_OUTCOMES:
            raise ValueError(
                "The replay risk request lacked a supported fixture outcome."
            )

        raw_template = self._risk_templates.get(outcome)
        if not isinstance(raw_template, Mapping):
            raise TypeError("The risk fixture lacked the requested recorded outcome.")

        snapshot_key = _string_value(request.payload, "snapshot_key")
        context_sha256 = _string_value(request.payload, "context_sha256")
        evidence_envelope = _mapping_value(request.payload, "evidence_envelope")
        evidence_envelope_sha256 = _string_value(
            evidence_envelope,
            "envelope_sha256",
        )
        integration_anchor_id = _integration_anchor_id(request.payload)

        result = _substitute(
            raw_template,
            {
                "__SNAPSHOT_KEY__": snapshot_key,
                "__CONTEXT_SHA256__": context_sha256,
                "__EVIDENCE_ENVELOPE_SHA256__": evidence_envelope_sha256,
                "__INTEGRATION_ANCHOR_ID__": integration_anchor_id,
            },
        )
        if not isinstance(result, dict):
            raise TypeError("The substituted risk fixture must be an object.")
        return result

    def _gherkin_response(self, request: ModelRequest) -> dict[str, object]:
        snapshot_key = _string_value(request.payload, "snapshot_key")
        reviewed_risk_sha256 = _string_value(request.payload, "reviewed_risk_sha256")
        evidence_envelope = _mapping_value(request.payload, "evidence_envelope")
        context_sha256 = _string_value(evidence_envelope, "context_sha256")
        evidence_envelope_sha256 = _string_value(
            evidence_envelope,
            "envelope_sha256",
        )
        approved_risk = _mapping_value(request.payload, "approved_risk")

        result = _substitute(
            self._gherkin_template,
            {
                "__SNAPSHOT_KEY__": snapshot_key,
                "__CONTEXT_SHA256__": context_sha256,
                "__EVIDENCE_ENVELOPE_SHA256__": evidence_envelope_sha256,
                "__REVIEWED_RISK_SHA256__": reviewed_risk_sha256,
                "__APPROVED_RISK__": dict(approved_risk),
                "__INTEGRATION_ANCHOR_ID__": _integration_anchor_id(
                    request.payload,
                ),
            },
        )
        if not isinstance(result, dict):
            raise TypeError("The substituted Gherkin fixture must be an object.")
        return result

    def _testability_response(self, request: ModelRequest) -> dict[str, object]:
        """Bind the recorded testability decision to this exact review/context."""
        evidence_envelope = _mapping_value(request.payload, "evidence_envelope")
        result = _substitute(
            self._testability_template,
            {
                "__SNAPSHOT_KEY__": _string_value(request.payload, "snapshot_key"),
                "__CONTEXT_SHA256__": _string_value(
                    evidence_envelope,
                    "context_sha256",
                ),
                "__EVIDENCE_ENVELOPE_SHA256__": _string_value(
                    evidence_envelope,
                    "envelope_sha256",
                ),
                "__REVIEWED_RISK_SHA256__": _string_value(
                    request.payload,
                    "reviewed_risk_sha256",
                ),
                "__INTEGRATION_ANCHOR_ID__": _integration_anchor_id(
                    request.payload,
                ),
            },
        )
        if not isinstance(result, dict):
            raise TypeError("The substituted testability fixture must be an object.")
        return result


def build_milestone_two_replay_workflow(
    settings: Settings,
    *,
    outcome: FixtureOutcome = "risks_proposed",
) -> MilestoneTwoWorkflow:
    """Build one entirely offline workflow for the synthetic OpenMRS-shaped PR."""
    if settings.llm_mode != "replay":
        raise ValueError("The synthetic workflow requires replay model mode.")
    if settings.environment_kind is not EnvironmentKind.CONTROLLED_FIXTURE:
        raise ValueError("The synthetic workflow requires a controlled fixture.")
    if outcome not in _SUPPORTED_OUTCOMES:
        raise ValueError("The requested synthetic outcome is not supported.")

    repository_metadata = _load_fixture_object(
        _FIXTURE_ROOT / "metadata" / "repository.json"
    )
    pull_metadata = _load_fixture_object(_FIXTURE_ROOT / "metadata" / "pull.json")
    risk_templates = _load_fixture_object(
        _FIXTURE_ROOT / "model" / "risk_hypothesis.json"
    )
    gherkin_template = _load_fixture_object(
        _FIXTURE_ROOT / "model" / "gherkin_generation.json"
    )
    testability_template = _load_fixture_object(
        _FIXTURE_ROOT / "model" / "testability_assessment.json"
    )

    fixture_commits = _mapping_value(repository_metadata, "fixture_commits")
    base_sha = _fixture_commit_sha(fixture_commits, "base")
    head_sha = _fixture_commit_sha(fixture_commits, "head")
    candidate_sha = _fixture_commit_sha(fixture_commits, "candidate")

    run_id = f"m2-replay-{uuid4().hex}"
    store = _FixtureGitObjectStore(
        root=settings.analysis_cache_dir / "milestone-two-replay" / f"{run_id}.git",
        bundle_path=_FIXTURE_ROOT / "git" / "repository.bundle",
        base_sha=base_sha,
        head_sha=head_sha,
        candidate_sha=candidate_sha,
    )
    store.initialize()

    gateway = _TemplateReplayGateway(
        risk_templates=risk_templates,
        gherkin_template=gherkin_template,
        testability_template=testability_template,
    )
    workflow = MilestoneTwoWorkflow(
        run_id=run_id,
        settings=settings,
        recorder=ArtifactRecorder(settings.artifacts_dir),
        snapshot_acquirer=SnapshotAcquirer(
            github=GitHubClient(
                session=_FixtureSession(
                    repository_metadata=repository_metadata,
                    pull_metadata=pull_metadata,
                ),
                api_version=settings.github_api_version,
            ),
            store=store,
            settings=settings,
            sleep=lambda _seconds: None,
        ),
        diff_builder=DiffBuilder(
            store,
            max_files=settings.max_diff_files,
            max_bytes=settings.max_diff_bytes,
        ),
        context_builder=ContextBuilder(),
        store=store,
        gateway=_OutcomeBoundGateway(gateway=gateway, outcome=outcome),
        evidence_refiner=FrozenContextRefiner(),
    )
    return workflow


class _OutcomeBoundGateway:
    """Add one explicit test-fixture outcome selector to the risk request only."""

    def __init__(
        self,
        *,
        gateway: _TemplateReplayGateway,
        outcome: FixtureOutcome,
    ) -> None:
        self._gateway = gateway
        self._outcome = outcome

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Pass the normal request through with the fixture outcome annotation."""
        if request.purpose != "risk_hypothesis":
            return self._gateway.generate(request)

        replay_request = request.model_copy(
            update={
                "payload": {
                    **request.payload,
                    "fixture_outcome": self._outcome,
                }
            }
        )
        return self._gateway.generate(replay_request)


def _load_fixture_object(path: Path) -> dict[str, object]:
    """Load one checked-in fixture JSON object with no fallback behavior."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "The synthetic replay fixture could not be loaded."
        ) from error

    if not isinstance(value, dict):
        raise TypeError("Each synthetic replay fixture must be a JSON object.")
    return value


def _fixture_commit_sha(
    fixture_commits: Mapping[str, object],
    role: str,
) -> str:
    """Read one recorded fixture commit SHA by its documented role."""
    commit = fixture_commits.get(role)
    if not isinstance(commit, Mapping):
        raise TypeError("The synthetic fixture lacks a documented commit role.")
    return _string_value(commit, "commit_sha")


def _mapping_value(
    source: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    """Return a named mapping value or fail closed."""
    value = source.get(name)
    if not isinstance(value, Mapping):
        raise TypeError("The synthetic replay request had an invalid object field.")
    return value


def _string_value(
    source: Mapping[str, object],
    name: str,
) -> str:
    """Return a named non-empty string value or fail closed."""
    value = source.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError("The synthetic replay request had an invalid text field.")
    return value


def _integration_anchor_id(payload: Mapping[str, Any]) -> str:
    """Require exactly one citeable primary integration anchor in the request."""
    raw_envelope = payload.get("evidence_envelope")
    if isinstance(raw_envelope, Mapping):
        anchors = raw_envelope.get("visible_anchors")
    else:
        anchors = payload.get("context_anchors")
    if not isinstance(anchors, list):
        raise TypeError("The synthetic replay request lacked context anchors.")

    anchor_ids = [
        _string_value(anchor, "anchor_id")
        for anchor in anchors
        if isinstance(anchor, Mapping)
        and anchor.get("change_relation") == "integration_change"
    ]
    if len(anchor_ids) != 1:
        raise ValueError(
            "The synthetic fixture requires exactly one integration evidence anchor."
        )
    return anchor_ids[0]


def _substitute(
    value: object,
    replacements: Mapping[str, object],
) -> object:
    """Recursively replace only explicit fixture placeholders."""
    if isinstance(value, str):
        if value in replacements:
            return copy.deepcopy(replacements[value])
        if value.startswith("__") and value.endswith("__"):
            raise ValueError(
                "The synthetic fixture contained an unresolved placeholder."
            )
        return value
    if isinstance(value, list):
        return [_substitute(item, replacements) for item in value]
    if isinstance(value, Mapping):
        return {key: _substitute(item, replacements) for key, item in value.items()}
    return value
