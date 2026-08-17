"""Typed runtime configuration for the isolated V2 prototype."""

from __future__ import annotations

import os
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Literal
from weakref import finalize

from triageguard.domain.statuses import EnvironmentKind

MIN_REPEAT_COUNT = 1
MAX_REPEAT_COUNT = 20
DEFAULT_MAX_DIFF_FILES = 1_000
DEFAULT_MAX_DIFF_BYTES = 25_000_000
DEFAULT_MAX_MODEL_REQUEST_BYTES = 7_000
DEFAULT_MAX_MODEL_EVIDENCE_ROUNDS = 2
_PROCESS_SECRETS: dict[int, tuple[str | None, str | None]] = {}


@dataclass(frozen=True)
class Settings:
    """Configuration sourced explicitly from the process environment."""

    llm_mode: Literal["live", "replay"] = "replay"
    llm_provider: Literal["groq"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: InitVar[str | None] = None
    github_token: InitVar[str | None] = None
    artifacts_dir: Path = Path("artifacts")
    github_api_version: str = "2026-03-10"
    analysis_cache_dir: Path = Path("analysis-cache")
    max_diff_files: int = DEFAULT_MAX_DIFF_FILES
    max_diff_bytes: int = DEFAULT_MAX_DIFF_BYTES
    max_context_files: int = 40
    max_context_anchors: int = 80
    max_context_bytes: int = 160_000
    max_context_anchor_lines: int = 120
    max_context_blob_bytes: int = 1_000_000
    max_context_search_identifiers: int = 100
    max_context_hits_per_identifier: int = 20
    max_model_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_model_evidence_rounds: int = DEFAULT_MAX_MODEL_EVIDENCE_ROUNDS
    repeat_count: int = 3
    environment_kind: EnvironmentKind = EnvironmentKind.CONTROLLED_FIXTURE

    def __post_init__(self, groq_api_key: str | None, github_token: str | None) -> None:
        if self.llm_mode not in {"live", "replay"}:
            raise ValueError("TRIAGEGUARD_LLM_MODE must be 'live' or 'replay'")
        if self.llm_mode == "replay":
            groq_api_key = None
            github_token = None
        if (
            type(self.repeat_count) is not int
            or not MIN_REPEAT_COUNT <= self.repeat_count <= MAX_REPEAT_COUNT
        ):
            raise ValueError(
                f"repeat_count must be an integer from {MIN_REPEAT_COUNT} "
                f"to {MAX_REPEAT_COUNT}"
            )
        if self.llm_provider != "groq":
            raise ValueError("TRIAGEGUARD_LLM_PROVIDER must be 'groq'")
        if self.llm_mode == "live" and not groq_api_key:
            raise ValueError("GROQ_API_KEY is required when TRIAGEGUARD_LLM_MODE=live")
        if not isinstance(self.github_api_version, str) or not self.github_api_version:
            raise ValueError("github_api_version must be a non-empty string")
        for name in (
            "max_diff_files",
            "max_diff_bytes",
            "max_context_files",
            "max_context_anchors",
            "max_context_bytes",
            "max_context_anchor_lines",
            "max_context_blob_bytes",
            "max_context_search_identifiers",
            "max_context_hits_per_identifier",
            "max_model_request_bytes",
            "max_model_evidence_rounds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        identity = id(self)
        _PROCESS_SECRETS[identity] = (groq_api_key, github_token)
        finalize(self, _PROCESS_SECRETS.pop, identity, None)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings without permitting credentials in replay mode."""
        llm_mode = os.getenv("TRIAGEGUARD_LLM_MODE", "replay")
        if llm_mode not in {"live", "replay"}:
            raise ValueError("TRIAGEGUARD_LLM_MODE must be 'live' or 'replay'")
        llm_provider = os.getenv("TRIAGEGUARD_LLM_PROVIDER", "groq")
        if llm_provider != "groq":
            raise ValueError("TRIAGEGUARD_LLM_PROVIDER must be 'groq'")
        groq_api_key = os.getenv("GROQ_API_KEY") if llm_mode == "live" else None
        github_token = os.getenv("GITHUB_TOKEN") if llm_mode == "live" else None
        if github_token is not None and not github_token.strip():
            github_token = None

        try:
            environment_kind = EnvironmentKind(
                os.getenv(
                    "TRIAGEGUARD_ENVIRONMENT_KIND",
                    EnvironmentKind.CONTROLLED_FIXTURE.value,
                )
            )
        except ValueError as error:
            raise ValueError("TRIAGEGUARD_ENVIRONMENT_KIND is invalid") from error

        repeat_count_text = os.getenv("TRIAGEGUARD_REPEAT_COUNT", "3")
        try:
            repeat_count = int(repeat_count_text)
        except ValueError as error:
            raise ValueError("TRIAGEGUARD_REPEAT_COUNT must be an integer") from error

        context_environment_names = {
            "max_diff_files": "TRIAGEGUARD_MAX_DIFF_FILES",
            "max_diff_bytes": "TRIAGEGUARD_MAX_DIFF_BYTES",
            "max_context_files": "TRIAGEGUARD_MAX_CONTEXT_FILES",
            "max_context_anchors": "TRIAGEGUARD_MAX_CONTEXT_ANCHORS",
            "max_context_bytes": "TRIAGEGUARD_MAX_CONTEXT_BYTES",
            "max_context_anchor_lines": "TRIAGEGUARD_MAX_CONTEXT_ANCHOR_LINES",
            "max_context_blob_bytes": "TRIAGEGUARD_MAX_CONTEXT_BLOB_BYTES",
            "max_context_search_identifiers": "TRIAGEGUARD_MAX_CONTEXT_SEARCH_IDENTIFIERS",
            "max_context_hits_per_identifier": "TRIAGEGUARD_MAX_CONTEXT_HITS_PER_IDENTIFIER",
            "max_model_request_bytes": "TRIAGEGUARD_MAX_MODEL_REQUEST_BYTES",
            "max_model_evidence_rounds": "TRIAGEGUARD_MAX_MODEL_EVIDENCE_ROUNDS",
        }
        context_defaults = {
            "max_diff_files": DEFAULT_MAX_DIFF_FILES,
            "max_diff_bytes": DEFAULT_MAX_DIFF_BYTES,
            "max_context_files": 40,
            "max_context_anchors": 80,
            "max_context_bytes": 160_000,
            "max_context_anchor_lines": 120,
            "max_context_blob_bytes": 1_000_000,
            "max_context_search_identifiers": 100,
            "max_context_hits_per_identifier": 20,
            "max_model_request_bytes": DEFAULT_MAX_MODEL_REQUEST_BYTES,
            "max_model_evidence_rounds": DEFAULT_MAX_MODEL_EVIDENCE_ROUNDS,
        }
        context_limits: dict[str, int] = {}
        for field_name, environment_name in context_environment_names.items():
            try:
                context_limits[field_name] = int(
                    os.getenv(environment_name, str(context_defaults[field_name]))
                )
            except ValueError as error:
                raise ValueError(f"{environment_name} must be an integer") from error

        return cls(
            llm_mode=llm_mode,
            llm_provider=llm_provider,
            llm_model=os.getenv("TRIAGEGUARD_LLM_MODEL", "openai/gpt-oss-120b"),
            groq_api_key=groq_api_key,
            github_token=github_token,
            artifacts_dir=Path(os.getenv("TRIAGEGUARD_ARTIFACTS_DIR", "artifacts")),
            github_api_version=os.getenv(
                "TRIAGEGUARD_GITHUB_API_VERSION", "2026-03-10"
            ),
            analysis_cache_dir=Path(
                os.getenv("TRIAGEGUARD_ANALYSIS_CACHE_DIR", "analysis-cache")
            ),
            **context_limits,
            repeat_count=repeat_count,
            environment_kind=environment_kind,
        )

    def public_view(self) -> PublicSettings:
        """Return the non-secret configuration safe for app/session state."""
        return PublicSettings(
            llm_mode=self.llm_mode,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            artifacts_dir=self.artifacts_dir,
            github_api_version=self.github_api_version,
            analysis_cache_dir=self.analysis_cache_dir,
            max_diff_files=self.max_diff_files,
            max_diff_bytes=self.max_diff_bytes,
            max_context_files=self.max_context_files,
            max_context_anchors=self.max_context_anchors,
            max_context_bytes=self.max_context_bytes,
            max_context_anchor_lines=self.max_context_anchor_lines,
            max_context_blob_bytes=self.max_context_blob_bytes,
            max_context_search_identifiers=self.max_context_search_identifiers,
            max_context_hits_per_identifier=self.max_context_hits_per_identifier,
            max_model_request_bytes=self.max_model_request_bytes,
            max_model_evidence_rounds=self.max_model_evidence_rounds,
            repeat_count=self.repeat_count,
            environment_kind=self.environment_kind,
        )


@dataclass(frozen=True)
class PublicSettings:
    """Secret-free settings retained by public UI and session state."""

    llm_mode: Literal["live", "replay"]
    llm_provider: Literal["groq"]
    llm_model: str
    artifacts_dir: Path
    github_api_version: str
    analysis_cache_dir: Path
    max_diff_files: int
    max_diff_bytes: int
    max_context_files: int
    max_context_anchors: int
    max_context_bytes: int
    max_context_anchor_lines: int
    max_context_blob_bytes: int
    max_context_search_identifiers: int
    max_context_hits_per_identifier: int
    max_model_request_bytes: int
    max_model_evidence_rounds: int
    repeat_count: int
    environment_kind: EnvironmentKind
    groq_api_key: None = field(default=None, init=False, repr=False)
    github_token: None = field(default=None, init=False, repr=False)


def _groq_api_key(settings: Settings) -> str | None:
    return _PROCESS_SECRETS.get(id(settings), (None, None))[0]


def _github_token(settings: Settings) -> str | None:
    return _PROCESS_SECRETS.get(id(settings), (None, None))[1]


Settings.groq_api_key = property(_groq_api_key)  # type: ignore[attr-defined]
Settings.github_token = property(_github_token)  # type: ignore[attr-defined]
