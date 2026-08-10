"""Typed runtime configuration for the isolated V2 prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from triageguard.domain.statuses import EnvironmentKind

MIN_REPEAT_COUNT = 1
MAX_REPEAT_COUNT = 20


@dataclass(frozen=True)
class Settings:
    """Configuration sourced explicitly from the process environment."""

    llm_mode: Literal["live", "replay"] = "replay"
    llm_provider: Literal["groq"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: str | None = field(default=None, repr=False)
    artifacts_dir: Path = Path("artifacts")
    repeat_count: int = 3
    environment_kind: EnvironmentKind = EnvironmentKind.CONTROLLED_FIXTURE

    def __post_init__(self) -> None:
        if self.llm_mode not in {"live", "replay"}:
            raise ValueError("TRIAGEGUARD_LLM_MODE must be 'live' or 'replay'")
        if self.llm_mode == "replay" and self.groq_api_key is not None:
            object.__setattr__(self, "groq_api_key", None)
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
        if self.llm_mode == "live" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when TRIAGEGUARD_LLM_MODE=live")

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

        return cls(
            llm_mode=llm_mode,
            llm_provider=llm_provider,
            llm_model=os.getenv("TRIAGEGUARD_LLM_MODEL", "openai/gpt-oss-120b"),
            groq_api_key=groq_api_key,
            artifacts_dir=Path(os.getenv("TRIAGEGUARD_ARTIFACTS_DIR", "artifacts")),
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
    repeat_count: int
    environment_kind: EnvironmentKind
    groq_api_key: None = field(default=None, init=False, repr=False)
