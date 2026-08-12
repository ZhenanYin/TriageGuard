from dataclasses import asdict

import pytest

from triageguard import config as config_module
from triageguard.config import MAX_REPEAT_COUNT, MIN_REPEAT_COUNT, Settings


def test_live_mode_requires_groq_key(monkeypatch):
    """A missing live credential must not be treated as a replay setting."""
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "live")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Settings.from_env()


def test_replay_mode_does_not_require_key(monkeypatch):
    """Offline replay must remain usable without credentials."""
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "replay")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.llm_model == "openai/gpt-oss-120b"


def test_replay_mode_discards_an_inherited_groq_key(monkeypatch) -> None:
    """A developer shell key must not enter replay settings or their representation."""
    secret = "inherited-replay-secret"
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "replay")
    monkeypatch.setenv("GROQ_API_KEY", secret)

    settings = Settings.from_env()

    assert settings.groq_api_key is None
    assert secret not in repr(settings)


def test_replay_discards_both_provider_secrets(monkeypatch) -> None:
    """Replay configuration must not retain either process-only provider secret."""
    monkeypatch.setenv("TRIAGEGUARD_LLM_MODE", "replay")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    settings = Settings.from_env()

    assert settings.groq_api_key is None
    assert settings.github_token is None
    assert "secret" not in repr(settings)


def test_context_settings_have_documented_bounded_defaults() -> None:
    """Unconfigured context collection must remain reproducibly bounded."""
    settings = Settings()

    assert settings.github_api_version == "2026-03-10"
    assert settings.max_context_files == 40
    assert settings.max_context_anchors == 80
    assert settings.max_context_bytes == 160_000
    assert settings.max_context_anchor_lines == 120
    assert settings.max_context_blob_bytes == 1_000_000
    assert settings.max_context_search_identifiers == 100
    assert settings.max_context_hits_per_identifier == 20


def test_diff_settings_have_generous_bounded_defaults() -> None:
    """Raw diffs stay bounded without rejecting ordinary large code changes."""
    settings = Settings()
    public_settings = settings.public_view()

    assert settings.max_diff_files == 1_000
    assert settings.max_diff_bytes == 25_000_000
    assert public_settings.max_diff_files == 1_000
    assert public_settings.max_diff_bytes == 25_000_000


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_diff_files", 0),
        ("max_diff_bytes", 0),
        ("max_diff_files", -1),
        ("max_diff_bytes", -1),
        ("max_diff_files", True),
        ("max_diff_bytes", True),
    ],
)
def test_diff_settings_require_positive_strict_integers(
    field_name: str,
    value: object,
) -> None:
    """A disabled or nonnumeric diff bound would remove the safety boundary."""
    with pytest.raises(ValueError, match=field_name):
        Settings(**{field_name: value})


def test_invalid_mode_is_rejected_before_secret_lookup(monkeypatch) -> None:
    """Mode validation must precede any attempt to read credential material."""
    real_getenv = config_module.os.getenv
    lookups: list[str] = []

    def tracked_getenv(name: str, default=None):
        lookups.append(name)
        if name == "GROQ_API_KEY":
            raise AssertionError("secret lookup occurred before mode validation")
        if name == "TRIAGEGUARD_LLM_MODE":
            return "invalid-mode"
        return real_getenv(name, default)

    monkeypatch.setattr(config_module.os, "getenv", tracked_getenv)

    with pytest.raises(ValueError, match="TRIAGEGUARD_LLM_MODE"):
        Settings.from_env()

    assert lookups == ["TRIAGEGUARD_LLM_MODE"]


def test_live_secret_is_non_printable() -> None:
    """Diagnostic representations must never disclose the live credential."""
    secret = "live-secret-never-print"

    settings = Settings(llm_mode="live", groq_api_key=secret)

    assert settings.groq_api_key == secret
    assert secret not in repr(settings)


def test_live_secrets_are_absent_from_dataclass_serialization() -> None:
    """A generic dataclass serializer must not expose process-only credentials."""
    settings = Settings(
        llm_mode="live",
        groq_api_key="groq-live-secret",
        github_token="github-live-secret",
    )

    serialized = asdict(settings)

    assert "groq_api_key" not in serialized
    assert "github_token" not in serialized
    assert "secret" not in repr(serialized)


def test_value_equal_settings_isolate_process_only_secrets() -> None:
    """Two equivalent public settings objects must never share provider credentials."""
    first = Settings(
        llm_mode="live", groq_api_key="groq-first", github_token="github-first"
    )
    second = Settings(
        llm_mode="live", groq_api_key="groq-second", github_token="github-second"
    )

    assert first == second
    assert (first.groq_api_key, first.github_token) == ("groq-first", "github-first")
    assert (second.groq_api_key, second.github_token) == (
        "groq-second",
        "github-second",
    )


def test_repeat_count_must_be_at_least_one(monkeypatch):
    """A zero-repeat experiment would create evidence without repetitions."""
    monkeypatch.setenv("TRIAGEGUARD_REPEAT_COUNT", "0")

    with pytest.raises(ValueError, match="repeat_count"):
        Settings.from_env()


@pytest.mark.parametrize("value", [0, 21, True, False, "3"])
def test_settings_rejects_non_strict_or_out_of_range_repeat_counts(value) -> None:
    with pytest.raises(
        ValueError, match="repeat_count must be an integer from 1 to 20"
    ):
        Settings(repeat_count=value)


@pytest.mark.parametrize("value", [MIN_REPEAT_COUNT, MAX_REPEAT_COUNT])
def test_settings_accepts_both_repeat_count_boundaries(value: int) -> None:
    assert Settings(repeat_count=value).repeat_count == value


def test_repeat_count_above_shared_bound_fails_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TRIAGEGUARD_REPEAT_COUNT", "21")

    with pytest.raises(
        ValueError, match="repeat_count must be an integer from 1 to 20"
    ):
        Settings.from_env()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "replay", "GROQ"])
def test_settings_rejects_every_non_groq_provider(provider: str) -> None:
    with pytest.raises(ValueError, match="TRIAGEGUARD_LLM_PROVIDER must be 'groq'"):
        Settings(llm_provider=provider)


def test_non_groq_environment_fails_before_settings_are_constructed(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRIAGEGUARD_LLM_PROVIDER", "openai")

    with pytest.raises(ValueError, match="TRIAGEGUARD_LLM_PROVIDER must be 'groq'"):
        Settings.from_env()
