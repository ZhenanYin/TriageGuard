"""Tests for the read-only OpenMRS Core GitHub metadata client."""

import pytest

from triageguard.sources.github import (
    GitHubClient,
    GitHubReadError,
    parse_openmrs_pr_url,
)


def test_pr_url_parser_accepts_the_one_supported_openmrs_core_url() -> None:
    """The canonical OpenMRS Core pull-request URL yields its positive PR number."""
    assert (
        parse_openmrs_pr_url("https://github.com/openmrs/openmrs-core/pull/7312")
        == 7312
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/openmrs/openmrs-core/pull/1",
        "https://evil.example/openmrs/openmrs-core/pull/1",
        "https://github.com/openmrs/openmrs-core/issues/1",
        "https://github.com/openmrs/openmrs-core/pull/0",
        "https://github.com/openmrs/openmrs-core/pull/1?diff=split",
        "https://github.com/openmrs/openmrs-core/pull/1/",
    ],
)
def test_pr_url_parser_rejects_every_noncanonical_input(value: str) -> None:
    """Only the one exact pull-request URL shape is safe to analyze."""
    with pytest.raises(ValueError, match="OpenMRS Core pull request"):
        parse_openmrs_pr_url(value)


class FakeResponse:
    """Small deterministic stand-in for one GitHub HTTP response."""

    def __init__(
        self,
        status_code: int,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    """Records client requests and returns preplanned responses in order."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: tuple[float, float],
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self._responses.pop(0)


def test_client_preserves_unknown_mergeability_without_leaking_token() -> None:
    """GitHub's temporary mergeability unknown state must remain intact."""
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "number": 7312,
                    "html_url": ("https://github.com/openmrs/openmrs-core/pull/7312"),
                    "state": "open",
                    "base": {"ref": "main", "sha": "2" * 40},
                    "head": {"sha": "3" * 40},
                    "mergeable": None,
                    "merge_commit_sha": None,
                },
                {
                    "ETag": '"pull-etag"',
                    "X-RateLimit-Remaining": "42",
                    "X-RateLimit-Reset": "1725000000",
                },
            )
        ]
    )
    client = GitHubClient(session=session, token="never-record-me")

    metadata = client.get_pull(7312)

    assert metadata.number == 7312
    assert metadata.mergeable is None
    assert metadata.merge_commit_sha is None
    assert metadata.response_provenance.etag == '"pull-etag"'
    assert metadata.response_provenance.rate_limit_remaining == 42
    assert metadata.response_provenance.rate_limit_reset == 1725000000
    assert session.calls == [
        {
            "url": ("https://api.github.com/repos/openmrs/openmrs-core/pulls/7312"),
            "headers": {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "TriageGuard-Research",
                "Authorization": "Bearer never-record-me",
            },
            "timeout": (5.0, 20.0),
        }
    ]
    assert "never-record-me" not in repr(client)


def test_client_rejects_metadata_for_a_different_pull_number() -> None:
    """The fixed PR endpoint must not be confused with another pull request."""
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "number": 7313,
                    "html_url": ("https://github.com/openmrs/openmrs-core/pull/7313"),
                    "state": "open",
                    "base": {"ref": "main", "sha": "2" * 40},
                    "head": {"sha": "3" * 40},
                    "mergeable": True,
                    "merge_commit_sha": "4" * 40,
                },
            )
        ]
    )
    client = GitHubClient(session=session)

    with pytest.raises(GitHubReadError, match="different pull request") as error:
        client.get_pull(7312)

    assert error.value.reason_code == "github_read_failed"


def test_client_rejects_noncanonical_pull_url_in_metadata() -> None:
    """GitHub metadata cannot introduce an arbitrary pull-request URL."""
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "number": 7312,
                    "html_url": "https://evil.example/not-an-openmrs-pr",
                    "state": "open",
                    "base": {"ref": "main", "sha": "2" * 40},
                    "head": {"sha": "3" * 40},
                    "mergeable": True,
                    "merge_commit_sha": "4" * 40,
                },
            )
        ]
    )
    client = GitHubClient(session=session)

    with pytest.raises(GitHubReadError, match="invalid pull-request metadata") as error:
        client.get_pull(7312)

    assert error.value.reason_code == "github_read_failed"


def test_client_reads_default_branch_from_the_fixed_openmrs_core_endpoint() -> None:
    """Repository metadata always comes from the one supported repository."""
    session = FakeSession(
        [
            FakeResponse(
                200,
                {"default_branch": "main"},
                {"ETag": '"repository-etag"'},
            )
        ]
    )
    client = GitHubClient(session=session)

    repository = client.get_repository()

    assert repository.default_branch == "main"
    assert repository.response_provenance.etag == '"repository-etag"'
    assert session.calls[0]["url"] == (
        "https://api.github.com/repos/openmrs/openmrs-core"
    )
    assert session.calls[0]["timeout"] == (5.0, 20.0)


@pytest.mark.parametrize(
    ("status_code", "headers", "reason_code"),
    [
        (
            429,
            {"X-RateLimit-Remaining": "0"},
            "github_rate_limited",
        ),
        (
            404,
            {},
            "pr_not_found",
        ),
    ],
)
def test_client_maps_safe_github_errors_without_exposing_response_data(
    status_code: int,
    headers: dict[str, str],
    reason_code: str,
) -> None:
    """HTTP failures become safe reason codes rather than copied server messages."""
    session = FakeSession(
        [
            FakeResponse(
                status_code,
                {"message": "server-body-secret-must-not-appear"},
                headers,
            )
        ]
    )
    client = GitHubClient(session=session, token="token-must-not-appear")

    with pytest.raises(GitHubReadError) as error:
        client.get_pull(7312)

    assert error.value.reason_code == reason_code
    assert "server-body-secret-must-not-appear" not in str(error.value)
    assert "token-must-not-appear" not in str(error.value)
