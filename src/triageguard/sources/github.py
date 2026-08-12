"""Narrow, read-only GitHub support for OpenMRS Core pull requests."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
)

from triageguard.domain.models import ResearchArtifact

_API_ROOT = "https://api.github.com/repos/openmrs/openmrs-core"
_TIMEOUT_SECONDS = (5.0, 20.0)
_SHA_PATTERN = r"^[0-9a-f]{40}$"


class GitHubReadError(RuntimeError):
    """A safe GitHub-read failure that excludes bodies, tokens, and arbitrary URLs."""

    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.reason_code = reason_code
        self.safe_message = safe_message


class GitHubResponseProvenance(ResearchArtifact):
    """Selected safe response metadata retained for reproducibility."""

    etag: StrictStr | None = Field(default=None, max_length=512)
    rate_limit_remaining: StrictInt | None = Field(default=None, ge=0)
    rate_limit_reset: StrictInt | None = Field(default=None, ge=0)
    api_version: StrictStr = Field(min_length=1)


class GitHubRepositoryMetadata(ResearchArtifact):
    """Validated repository metadata required before snapshot acquisition."""

    default_branch: StrictStr = Field(min_length=1)
    response_provenance: GitHubResponseProvenance


class GitHubPullMetadata(ResearchArtifact):
    """Validated pull-request metadata from one fixed OpenMRS Core endpoint."""

    number: StrictInt = Field(gt=0)
    html_url: StrictStr = Field(min_length=1)
    state: StrictStr = Field(min_length=1)
    base_branch: StrictStr = Field(min_length=1)
    base_sha: StrictStr = Field(pattern=_SHA_PATTERN)
    head_sha: StrictStr = Field(pattern=_SHA_PATTERN)
    mergeable: StrictBool | None
    merge_commit_sha: StrictStr | None = Field(default=None, pattern=_SHA_PATTERN)
    observed_at: datetime
    response_provenance: GitHubResponseProvenance


class _RepositoryPayload(BaseModel):
    """Permissive boundary for GitHub's larger response object."""

    model_config = ConfigDict(extra="ignore")

    default_branch: StrictStr = Field(min_length=1)


class _PullSidePayload(BaseModel):
    """The small part of GitHub's base/head payload that this milestone needs."""

    model_config = ConfigDict(extra="ignore")

    ref: StrictStr = Field(min_length=1)
    sha: StrictStr = Field(pattern=_SHA_PATTERN)


class _PullHeadPayload(BaseModel):
    """The one head-side field needed for frozen snapshot identity."""

    model_config = ConfigDict(extra="ignore")

    sha: StrictStr = Field(pattern=_SHA_PATTERN)


class _PullPayload(BaseModel):
    """Permissive boundary for GitHub's larger pull-request response object."""

    model_config = ConfigDict(extra="ignore")

    number: StrictInt = Field(gt=0)
    html_url: StrictStr = Field(min_length=1)
    state: StrictStr = Field(min_length=1)
    base: _PullSidePayload
    head: _PullHeadPayload
    mergeable: StrictBool | None
    merge_commit_sha: StrictStr | None = Field(default=None, pattern=_SHA_PATTERN)


def parse_openmrs_pr_url(value: str) -> int:
    """Return the PR number only for the one canonical OpenMRS Core URL shape."""
    if not isinstance(value, str):
        raise TypeError("OpenMRS Core pull request URL must be a string")

    parsed = urlsplit(value)
    match = re.fullmatch(
        r"/openmrs/openmrs-core/pull/([1-9][0-9]*)",
        parsed.path,
    )

    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        raise ValueError("URL must be a canonical OpenMRS Core pull request URL")

    return int(match.group(1))


class GitHubClient:
    """Read validated metadata only from the fixed OpenMRS Core GitHub API."""

    def __init__(
        self,
        *,
        session: Any | None = None,
        token: str | None = None,
        api_version: str = "2026-03-10",
    ) -> None:
        if token is not None and not isinstance(token, str):
            raise TypeError("GitHub token must be a string or None")
        if not isinstance(api_version, str) or not api_version:
            raise ValueError("GitHub API version must be a non-empty string")

        self._session = session if session is not None else requests.Session()
        self._token = token
        self._api_version = api_version

    def __repr__(self) -> str:
        return (
            "GitHubClient("
            f"api_version={self._api_version!r}, "
            f"token_configured={self._token is not None}"
            ")"
        )

    def get_repository(self) -> GitHubRepositoryMetadata:
        """Read the fixed repository's default branch."""
        payload, provenance = self._get(_API_ROOT)

        try:
            parsed = _RepositoryPayload.model_validate(payload)
        except ValidationError as error:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned invalid repository metadata.",
            ) from error

        return GitHubRepositoryMetadata(
            default_branch=parsed.default_branch,
            response_provenance=provenance,
        )

    def get_pull(self, number: int) -> GitHubPullMetadata:
        """Read one positive-numbered pull request from the fixed repository."""
        if isinstance(number, bool) or not isinstance(number, int):
            raise TypeError("pull request number must be an integer")
        if number <= 0:
            raise ValueError("pull request number must be positive")

        payload, provenance = self._get(f"{_API_ROOT}/pulls/{number}")

        try:
            parsed = _PullPayload.model_validate(payload)
        except ValidationError as error:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned invalid pull-request metadata.",
            ) from error

        if parsed.number != number:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned metadata for a different pull request.",
            )

        try:
            returned_url_number = parse_openmrs_pr_url(parsed.html_url)
        except (TypeError, ValueError) as error:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned invalid pull-request metadata.",
            ) from error

        if returned_url_number != number:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned metadata for a different pull request.",
            )

        return GitHubPullMetadata(
            number=parsed.number,
            html_url=parsed.html_url,
            state=parsed.state,
            base_branch=parsed.base.ref,
            base_sha=parsed.base.sha,
            head_sha=parsed.head.sha,
            mergeable=parsed.mergeable,
            merge_commit_sha=parsed.merge_commit_sha,
            observed_at=datetime.now(UTC),
            response_provenance=provenance,
        )

    def _get(
        self,
        url: str,
    ) -> tuple[dict[str, object], GitHubResponseProvenance]:
        try:
            response = self._session.get(
                url,
                headers=self._headers(),
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub metadata request could not be completed.",
            ) from error

        self._raise_for_non_success(response)

        try:
            payload = response.json()
        except ValueError as error:
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned a non-JSON metadata response.",
            ) from error

        if not isinstance(payload, dict):
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned an invalid metadata response.",
            )

        return payload, self._provenance(response.headers)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self._api_version,
            "User-Agent": "TriageGuard-Research",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _provenance(self, headers: Any) -> GitHubResponseProvenance:
        etag = headers.get("ETag")
        if etag is not None and (not isinstance(etag, str) or len(etag) > 512):
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned invalid response provenance.",
            )

        return GitHubResponseProvenance(
            etag=etag,
            rate_limit_remaining=self._integer_header(
                headers,
                "X-RateLimit-Remaining",
            ),
            rate_limit_reset=self._integer_header(
                headers,
                "X-RateLimit-Reset",
            ),
            api_version=self._api_version,
        )

    @staticmethod
    def _integer_header(headers: Any, name: str) -> int | None:
        value = headers.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.isdecimal():
            raise GitHubReadError(
                "github_read_failed",
                "GitHub returned invalid rate-limit metadata.",
            )
        return int(value)

    @staticmethod
    def _raise_for_non_success(response: Any) -> None:
        if response.status_code == 200:
            return

        remaining = response.headers.get("X-RateLimit-Remaining")
        if response.status_code == 429 or (
            response.status_code == 403 and remaining == "0"
        ):
            raise GitHubReadError(
                "github_rate_limited",
                "GitHub rate limit prevented metadata retrieval.",
            )
        if response.status_code == 404:
            raise GitHubReadError(
                "pr_not_found",
                "The requested OpenMRS Core pull request was not found.",
            )
        raise GitHubReadError(
            "github_read_failed",
            "GitHub metadata retrieval failed.",
        )
