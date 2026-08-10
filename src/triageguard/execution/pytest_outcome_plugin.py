"""Trusted structured pytest outcome capture for the execution subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _pytest.stash import StashKey

_REPORTS_KEY = StashKey[list[dict[str, Any]]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--triageguard-outcome-path", required=True)


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_REPORTS_KEY] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Any:
    outcome = yield
    report = outcome.get_result()
    exception_type: str | None = None
    frames: list[dict[str, str | int]] = []
    if call.excinfo is not None:
        exception_type = call.excinfo.typename
        frames = [
            {
                "path": str(Path(entry.path).resolve()),
                "lineno": entry.lineno + 1,
                "function": entry.frame.code.name,
            }
            for entry in call.excinfo.traceback
        ]
    item.config.stash[_REPORTS_KEY].append(
        {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "exception_type": exception_type,
            "frames": frames,
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    destination = Path(
        session.config.getoption("--triageguard-outcome-path")
    )
    serialized = json.dumps(
        {
            "exitstatus": int(exitstatus),
            "reports": session.config.stash[_REPORTS_KEY],
        },
        indent=2,
        sort_keys=True,
    )
    with destination.open("x", encoding="utf-8") as outcome_file:
        outcome_file.write(f"{serialized}\n")
