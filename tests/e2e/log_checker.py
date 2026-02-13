"""
Structured JSON log parser for E2E test result extraction.

Parses container JSON-lines logs and extracts worker completions,
manager phase timings, prompt content, and error/warning entries
scoped to a specific test run's time window.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class WorkerLog:
    """Parsed worker completion log entry."""

    worker_type: str
    agent_key: str
    success: bool
    wall_time_s: float
    model: str
    response_length: int
    tool_calls: int = 0
    hook_blocks: int = 0
    tool_call_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ManagerPhaseLog:
    """Parsed manager phase timing log entry."""

    phase: str
    level: int
    wall_time_s: float
    session_id: str | None = None


@dataclass
class PromptLog:
    """Parsed prompt content log entry (DEBUG level)."""

    prompt_type: str
    prompt: str
    level: int
    session_id: str | None = None


@dataclass
class TestLogs:
    """Aggregated log data for a single test run."""

    workers: list[WorkerLog] = field(default_factory=list)
    manager_phases: list[ManagerPhaseLog] = field(default_factory=list)
    prompts: list[PromptLog] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def fetch_container_logs(container_name: str, tail: int = 10000) -> str:
    """
    Fetch raw container logs via podman.

    Args:
        container_name: Podman container name
        tail: Number of most recent lines to fetch

    Returns:
        Combined stdout + stderr log text
    """
    try:
        result = subprocess.run(
            ["podman", "logs", "--tail", str(tail), container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse ISO 8601 timestamp string to datetime."""
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _in_window(
    ts: datetime,
    window_start: datetime,
    window_end: datetime,
) -> bool:
    """Check if timestamp falls within the window (with 2s buffer each side)."""
    start = window_start - timedelta(seconds=2)
    end = window_end + timedelta(seconds=2)
    return start <= ts <= end


def parse_test_logs(
    container_name: str,
    start_time: datetime,
    end_time: datetime,
    tail: int = 10000,
) -> TestLogs:
    """
    Parse container JSON-lines logs for entries within a test's time window.

    Extracts worker completions, manager phases, prompts, and errors/warnings
    that occurred between start_time and end_time.

    Args:
        container_name: Podman container name
        start_time: Test start time (UTC)
        end_time: Test end time (UTC)
        tail: Number of recent log lines to fetch

    Returns:
        TestLogs with categorized entries
    """
    raw = fetch_container_logs(container_name, tail=tail)
    logs = TestLogs()

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        ts = _parse_timestamp(entry.get("timestamp", ""))
        if ts is None or not _in_window(ts, start_time, end_time):
            continue

        level = entry.get("level", "")
        message = entry.get("message", "")

        if message == "Worker completed":
            logs.workers.append(
                WorkerLog(
                    worker_type=entry.get("worker_type", "unknown"),
                    agent_key=entry.get("agent_key", "unknown"),
                    success=entry.get("success", False),
                    wall_time_s=entry.get("wall_time_s", 0),
                    model=entry.get("model", "unknown"),
                    response_length=entry.get("response_length", 0),
                    tool_calls=entry.get("tool_calls", 0),
                    hook_blocks=entry.get("hook_blocks", 0),
                    tool_call_details=entry.get("tool_call_details", []),
                )
            )
        elif message == "Manager phase completed":
            logs.manager_phases.append(
                ManagerPhaseLog(
                    phase=entry.get("phase", "unknown"),
                    level=entry.get("search_level", 0),
                    wall_time_s=entry.get("wall_time_s", 0),
                    session_id=entry.get("session_id"),
                )
            )
        elif message == "Manager prompt":
            logs.prompts.append(
                PromptLog(
                    prompt_type=entry.get("prompt_type", "unknown"),
                    prompt=entry.get("prompt", ""),
                    level=entry.get("search_level", 0),
                    session_id=entry.get("session_id"),
                )
            )

        if level == "ERROR":
            logs.errors.append(line[:500])
        elif level == "WARNING":
            logs.warnings.append(line[:500])

    return logs
