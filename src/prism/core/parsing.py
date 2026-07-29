"""
Shared parsing utilities for Claude CLI output.
"""

from __future__ import annotations

import json
from typing import Any


def extract_content_from_cli_output(raw_output: str) -> str:
    """
    Extract text content from Claude CLI JSON output.

    Handles the standard Claude CLI response format:
    {"type": "result", "result": "...", "session_id": "..."}

    Args:
        raw_output: Raw JSON output from Claude CLI

    Returns:
        Extracted text content, or raw output if parsing fails
    """
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict):
            return str(data.get("result", raw_output))
    except json.JSONDecodeError:
        pass

    return raw_output


def _describe_error_value(value: Any) -> str | None:
    """Render one candidate error field as text, descending into nested shapes."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("message", "content", "text", "detail", "type"):
            described = _describe_error_value(value.get(key))
            if described:
                return described
        return None
    if isinstance(value, list):
        parts = [described for item in value if (described := _describe_error_value(item))]
        return " ".join(parts) or None
    return None


def cli_failure_message(
    tool: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    max_chars: int,
) -> str:
    """
    Compose the best available failure reason for a CLI run.

    Order matters: the payload carries the actual API error, stderr carries
    warnings that would otherwise masquerade as one, and the exit code says
    nothing at all.

    Args:
        tool: Human-readable CLI name for the last-resort message
        stdout: Raw stdout from the CLI
        stderr: Raw stderr from the CLI
        returncode: Process exit code
        max_chars: Truncation limit for a parsed reason
    """
    return (
        extract_error_from_cli_output(stdout, max_chars)
        or stderr.strip()
        or f"{tool} exited with code {returncode}"
    )


def cli_output_flags_error(raw_output: str) -> bool:
    """
    Report whether a CLI payload declares itself an error.

    The CLIs can surface an API failure while still exiting 0, so the exit
    code alone does not decide whether a run succeeded.
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return False

    return isinstance(data, dict) and bool(data.get("is_error"))


def extract_error_from_cli_output(raw_output: str, max_chars: int) -> str | None:
    """
    Extract a human-readable failure reason from CLI output.

    Both CLIs report why a run failed in their stdout JSON while leaving stderr
    empty, so a bare exit code is all a caller sees unless the payload is read.
    Handles the flat error/result shapes and the nested message-content shape.

    Args:
        raw_output: Raw stdout from the CLI
        max_chars: Truncation limit for the returned reason

    Returns:
        Failure reason if one can be found, None otherwise
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    reasons: list[str] = []
    for key in ("error", "subtype", "message", "result"):
        described = _describe_error_value(data.get(key))
        # A failing payload can still carry subtype "success"; saying so in a
        # failure reason is worse than saying nothing.
        if described and described != "success" and described not in reasons:
            reasons.append(described)

    if not reasons:
        return None

    reason = ": ".join(reasons)
    if len(reason) > max_chars:
        reason = reason[:max_chars] + "..."
    return reason
