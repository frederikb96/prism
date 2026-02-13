#!/usr/bin/env python3
"""
Time-aware hook for CLI providers.

Supports both Claude (PreToolUse/PostToolUse) and Gemini (BeforeTool/AfterTool).
Format selected via PRISM_HOOK_FORMAT env var ("claude" or "gemini").

Pre hooks: block if time expired, inject remaining time if not.
Post hooks: always allow, inject remaining time.

Logs each invocation to PRISM_HOOK_LOG (default: /tmp/hook-fired.log) as JSON lines.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _read_tool_name() -> str:
    """Extract tool name from stdin (JSON payload from CLI hook system)."""
    try:
        if not sys.stdin.isatty():
            data = json.loads(sys.stdin.read())
            return data.get("tool_name", "")
    except (json.JSONDecodeError, OSError):
        pass
    return ""


def _write_log(log_file: str, hook_type: str, fmt: str, decision: str, remaining: int, tool_name: str = "") -> None:
    try:
        entry: dict[str, Any] = {
            "hook": hook_type,
            "format": fmt,
            "decision": decision,
            "remaining_s": remaining,
            "time": time.time(),
        }
        if tool_name:
            entry["tool_name"] = tool_name
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def main() -> int:
    hook_type = sys.argv[1] if len(sys.argv) > 1 else "post"
    fmt = os.environ.get("PRISM_HOOK_FORMAT", "claude")
    tool_name = _read_tool_name()
    log_file = os.environ.get("PRISM_HOOK_LOG", "/tmp/hook-fired.log")

    start_time_str = os.environ.get("PRISM_START_TIME")
    timeout_str = os.environ.get("PRISM_TOOL_TIMEOUT")

    if not start_time_str or not timeout_str:
        _write_log(log_file, hook_type, fmt, "allow", -1, tool_name)
        print(json.dumps({"decision": "allow"} if fmt == "gemini" else {}))
        return 0

    try:
        start_time = float(start_time_str)
        timeout = int(timeout_str)
    except ValueError:
        _write_log(log_file, hook_type, fmt, "allow", -1, tool_name)
        print(json.dumps({"decision": "allow"} if fmt == "gemini" else {}))
        return 0

    elapsed = int(time.time() - start_time)
    remaining = max(0, timeout - elapsed)

    # Pre hook can block; post hook always allows
    if hook_type == "pre" and remaining <= 0:
        _write_log(log_file, hook_type, fmt, "block", remaining, tool_name)
        reason = (
            f"TIME EXPIRED ({elapsed}s of {timeout}s used). "
            "Do NOT attempt more searches. Write your final answer NOW."
        )
        if fmt == "gemini":
            print(json.dumps({"decision": "block", "reason": reason}))
        else:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }))
        return 0

    # Allow with time context
    _write_log(log_file, hook_type, fmt, "allow", remaining, tool_name)
    if remaining <= 0 and hook_type == "post":
        msg = (
            f"⚠️ TIME EXPIRED: 0s remaining of {timeout}s ({elapsed}s elapsed). "
            "Do NOT make more tool calls. Write your final answer NOW."
        )
    elif remaining <= 10:
        msg = f"⚠️ LOW TIME: {remaining}s remaining of {timeout}s ({elapsed}s elapsed)"
    else:
        msg = f"⏱️ {remaining}s remaining of {timeout}s ({elapsed}s elapsed)"

    hook_event = "PreToolUse" if hook_type == "pre" else "PostToolUse"

    if fmt == "gemini":
        print(json.dumps({
            "decision": "allow",
            "hookSpecificOutput": {"additionalContext": msg},
        }))
    else:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": hook_event,
                "additionalContext": msg,
            }
        }))

    return 0


if __name__ == "__main__":
    sys.exit(main())
