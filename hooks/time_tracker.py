#!/usr/bin/env python3
"""
Time awareness hook for Prism Claude agents.

Injects elapsed time information into Claude's context during tool calls.
Reads timing configuration from environment variables set by Prism.
"""

from __future__ import annotations

import json
import os
import sys
import time


def get_time_message(elapsed: int, visible_timeout: int) -> str:
    """
    Format time message based on remaining time.

    Args:
        elapsed: Seconds elapsed since start
        visible_timeout: Total visible timeout in seconds

    Returns:
        Formatted message with appropriate urgency level
    """
    remaining = max(0, visible_timeout - elapsed)

    if remaining <= 5:
        return (
            f"🚨 CRITICAL: {elapsed}s of {visible_timeout}s used. "
            f"Only {remaining}s left! Finish NOW!"
        )
    elif remaining <= 15:
        return (
            f"⚠️ TIME WARNING: {elapsed}s of {visible_timeout}s used. "
            f"{remaining}s remaining - WRAP UP!"
        )
    else:
        return f"⏱️ Time: {elapsed}s of {visible_timeout}s elapsed. {remaining}s remaining."


def main() -> int:
    """
    Main hook entry point.

    Reads PRISM_START_TIME and PRISM_VISIBLE_TIMEOUT from environment,
    calculates elapsed time, and outputs JSON for Claude.

    Returns:
        Exit code (0 for success)
    """
    hook_type = sys.argv[1] if len(sys.argv) > 1 else "post"

    start_time_str = os.environ.get("PRISM_START_TIME")
    visible_timeout_str = os.environ.get("PRISM_VISIBLE_TIMEOUT")

    if not start_time_str or not visible_timeout_str:
        return 0

    try:
        start_time = float(start_time_str)
        visible_timeout = int(visible_timeout_str)
    except ValueError:
        return 0

    elapsed = int(time.time() - start_time)
    message = get_time_message(elapsed, visible_timeout)

    hook_event = "PreToolUse" if hook_type == "pre" else "PostToolUse"

    output = {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": hook_event,
            "additionalContext": message,
        },
    }

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
