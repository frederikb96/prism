"""
Hooks configuration builder for time-aware Claude agents.

Provides utilities to build Claude CLI hooks configuration
and environment variables for time tracking.
"""

from __future__ import annotations

from typing import Any

HOOK_SCRIPT_PATH = "/app/hooks/time_tracker.py"


def build_time_awareness_hooks() -> dict[str, Any]:
    """
    Build hooks configuration for time-aware agents.

    Configures PreToolUse and PostToolUse hooks to inject
    elapsed time information into Claude's context.

    Returns:
        Hooks configuration dict for Claude --settings flag
    """
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": ".*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {HOOK_SCRIPT_PATH} pre",
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": ".*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {HOOK_SCRIPT_PATH} post",
                        }
                    ],
                }
            ],
        }
    }


def build_time_env_vars(
    start_time: float,
    visible_timeout: int,
) -> tuple[tuple[str, str], ...]:
    """
    Build environment variables for time tracking hooks.

    Args:
        start_time: Unix timestamp when agent started
        visible_timeout: Timeout in seconds that agent sees

    Returns:
        Tuple of (key, value) pairs for subprocess environment
    """
    return (
        ("PRISM_START_TIME", str(start_time)),
        ("PRISM_VISIBLE_TIMEOUT", str(visible_timeout)),
    )
