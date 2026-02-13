"""
Hooks configuration builders for time-aware CLI agents.

Provides utilities to build hook configurations and environment
variables for both Claude CLI and Gemini CLI time tracking.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

HOOK_SCRIPT_PATH = "/app/hooks/time_hook.py"


def build_claude_hooks() -> dict[str, Any]:
    """
    Build Claude CLI hooks configuration for time-aware agents.

    Configures PreToolUse and PostToolUse hooks to inject
    elapsed time information and block when time expires.

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


def build_gemini_settings_file() -> str:
    """
    Write a Gemini CLI settings file for time-aware hooks.

    Creates a temporary JSON file with BeforeTool/AfterTool hooks
    pointing to the time hook script. Caller must clean up the file.

    Returns:
        Path to the temporary settings file
    """
    settings = {
        "hooksConfig": {"enabled": True},
        "hooks": {
            "BeforeTool": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {HOOK_SCRIPT_PATH} pre",
                            "name": "prism-time-before",
                            "timeout": 5000,
                        }
                    ],
                }
            ],
            "AfterTool": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {HOOK_SCRIPT_PATH} post",
                            "name": "prism-time-after",
                            "timeout": 5000,
                        }
                    ],
                }
            ],
        },
    }

    path = f"/tmp/prism-gemini-settings-{uuid.uuid4()}.json"
    with open(path, "w") as f:
        json.dump(settings, f)
    return path


def build_time_env_vars(
    start_time: float,
    tool_timeout: int,
    hook_format: str,
    log_path: str,
) -> tuple[tuple[str, str], ...]:
    """
    Build environment variables for time tracking hooks.

    Args:
        start_time: Unix timestamp when agent started
        tool_timeout: Timeout in seconds that agent sees
        hook_format: Hook format identifier ("claude" or "gemini")
        log_path: Path for the hook JSON-lines log file

    Returns:
        Tuple of (key, value) pairs for subprocess environment
    """
    return (
        ("PRISM_START_TIME", str(start_time)),
        ("PRISM_TOOL_TIMEOUT", str(tool_timeout)),
        ("PRISM_HOOK_FORMAT", hook_format),
        ("PRISM_HOOK_LOG", log_path),
    )
