"""
Shared parsing utilities for Claude CLI output.
"""

from __future__ import annotations

import json


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
