"""Log analysis utilities for E2E tests."""

from __future__ import annotations

import json
import re
import subprocess


def check_container_logs(container_name: str) -> list[str]:
    """
    Get container logs and parse for ERROR/WARNING entries.

    Args:
        container_name: Podman container name to check

    Returns:
        List of issue strings found in logs (ERROR or WARNING lines)
    """
    try:
        result = subprocess.run(
            ["podman", "logs", "--tail", "500", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Combine stdout and stderr (logs may be in either)
        logs = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ["Timeout getting container logs"]
    except FileNotFoundError:
        return ["Podman command not found"]
    except Exception as e:
        return [f"Error getting logs: {e}"]

    issues: list[str] = []

    # Pattern matches common log formats:
    # - Python logging: "ERROR:" or "WARNING:"
    # - JSON logs: '"level":"error"' or '"level":"warning"'
    error_pattern = re.compile(r"\bERROR\b|\"level\":\s*\"error\"", re.IGNORECASE)
    warning_pattern = re.compile(r"\bWARNING\b|\"level\":\s*\"warning\"", re.IGNORECASE)

    for line in logs.splitlines():
        line = line.strip()
        if not line:
            continue

        if error_pattern.search(line):
            issues.append(f"ERROR: {line[:200]}")
        elif warning_pattern.search(line):
            issues.append(f"WARNING: {line[:200]}")

    return issues


def find_hook_blocks_in_logs(container_name: str) -> int:
    """
    Parse container JSON logs for worker completion entries with hook_blocks > 0.

    The Prism server emits structured JSON log lines via log_worker_completion().
    Each line includes a hook_blocks field when tools were blocked by the time hook.

    Args:
        container_name: Podman container name to check

    Returns:
        Total hook_blocks count across all log entries
    """
    try:
        result = subprocess.run(
            ["podman", "logs", "--tail", "1000", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        logs = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return 0

    total_blocks = 0
    for line in logs.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        blocks = entry.get("hook_blocks", 0)
        if isinstance(blocks, int) and blocks > 0:
            total_blocks += blocks

    return total_blocks
