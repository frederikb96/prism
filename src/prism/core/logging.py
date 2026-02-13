"""
JSON-lines logging infrastructure for Prism.

Provides structured JSON logging to stdout for container log collection.
All Prism and library loggers emit one JSON object per line.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STANDARD_RECORD_ATTRS = frozenset({
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
})


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                entry[key] = value

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            entry["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(entry, default=str)


_LIBRARY_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "asyncio",
    "fastmcp",
    "httpx",
)

# Extremely verbose loggers that should be suppressed to WARNING
_NOISY_LOGGERS = (
    "docket",
    "docket.worker",
    "fakeredis",
)


def setup_logging(level: str) -> None:
    """
    Configure root logger with JSON formatter and redirect library loggers.

    Must be called before any other module logs (typically in __main__.py).

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Prevent uvicorn from overriding our logging config during startup
    try:
        import uvicorn.config

        uvicorn.config.LOGGING_CONFIG = {  # type: ignore[assignment]
            "version": 1,
            "disable_existing_loggers": False,
        }
    except ImportError:
        pass

    for name in _LIBRARY_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_hook_log(log_path: str | Path) -> dict[str, int]:
    """
    Parse a hook JSONL log file and extract execution metrics.

    Each line is a JSON object written by the time hook script during
    Claude CLI execution. Pre-hook events -> tool_calls, block decisions -> hook_blocks.

    Args:
        log_path: Path to the JSONL hook log file

    Returns:
        Dict with tool_calls, hook_blocks, and total_events counts
    """
    try:
        text = Path(log_path).read_text().strip()
    except OSError:
        return {"tool_calls": 0, "hook_blocks": 0, "total_events": 0}

    if not text:
        return {"tool_calls": 0, "hook_blocks": 0, "total_events": 0}

    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    pre_events = [e for e in events if e.get("hook") == "pre"]
    blocks = [e for e in events if e.get("decision") == "block"]

    return {
        "tool_calls": len(pre_events),
        "hook_blocks": len(blocks),
        "total_events": len(events),
    }


def parse_hook_log_detailed(
    log_path: str | Path,
    start_time: float,
) -> dict[str, Any]:
    """
    Parse hook JSONL log with per-tool-call timing detail.

    Pairs pre/post hook events sequentially to compute per-call timing
    relative to the worker start time.

    Args:
        log_path: Path to the JSONL hook log file
        start_time: Unix timestamp (time.time()) when the worker started

    Returns:
        Dict with tool_calls, hook_blocks, and calls list with per-call details
    """
    empty: dict[str, Any] = {"tool_calls": 0, "hook_blocks": 0, "calls": []}

    try:
        text = Path(log_path).read_text().strip()
    except OSError:
        return empty

    if not text:
        return empty

    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not events:
        return empty

    # Separate pre and post events, then pair by order.
    # Parallel tool calls produce multiple pre events before their post events,
    # so strict sequential (i, i+1) pairing fails. Instead, collect all
    # non-blocked pre events and all post events, then pair them in order
    # (first pre with first post, etc.) since hooks fire chronologically.
    pre_events: list[dict[str, Any]] = []
    post_events: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    for event in events:
        hook = event.get("hook")
        if hook == "pre":
            if event.get("decision") == "block":
                # Blocked calls have no corresponding post event
                tool = event.get("tool_name", "unknown")
                event_time = event.get("time", start_time)
                calls.append({
                    "tool": tool,
                    "start_s": round(event_time - start_time, 1),
                    "end_s": None,
                    "duration_s": None,
                    "blocked": True,
                    "remaining_s": event.get("remaining_s"),
                })
            else:
                pre_events.append(event)
        elif hook == "post":
            post_events.append(event)

    # Pair pre/post by chronological order
    for idx, pre in enumerate(pre_events):
        tool = pre.get("tool_name", "unknown")
        event_time = pre.get("time", start_time)
        remaining = pre.get("remaining_s")

        if idx < len(post_events):
            post = post_events[idx]
            post_time = post.get("time", event_time)
            calls.append({
                "tool": tool,
                "start_s": round(event_time - start_time, 1),
                "end_s": round(post_time - start_time, 1),
                "duration_s": round(post_time - event_time, 1),
                "blocked": False,
                "remaining_s": remaining,
            })
        else:
            # No matching post event (process killed, or post not logged)
            calls.append({
                "tool": tool,
                "start_s": round(event_time - start_time, 1),
                "end_s": None,
                "duration_s": None,
                "blocked": False,
                "remaining_s": remaining,
            })

    # Sort by start_s for chronological display
    calls.sort(key=lambda c: c.get("start_s", 0))

    return {
        "tool_calls": sum(1 for c in calls if not c["blocked"]),
        "hook_blocks": sum(1 for c in calls if c["blocked"]),
        "calls": calls,
    }


def log_worker_completion(
    *,
    worker_type: str,
    agent_key: str,
    success: bool,
    wall_time_s: float,
    model: str,
    response_length: int,
    tool_calls: int = 0,
    hook_blocks: int = 0,
    tool_call_details: list[dict[str, Any]] | None = None,
) -> None:
    """
    Emit a structured worker completion log record.

    Args:
        worker_type: Agent type identifier
        agent_key: Unique key for this worker instance (e.g. claude_search_1)
        success: Whether the worker completed successfully
        wall_time_s: Wall clock time in seconds
        model: Model used
        response_length: Length of the response content
        tool_calls: Number of tool calls observed (from hook log)
        hook_blocks: Number of blocked tool calls (from hook log)
        tool_call_details: Per-tool-call timing detail list
    """
    logger.info(
        "Worker completed",
        extra={
            "worker_type": worker_type,
            "agent_key": agent_key,
            "success": success,
            "wall_time_s": round(wall_time_s, 2),
            "model": model,
            "response_length": response_length,
            "tool_calls": tool_calls,
            "hook_blocks": hook_blocks,
            "tool_call_details": tool_call_details or [],
        },
    )


def log_manager_phase(
    *,
    phase: str,
    level: int,
    wall_time_s: float,
    session_id: str | None = None,
) -> None:
    """
    Emit a structured manager phase timing log record.

    Args:
        phase: Phase name ("planning" or "synthesis")
        level: Search level (1-3)
        wall_time_s: Wall clock time in seconds
        session_id: Manager's Claude session ID
    """
    logger.info(
        "Manager phase completed",
        extra={
            "phase": phase,
            "search_level": level,
            "wall_time_s": round(wall_time_s, 2),
            "session_id": session_id,
        },
    )


def log_prompt(
    *,
    prompt_type: str,
    prompt: str,
    level: int,
    session_id: str | None = None,
) -> None:
    """
    Emit full prompt content at DEBUG level for E2E result inspection.

    Args:
        prompt_type: Prompt type ("search_planning", "synthesis", "follow_up_chat")
        prompt: Full rendered prompt text
        level: Search level
        session_id: Manager's Claude session ID
    """
    logger.debug(
        "Manager prompt",
        extra={
            "prompt_type": prompt_type,
            "prompt": prompt,
            "search_level": level,
            "session_id": session_id,
        },
    )
