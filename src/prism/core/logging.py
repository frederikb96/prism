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


def parse_hook_log(log_path: str | Path) -> dict[str, int]:
    """
    Parse a hook JSONL log file and extract execution metrics.

    Each line is a JSON object written by the time hook script during
    Claude CLI execution. Pre-hook events → tool_calls, block decisions → hook_blocks.

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


def log_worker_completion(
    *,
    worker_type: str,
    success: bool,
    wall_time_s: float,
    model: str,
    response_length: int,
    tool_calls: int = 0,
    hook_blocks: int = 0,
) -> None:
    """
    Emit a structured worker completion log record.

    Args:
        worker_type: Agent type identifier (researcher, tavily, perplexity)
        success: Whether the worker completed successfully
        wall_time_s: Wall clock time in seconds
        model: Claude model used
        response_length: Length of the response content
        tool_calls: Number of tool calls observed (from hook log)
        hook_blocks: Number of blocked tool calls (from hook log)
    """
    logger.info(
        "Worker completed",
        extra={
            "worker_type": worker_type,
            "success": success,
            "wall_time_s": round(wall_time_s, 2),
            "model": model,
            "response_length": response_length,
            "tool_calls": tool_calls,
            "hook_blocks": hook_blocks,
        },
    )
