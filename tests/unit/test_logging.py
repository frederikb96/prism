"""Tests for JSON-lines logging infrastructure."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from prism.core.logging import (
    JSONFormatter,
    log_worker_completion,
    parse_hook_log,
    setup_logging,
)


class TestJSONFormatter:
    """Test JSON-lines formatter."""

    def _make_record(
        self,
        msg: str = "Test message",
        level: int = logging.INFO,
        name: str = "test.logger",
        **extra: object,
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_basic_fields(self) -> None:
        """Output contains required fields and is valid JSON."""
        formatter = JSONFormatter()
        record = self._make_record()

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert "T" in data["timestamp"]

    def test_extra_fields_included(self) -> None:
        """Extra kwargs from log calls appear in output."""
        formatter = JSONFormatter()
        record = self._make_record(worker_type="researcher", wall_time_s=15.2)

        output = formatter.format(record)
        data = json.loads(output)

        assert data["worker_type"] == "researcher"
        assert data["wall_time_s"] == 15.2

    def test_exception_captured(self) -> None:
        """Exception info is included as a string field."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "test error" in data["exception"]

    def test_standard_attrs_excluded(self) -> None:
        """Standard LogRecord attributes don't leak into output."""
        formatter = JSONFormatter()
        record = self._make_record()

        output = formatter.format(record)
        data = json.loads(output)

        for attr in ("pathname", "lineno", "funcName", "process", "thread", "args"):
            assert attr not in data

    def test_message_formatting(self) -> None:
        """%-style message arguments are applied."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s, count=%d",
            args=("world", 42),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Hello world, count=42"

    def test_non_serializable_uses_str(self) -> None:
        """Non-JSON-serializable objects are converted via str()."""
        formatter = JSONFormatter()
        record = self._make_record(path_extra=Path("/tmp/test"))

        output = formatter.format(record)
        data = json.loads(output)

        assert data["path_extra"] == "/tmp/test"


class TestSetupLogging:
    """Test logging setup and library logger configuration."""

    @pytest.fixture(autouse=True)
    def _clean_logging(self) -> Iterator[None]:
        """Save and restore root logger state after each test."""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level

        import uvicorn.config

        original_uvicorn_config = uvicorn.config.LOGGING_CONFIG

        yield

        root.handlers = original_handlers
        root.level = original_level
        uvicorn.config.LOGGING_CONFIG = original_uvicorn_config

    def test_root_handler_configured(self) -> None:
        """Root logger gets exactly one handler with JSONFormatter."""
        setup_logging("INFO")

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_log_level_applied(self) -> None:
        """Root logger level matches the requested level."""
        setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

        setup_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_library_loggers_propagate(self) -> None:
        """Library loggers have no handlers and propagate to root."""
        setup_logging("INFO")

        for name in ("uvicorn", "uvicorn.error", "sqlalchemy.engine", "fastmcp"):
            lib_logger = logging.getLogger(name)
            assert lib_logger.propagate is True
            assert lib_logger.handlers == []

    def test_uvicorn_config_overridden(self) -> None:
        """uvicorn LOGGING_CONFIG is replaced with minimal no-op dict."""
        setup_logging("INFO")

        import uvicorn.config

        cfg = uvicorn.config.LOGGING_CONFIG
        assert isinstance(cfg, dict)
        assert cfg.get("version") == 1
        assert cfg.get("disable_existing_loggers") is False

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Logging through setup produces valid JSON to stdout."""
        setup_logging("INFO")

        logging.getLogger("test.output").info("Hello from test")

        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert len(lines) >= 1

        data = json.loads(lines[-1])
        assert data["message"] == "Hello from test"
        assert data["logger"] == "test.output"

    def test_idempotent(self) -> None:
        """Calling setup_logging twice doesn't duplicate handlers."""
        setup_logging("INFO")
        setup_logging("DEBUG")

        root = logging.getLogger()
        assert len(root.handlers) == 1


class TestParseHookLog:
    """Test hook log file parsing."""

    def test_parse_valid_log(self, tmp_path: Path) -> None:
        """Correctly counts pre events and blocks."""
        log_file = tmp_path / "hook.log"
        log_file.write_text(
            '{"hook":"pre","decision":"allow","remaining_s":50}\n'
            '{"hook":"post","decision":"allow","remaining_s":48}\n'
            '{"hook":"pre","decision":"allow","remaining_s":45}\n'
            '{"hook":"pre","decision":"block","remaining_s":0}\n'
            '{"hook":"post","decision":"allow","remaining_s":0}\n'
        )

        result = parse_hook_log(log_file)

        assert result["tool_calls"] == 3
        assert result["hook_blocks"] == 1
        assert result["total_events"] == 5

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns zeros."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("")

        result = parse_hook_log(log_file)

        assert result == {"tool_calls": 0, "hook_blocks": 0, "total_events": 0}

    def test_parse_missing_file(self) -> None:
        """Missing file returns zeros."""
        result = parse_hook_log("/nonexistent/path/hook.log")

        assert result == {"tool_calls": 0, "hook_blocks": 0, "total_events": 0}

    def test_parse_malformed_lines_skipped(self, tmp_path: Path) -> None:
        """Malformed lines are skipped, valid lines counted."""
        log_file = tmp_path / "mixed.log"
        log_file.write_text(
            '{"hook":"pre","decision":"allow"}\n'
            "not json at all\n"
            '{"hook":"post","decision":"allow"}\n'
        )

        result = parse_hook_log(log_file)

        assert result["tool_calls"] == 1
        assert result["total_events"] == 2

    def test_parse_no_pre_events(self, tmp_path: Path) -> None:
        """All post events → tool_calls=0."""
        log_file = tmp_path / "post_only.log"
        log_file.write_text(
            '{"hook":"post","decision":"allow"}\n'
            '{"hook":"post","decision":"allow"}\n'
        )

        result = parse_hook_log(log_file)

        assert result["tool_calls"] == 0
        assert result["total_events"] == 2

    def test_parse_string_path(self, tmp_path: Path) -> None:
        """Accepts string paths in addition to Path objects."""
        log_file = tmp_path / "hook.log"
        log_file.write_text('{"hook":"pre","decision":"allow"}\n')

        result = parse_hook_log(str(log_file))

        assert result["tool_calls"] == 1


class TestLogWorkerCompletion:
    """Test structured worker completion logging."""

    def test_structured_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Log record contains all expected structured fields."""
        with caplog.at_level(logging.INFO, logger="prism.core.logging"):
            log_worker_completion(
                worker_type="researcher",
                agent_key="researcher_1",
                success=True,
                wall_time_s=15.234,
                model="sonnet",
                response_length=5000,
                tool_calls=3,
                hook_blocks=0,
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Worker completed"
        assert record.worker_type == "researcher"  # type: ignore[attr-defined]
        assert record.success is True  # type: ignore[attr-defined]
        assert record.model == "sonnet"  # type: ignore[attr-defined]
        assert record.response_length == 5000  # type: ignore[attr-defined]
        assert record.tool_calls == 3  # type: ignore[attr-defined]
        assert record.hook_blocks == 0  # type: ignore[attr-defined]

    def test_wall_time_rounded(self, caplog: pytest.LogCaptureFixture) -> None:
        """wall_time_s is rounded to 2 decimal places."""
        with caplog.at_level(logging.INFO, logger="prism.core.logging"):
            log_worker_completion(
                worker_type="tavily",
                agent_key="tavily_1",
                success=True,
                wall_time_s=12.34567,
                model="sonnet",
                response_length=100,
            )

        record = caplog.records[0]
        assert record.wall_time_s == 12.35  # type: ignore[attr-defined]

    def test_default_hook_metrics(self, caplog: pytest.LogCaptureFixture) -> None:
        """tool_calls and hook_blocks default to 0 when not provided."""
        with caplog.at_level(logging.INFO, logger="prism.core.logging"):
            log_worker_completion(
                worker_type="perplexity",
                agent_key="perplexity_1",
                success=False,
                wall_time_s=2.0,
                model="sonnet",
                response_length=0,
            )

        record = caplog.records[0]
        assert record.tool_calls == 0  # type: ignore[attr-defined]
        assert record.hook_blocks == 0  # type: ignore[attr-defined]
