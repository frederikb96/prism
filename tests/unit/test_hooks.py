"""Unit tests for time awareness hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from prism.core.hooks import (
    HOOK_SCRIPT_PATH,
    build_time_awareness_hooks,
    build_time_env_vars,
)

HOOK_SCRIPT = Path(__file__).parent.parent.parent / "hooks" / "time_tracker.py"


class TestBuildTimeAwarenessHooks:
    """Tests for build_time_awareness_hooks function."""

    def test_returns_valid_structure(self) -> None:
        """Hook config has correct structure."""
        config = build_time_awareness_hooks()

        assert "hooks" in config
        assert "PreToolUse" in config["hooks"]
        assert "PostToolUse" in config["hooks"]

    def test_pre_tool_use_config(self) -> None:
        """PreToolUse hook is correctly configured."""
        config = build_time_awareness_hooks()
        pre_hooks = config["hooks"]["PreToolUse"]

        assert len(pre_hooks) == 1
        assert pre_hooks[0]["matcher"] == ".*"
        assert pre_hooks[0]["hooks"][0]["type"] == "command"
        assert "pre" in pre_hooks[0]["hooks"][0]["command"]

    def test_post_tool_use_config(self) -> None:
        """PostToolUse hook is correctly configured."""
        config = build_time_awareness_hooks()
        post_hooks = config["hooks"]["PostToolUse"]

        assert len(post_hooks) == 1
        assert post_hooks[0]["matcher"] == ".*"
        assert post_hooks[0]["hooks"][0]["type"] == "command"
        assert "post" in post_hooks[0]["hooks"][0]["command"]

    def test_uses_correct_script_path(self) -> None:
        """Hooks reference the correct script path."""
        config = build_time_awareness_hooks()

        pre_cmd = config["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        post_cmd = config["hooks"]["PostToolUse"][0]["hooks"][0]["command"]

        assert HOOK_SCRIPT_PATH in pre_cmd
        assert HOOK_SCRIPT_PATH in post_cmd


class TestBuildTimeEnvVars:
    """Tests for build_time_env_vars function."""

    def test_returns_tuple_of_tuples(self) -> None:
        """Returns correct format for env vars."""
        result = build_time_env_vars(1000.0, 60)

        assert isinstance(result, tuple)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_contains_required_vars(self) -> None:
        """Contains both required environment variables."""
        result = build_time_env_vars(1234.5, 90)
        result_dict = dict(result)

        assert "PRISM_START_TIME" in result_dict
        assert "PRISM_VISIBLE_TIMEOUT" in result_dict

    def test_start_time_value(self) -> None:
        """Start time is correctly formatted."""
        result = build_time_env_vars(1706900000.123, 60)
        result_dict = dict(result)

        assert result_dict["PRISM_START_TIME"] == "1706900000.123"

    def test_visible_timeout_value(self) -> None:
        """Visible timeout is correctly formatted."""
        result = build_time_env_vars(1000.0, 120)
        result_dict = dict(result)

        assert result_dict["PRISM_VISIBLE_TIMEOUT"] == "120"


class TestTimeTrackerScript:
    """Tests for the time_tracker.py hook script."""

    @pytest.fixture
    def run_hook(self):
        """Fixture to run the hook script with given env vars."""

        def _run(
            hook_type: str = "post",
            start_time: str | None = None,
            visible_timeout: str | None = None,
        ) -> dict:
            env = {}
            if start_time is not None:
                env["PRISM_START_TIME"] = start_time
            if visible_timeout is not None:
                env["PRISM_VISIBLE_TIMEOUT"] = visible_timeout

            result = subprocess.run(
                [sys.executable, str(HOOK_SCRIPT), hook_type],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )

            if result.stdout.strip():
                return json.loads(result.stdout)
            return {}

        return _run

    def test_missing_env_vars_returns_empty(self, run_hook) -> None:
        """Script exits cleanly when env vars are missing."""
        result = run_hook()
        assert result == {}

    def test_normal_time_message(self, run_hook) -> None:
        """Normal message when plenty of time remaining."""
        import time

        start = time.time() - 30
        result = run_hook(
            start_time=str(start),
            visible_timeout="60",
        )

        assert result["continue"] is True
        assert "⏱️" in result["hookSpecificOutput"]["additionalContext"]
        assert "30s" in result["hookSpecificOutput"]["additionalContext"]

    def test_warning_time_message(self, run_hook) -> None:
        """Warning message when 5-15 seconds remaining."""
        import time

        start = time.time() - 50
        result = run_hook(
            start_time=str(start),
            visible_timeout="60",
        )

        assert result["continue"] is True
        context = result["hookSpecificOutput"]["additionalContext"]
        assert "⚠️" in context
        assert "WRAP UP" in context

    def test_critical_time_message(self, run_hook) -> None:
        """Critical message when <5 seconds remaining."""
        import time

        start = time.time() - 57
        result = run_hook(
            start_time=str(start),
            visible_timeout="60",
        )

        assert result["continue"] is True
        context = result["hookSpecificOutput"]["additionalContext"]
        assert "🚨" in context
        assert "CRITICAL" in context

    def test_no_negative_remaining(self, run_hook) -> None:
        """Shows 0 seconds, not negative when over time."""
        import time

        start = time.time() - 100
        result = run_hook(
            start_time=str(start),
            visible_timeout="60",
        )

        context = result["hookSpecificOutput"]["additionalContext"]
        assert "0s left" in context or "Only 0s" in context
        assert "-" not in context.split("Only")[1] if "Only" in context else True

    def test_pre_hook_event_name(self, run_hook) -> None:
        """Pre hook sets correct event name."""
        import time

        start = time.time() - 10
        result = run_hook(
            hook_type="pre",
            start_time=str(start),
            visible_timeout="60",
        )

        assert result["hookSpecificOutput"]["hookEventName"] == "PreToolUse"

    def test_post_hook_event_name(self, run_hook) -> None:
        """Post hook sets correct event name."""
        import time

        start = time.time() - 10
        result = run_hook(
            hook_type="post",
            start_time=str(start),
            visible_timeout="60",
        )

        assert result["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

    def test_valid_json_output(self, run_hook) -> None:
        """Script outputs valid JSON."""
        import time

        start = time.time() - 10
        result = run_hook(
            start_time=str(start),
            visible_timeout="60",
        )

        assert "continue" in result
        assert "hookSpecificOutput" in result
        assert "hookEventName" in result["hookSpecificOutput"]
        assert "additionalContext" in result["hookSpecificOutput"]


class TestGetTimeMessage:
    """Tests for the get_time_message function in the hook script."""

    @pytest.fixture
    def get_time_message(self):
        """Import the function from the hook script."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("time_tracker", HOOK_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.get_time_message

    def test_normal_range(self, get_time_message) -> None:
        """Normal message for >15s remaining."""
        msg = get_time_message(30, 60)
        assert "⏱️" in msg
        assert "30s" in msg

    def test_warning_range_15s(self, get_time_message) -> None:
        """Warning at exactly 15s remaining."""
        msg = get_time_message(45, 60)
        assert "⚠️" in msg

    def test_warning_range_6s(self, get_time_message) -> None:
        """Warning at 6s remaining."""
        msg = get_time_message(54, 60)
        assert "⚠️" in msg

    def test_critical_range_5s(self, get_time_message) -> None:
        """Critical at exactly 5s remaining."""
        msg = get_time_message(55, 60)
        assert "🚨" in msg

    def test_critical_range_0s(self, get_time_message) -> None:
        """Critical at 0s remaining."""
        msg = get_time_message(60, 60)
        assert "🚨" in msg
        assert "0s" in msg

    def test_clamps_negative_to_zero(self, get_time_message) -> None:
        """Remaining time is clamped to 0, not negative."""
        msg = get_time_message(100, 60)
        assert "0s" in msg
        assert "-" not in msg
