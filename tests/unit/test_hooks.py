"""Unit tests for hooks configuration builders and time hook script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prism.core.hooks import (
    HOOK_SCRIPT_PATH,
    build_claude_hooks,
    build_gemini_settings_file,
    build_time_env_vars,
)

HOOK_SCRIPT = Path(__file__).parent.parent.parent / "hooks" / "time_hook.py"


class TestBuildClaudeHooks:
    """Tests for build_claude_hooks function."""

    def test_returns_valid_structure(self) -> None:
        """Hook config has correct structure."""
        config = build_claude_hooks()

        assert "hooks" in config
        assert "PreToolUse" in config["hooks"]
        assert "PostToolUse" in config["hooks"]

    def test_pre_tool_use_config(self) -> None:
        """PreToolUse hook is correctly configured."""
        config = build_claude_hooks()
        pre_hooks = config["hooks"]["PreToolUse"]

        assert len(pre_hooks) == 1
        assert pre_hooks[0]["matcher"] == ".*"
        assert pre_hooks[0]["hooks"][0]["type"] == "command"
        assert "pre" in pre_hooks[0]["hooks"][0]["command"]

    def test_post_tool_use_config(self) -> None:
        """PostToolUse hook is correctly configured."""
        config = build_claude_hooks()
        post_hooks = config["hooks"]["PostToolUse"]

        assert len(post_hooks) == 1
        assert post_hooks[0]["matcher"] == ".*"
        assert post_hooks[0]["hooks"][0]["type"] == "command"
        assert "post" in post_hooks[0]["hooks"][0]["command"]

    def test_uses_correct_script_path(self) -> None:
        """Hooks reference the correct script path."""
        config = build_claude_hooks()

        pre_cmd = config["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        post_cmd = config["hooks"]["PostToolUse"][0]["hooks"][0]["command"]

        assert HOOK_SCRIPT_PATH in pre_cmd
        assert HOOK_SCRIPT_PATH in post_cmd


class TestBuildGeminiSettingsFile:
    """Tests for build_gemini_settings_file function."""

    def test_creates_temp_file(self) -> None:
        """Creates a temp file and returns its path."""
        path = build_gemini_settings_file()
        try:
            assert os.path.exists(path)
            assert path.startswith("/tmp/prism-gemini-settings-")
            assert path.endswith(".json")
        finally:
            os.unlink(path)

    def test_valid_json_content(self) -> None:
        """File contains valid JSON."""
        path = build_gemini_settings_file()
        try:
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
        finally:
            os.unlink(path)

    def test_has_gemini_hook_structure(self) -> None:
        """Settings file has BeforeTool/AfterTool hooks."""
        path = build_gemini_settings_file()
        try:
            with open(path) as f:
                data = json.load(f)

            assert data["hooksConfig"]["enabled"] is True
            assert "BeforeTool" in data["hooks"]
            assert "AfterTool" in data["hooks"]

            before = data["hooks"]["BeforeTool"][0]
            assert before["matcher"] == "*"
            assert "pre" in before["hooks"][0]["command"]
            assert before["hooks"][0]["name"] == "prism-time-before"

            after = data["hooks"]["AfterTool"][0]
            assert after["matcher"] == "*"
            assert "post" in after["hooks"][0]["command"]
            assert after["hooks"][0]["name"] == "prism-time-after"
        finally:
            os.unlink(path)


class TestBuildTimeEnvVars:
    """Tests for build_time_env_vars function."""

    def test_returns_tuple_of_tuples(self) -> None:
        """Returns correct format for env vars."""
        result = build_time_env_vars(1000.0, 60, "claude", "/tmp/test.log")

        assert isinstance(result, tuple)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_contains_all_required_vars(self) -> None:
        """Contains all four required environment variables."""
        result = build_time_env_vars(1234.5, 90, "gemini", "/tmp/hook.log")
        result_dict = dict(result)

        assert "PRISM_START_TIME" in result_dict
        assert "PRISM_TOOL_TIMEOUT" in result_dict
        assert "PRISM_HOOK_FORMAT" in result_dict
        assert "PRISM_HOOK_LOG" in result_dict

    def test_values(self) -> None:
        """Values are correctly formatted."""
        result = build_time_env_vars(1706900000.123, 120, "claude", "/tmp/h.log")
        result_dict = dict(result)

        assert result_dict["PRISM_START_TIME"] == "1706900000.123"
        assert result_dict["PRISM_TOOL_TIMEOUT"] == "120"
        assert result_dict["PRISM_HOOK_FORMAT"] == "claude"
        assert result_dict["PRISM_HOOK_LOG"] == "/tmp/h.log"


class TestTimeHookScript:
    """Tests for the time_hook.py hook script."""

    @pytest.fixture
    def run_hook(self, tmp_path):
        """Fixture to run the hook script with given env vars."""
        log_file = str(tmp_path / "hook.log")

        def _run(
            hook_type: str = "post",
            start_time: str | None = None,
            tool_timeout: str | None = None,
            hook_format: str = "claude",
        ) -> dict:
            env = {"PRISM_HOOK_FORMAT": hook_format, "PRISM_HOOK_LOG": log_file}
            if start_time is not None:
                env["PRISM_START_TIME"] = start_time
            if tool_timeout is not None:
                env["PRISM_TOOL_TIMEOUT"] = tool_timeout

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

    def test_missing_env_vars_claude(self, run_hook) -> None:
        """Script exits cleanly when timing env vars are missing (Claude format)."""
        result = run_hook()
        assert result == {}

    def test_missing_env_vars_gemini(self, run_hook) -> None:
        """Script returns allow decision when timing vars missing (Gemini format)."""
        result = run_hook(hook_format="gemini")
        assert result == {"decision": "allow"}

    def test_claude_pre_allow(self, run_hook) -> None:
        """Claude pre hook allows when time remaining."""
        import time

        start = time.time() - 10
        result = run_hook(
            hook_type="pre",
            start_time=str(start),
            tool_timeout="60",
        )

        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        assert "remaining" in result["hookSpecificOutput"]["additionalContext"]

    def test_claude_post_allow(self, run_hook) -> None:
        """Claude post hook allows with time context."""
        import time

        start = time.time() - 10
        result = run_hook(
            hook_type="post",
            start_time=str(start),
            tool_timeout="60",
        )

        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

    def test_claude_pre_block_expired(self, run_hook) -> None:
        """Claude pre hook blocks when time expired."""
        import time

        start = time.time() - 100
        result = run_hook(
            hook_type="pre",
            start_time=str(start),
            tool_timeout="60",
        )

        assert "hookSpecificOutput" in result
        output = result["hookSpecificOutput"]
        assert output["permissionDecision"] == "deny"
        assert "TIME EXPIRED" in output["permissionDecisionReason"]

    def test_claude_low_time_warning(self, run_hook) -> None:
        """Claude hook shows low time warning when <=10s remaining."""
        import time

        start = time.time() - 55
        result = run_hook(
            hook_type="post",
            start_time=str(start),
            tool_timeout="60",
        )

        context = result["hookSpecificOutput"]["additionalContext"]
        assert "LOW TIME" in context

    def test_gemini_pre_allow(self, run_hook) -> None:
        """Gemini pre hook allows when time remaining."""
        import time

        start = time.time() - 10
        result = run_hook(
            hook_type="pre",
            start_time=str(start),
            tool_timeout="60",
            hook_format="gemini",
        )

        assert result["decision"] == "allow"
        assert "remaining" in result["hookSpecificOutput"]["additionalContext"]

    def test_gemini_pre_block_expired(self, run_hook) -> None:
        """Gemini pre hook blocks when time expired."""
        import time

        start = time.time() - 100
        result = run_hook(
            hook_type="pre",
            start_time=str(start),
            tool_timeout="60",
            hook_format="gemini",
        )

        assert result["decision"] == "block"
        assert "TIME EXPIRED" in result["reason"]

    def test_gemini_post_allow(self, run_hook) -> None:
        """Gemini post hook always allows with time context."""
        import time

        start = time.time() - 100
        result = run_hook(
            hook_type="post",
            start_time=str(start),
            tool_timeout="60",
            hook_format="gemini",
        )

        assert result["decision"] == "allow"

    def test_hook_log_written(self, run_hook, tmp_path) -> None:
        """Hook writes to log file."""
        import time

        log_file = tmp_path / "hook.log"
        start = time.time() - 10
        run_hook(
            hook_type="pre",
            start_time=str(start),
            tool_timeout="60",
        )

        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["hook"] == "pre"
        assert entry["decision"] == "allow"
