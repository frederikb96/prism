"""Tests for GeminiExecutor."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from prism.config import RetryConfig
from prism.core.gemini import GeminiExecutor
from prism.core.process import ProcessResult
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.session import SessionRegistry


class TestBuildCommand:
    """Test Gemini command building."""

    def test_basic_command(self) -> None:
        """Command includes all required flags."""
        executor = GeminiExecutor()
        request = ExecutionRequest(prompt="Search for X", model="gemini-2.5-flash")

        cmd = executor.build_command(request)

        assert cmd == [
            "gemini",
            "-p", "Search for X",
            "--model", "gemini-2.5-flash",
            "--allowed-tools", "google_web_search",
            "-o", "json",
            "--yolo",
        ]


class TestBuildEnv:
    """Test environment and temp file setup."""

    def test_system_prompt_creates_temp_file(self) -> None:
        """System prompt written to temp file with GEMINI_SYSTEM_MD env."""
        executor = GeminiExecutor()
        request = ExecutionRequest(
            prompt="test",
            model="gemini-2.5-flash",
            system_prompt="You are a search agent.",
        )

        env, temp_files = executor._build_env(request)

        try:
            assert "GEMINI_SYSTEM_MD" in env
            sys_md_path = env["GEMINI_SYSTEM_MD"]
            assert os.path.exists(sys_md_path)
            with open(sys_md_path) as f:
                assert f.read() == "You are a search agent."
            assert sys_md_path in temp_files
        finally:
            GeminiExecutor._cleanup_temp_files(temp_files)

    def test_gemini_settings_file_created(self) -> None:
        """Gemini settings file always created."""
        executor = GeminiExecutor()
        request = ExecutionRequest(prompt="test", model="gemini-2.5-flash")

        env, temp_files = executor._build_env(request)

        try:
            assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH" in env
            settings_path = env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"]
            assert os.path.exists(settings_path)
            with open(settings_path) as f:
                data = json.load(f)
            assert "hooks" in data
            assert "BeforeTool" in data["hooks"]
        finally:
            GeminiExecutor._cleanup_temp_files(temp_files)

    def test_google_api_key_mapping(self) -> None:
        """GOOGLE_API_KEY mapped to GEMINI_API_KEY when needed."""
        executor = GeminiExecutor()
        request = ExecutionRequest(prompt="test", model="gemini-2.5-flash")

        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test-key"},
            clear=False,
        ):
            # Remove GEMINI_API_KEY if present
            os.environ.pop("GEMINI_API_KEY", None)
            env, temp_files = executor._build_env(request)

        try:
            assert env.get("GEMINI_API_KEY") == "test-key"
        finally:
            GeminiExecutor._cleanup_temp_files(temp_files)

    def test_env_vars_from_request(self) -> None:
        """Request env_vars are included in environment."""
        executor = GeminiExecutor()
        request = ExecutionRequest(
            prompt="test",
            model="gemini-2.5-flash",
            env_vars=(("PRISM_START_TIME", "1234.0"),),
        )

        env, temp_files = executor._build_env(request)

        try:
            assert env["PRISM_START_TIME"] == "1234.0"
        finally:
            GeminiExecutor._cleanup_temp_files(temp_files)


class TestExecute:
    """Test execute method with mocked subprocess."""

    @pytest.fixture
    def fast_retry(self) -> RetryConfig:
        return RetryConfig(
            max_transient_retries=2,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
            max_validation_retries=0,
        )

    @pytest.mark.asyncio
    async def test_successful_execution(self, fast_retry) -> None:
        """Successful execution returns output."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=10,
        )

        output = json.dumps({"response": "result text", "session_id": "gem-123"})
        mock_result = ProcessResult(stdout=output, stderr="", returncode=0)

        with (
            patch.object(executor, "_build_env", return_value=({}, [])),
            patch("prism.core.gemini.CancellableProcess") as MockProcess,
        ):
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = False
            MockProcess.return_value = mock_process

            result = await executor.execute(request)

        assert result.success is True
        assert result.output == output

    @pytest.mark.asyncio
    async def test_execution_failure(self, fast_retry) -> None:
        """Failed execution returns error result."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=10,
        )

        mock_result = ProcessResult(
            stdout="", stderr="API error", returncode=1,
        )

        with (
            patch.object(executor, "_build_env", return_value=({}, [])),
            patch("prism.core.gemini.CancellableProcess") as MockProcess,
        ):
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = False
            MockProcess.return_value = mock_process

            result = await executor.execute(request)

        assert result.success is False
        assert "API error" in result.error_message

    @pytest.mark.asyncio
    async def test_timeout_not_retried(self, fast_retry) -> None:
        """Timeout returns immediately without retry."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=1,
        )

        call_count = 0

        async def mock_execute_once(req, sid):
            nonlocal call_count
            call_count += 1
            return ExecutionResult.from_timeout(1)

        with patch.object(executor, "_execute_once", side_effect=mock_execute_once):
            result = await executor.execute(request)

        assert result.success is False
        assert result.is_timeout is True
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_transient_error_retried(self, fast_retry) -> None:
        """Transient errors (503) trigger retry."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=10,
        )

        call_count = 0

        async def mock_execute_once(req, sid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ExecutionResult.from_error("503 unavailable")
            return ExecutionResult.from_success(output="ok", session_id=sid)

        with patch.object(executor, "_execute_once", side_effect=mock_execute_once):
            result = await executor.execute(request)

        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_transient_error_not_retried(self, fast_retry) -> None:
        """Non-transient errors are returned immediately."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=10,
        )

        call_count = 0

        async def mock_execute_once(req, sid):
            nonlocal call_count
            call_count += 1
            return ExecutionResult.from_error("Invalid API key")

        with patch.object(executor, "_execute_once", side_effect=mock_execute_once):
            result = await executor.execute(request)

        assert result.success is False
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cancelled_not_retried(self, fast_retry) -> None:
        """Cancelled execution is returned immediately."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=10,
        )

        async def mock_execute_once(req, sid):
            return ExecutionResult.from_cancelled()

        with patch.object(executor, "_execute_once", side_effect=mock_execute_once):
            result = await executor.execute(request)

        assert result.is_cancelled is True

    @pytest.mark.asyncio
    async def test_session_registry(self, fast_retry) -> None:
        """Session registered and unregistered during execution."""
        registry = SessionRegistry()
        executor = GeminiExecutor(
            session_registry=registry, retry_config=fast_retry,
        )
        request = ExecutionRequest(
            prompt="Test", model="gemini-2.5-flash", timeout_seconds=10,
        )

        mock_result = ProcessResult(stdout="{}", stderr="", returncode=0)

        with (
            patch.object(executor, "_build_env", return_value=({}, [])),
            patch("prism.core.gemini.CancellableProcess") as MockProcess,
        ):
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = False
            MockProcess.return_value = mock_process

            await executor.execute(request, session_id="test-session")

        session = await registry.get("test-session")
        assert session is None

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up(self, fast_retry) -> None:
        """Temp files cleaned up after execution."""
        executor = GeminiExecutor(retry_config=fast_retry)
        request = ExecutionRequest(
            prompt="Test",
            model="gemini-2.5-flash",
            timeout_seconds=10,
            system_prompt="Be helpful",
        )

        mock_result = ProcessResult(stdout="{}", stderr="", returncode=0)
        created_files: list[str] = []

        original_build_env = executor._build_env

        def tracking_build_env(req):
            env, temp_files = original_build_env(req)
            created_files.extend(temp_files)
            return env, temp_files

        with (
            patch.object(executor, "_build_env", side_effect=tracking_build_env),
            patch("prism.core.gemini.CancellableProcess") as MockProcess,
        ):
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = False
            MockProcess.return_value = mock_process

            await executor.execute(request)

        assert len(created_files) >= 2
        for f in created_files:
            assert not os.path.exists(f), f"Temp file not cleaned up: {f}"


class TestParseGeminiOutput:
    """Test Gemini output parsing."""

    def test_parse_full_output(self) -> None:
        """Parse response with stats."""
        raw = json.dumps({
            "session_id": "s1",
            "response": "Search results here",
            "stats": {
                "tools": {"byName": {"google_web_search": {"count": 3}}},
                "models": {
                    "gemini-2.5-flash": {
                        "tokens": {"input": 100, "output": 50}
                    }
                },
            },
        })

        result = GeminiExecutor.parse_gemini_output(raw)

        assert result["response"] == "Search results here"
        assert result["tool_usage"] == {"google_web_search": {"count": 3}}
        assert "gemini-2.5-flash" in result["token_counts"]

    def test_parse_minimal_output(self) -> None:
        """Parse response without stats."""
        raw = json.dumps({"response": "answer"})

        result = GeminiExecutor.parse_gemini_output(raw)

        assert result["response"] == "answer"
        assert result["tool_usage"] == {}
        assert result["token_counts"] == {}

    def test_parse_invalid_json(self) -> None:
        """Handle non-JSON output gracefully."""
        result = GeminiExecutor.parse_gemini_output("not json")

        assert result["response"] == "not json"
        assert result["tool_usage"] == {}


class TestIsTransientError:
    """Test transient error detection."""

    @pytest.mark.parametrize(
        "error",
        [
            "connection refused",
            "request timeout",
            "temporary failure",
            "service unavailable",
            "rate limit exceeded",
            "HTTP 429",
            "HTTP 503",
            "HTTP 502",
        ],
    )
    def test_transient_errors(self, error: str) -> None:
        """Known transient patterns are detected."""
        result = ExecutionResult.from_error(error)
        assert result.is_transient_error() is True

    def test_non_transient_error(self) -> None:
        """Non-transient errors are not flagged."""
        result = ExecutionResult.from_error("Invalid API key")
        assert result.is_transient_error() is False

    def test_timeout_is_transient(self) -> None:
        """Timeout results are transient."""
        result = ExecutionResult.from_timeout(30)
        assert result.is_transient_error() is True
