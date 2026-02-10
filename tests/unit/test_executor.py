"""Tests for ClaudeExecutor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from prism.core.executor import ClaudeExecutor
from prism.core.process import ProcessResult
from prism.core.response import ExecutionRequest
from prism.core.session import SessionRegistry


class TestBuildCommand:
    """Test command building logic."""

    def test_minimal_command(self) -> None:
        """Basic command with just prompt and model."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(prompt="Hello", model="sonnet")

        cmd = executor.build_command(request)

        assert cmd == [
            "claude",
            "-p", "Hello",
            "--model", "sonnet",
            "--output-format", "json",
        ]

    def test_with_tools(self) -> None:
        """Command with tools enabled."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(
            prompt="Search for X",
            model="sonnet",
            tools="mcp",
        )

        cmd = executor.build_command(request)

        assert "--tools" in cmd
        assert cmd[cmd.index("--tools") + 1] == "mcp"

    def test_with_allowed_tools(self) -> None:
        """Command with specific allowed tools."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(
            prompt="Search",
            model="sonnet",
            tools="mcp",
            allowed_tools=("web_search", "web_fetch"),
        )

        cmd = executor.build_command(request)

        # Should have --allowedTools for each tool
        allowed_indices = [i for i, x in enumerate(cmd) if x == "--allowedTools"]
        assert len(allowed_indices) == 2
        assert cmd[allowed_indices[0] + 1] == "web_search"
        assert cmd[allowed_indices[1] + 1] == "web_fetch"

    def test_with_json_schema(self) -> None:
        """Command with output schema."""
        executor = ClaudeExecutor()
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        request = ExecutionRequest(
            prompt="Return JSON",
            model="sonnet",
            json_schema=schema,
        )

        cmd = executor.build_command(request)

        assert "--json-schema" in cmd
        schema_str = cmd[cmd.index("--json-schema") + 1]
        assert json.loads(schema_str) == schema

    def test_with_system_prompt(self) -> None:
        """Command with system prompt."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(
            prompt="Do task",
            model="sonnet",
            system_prompt="You are a helpful assistant",
        )

        cmd = executor.build_command(request)

        assert "--system-prompt" in cmd
        assert cmd[cmd.index("--system-prompt") + 1] == "You are a helpful assistant"

    def test_with_resume_session(self) -> None:
        """Command with session resume."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(
            prompt="Continue",
            model="sonnet",
            resume_session="session-123",
        )

        cmd = executor.build_command(request)

        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == "session-123"

    def test_full_command(self) -> None:
        """Command with all options."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(
            prompt="Complex task",
            model="opus",
            tools="mcp",
            allowed_tools=("tool1",),
            json_schema={"type": "string"},
            system_prompt="Be concise",
            resume_session="sess-456",
        )

        cmd = executor.build_command(request)

        # Verify all parts present
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "--tools" in cmd
        assert "--allowedTools" in cmd
        assert "--json-schema" in cmd
        assert "--system-prompt" in cmd
        assert "--resume" in cmd


class TestExecute:
    """Test execute method with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        """Successful execution returns output and session."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(prompt="Test", model="sonnet", timeout_seconds=10)

        mock_result = ProcessResult(
            stdout='{"result": "done", "session_id": "abc-123"}',
            stderr="",
            returncode=0,
        )

        with patch.object(executor, "build_command", return_value=["claude", "-p", "Test"]):
            with patch("prism.core.executor.CancellableProcess") as MockProcess:
                mock_process = AsyncMock()
                mock_process.run.return_value = mock_result
                mock_process.is_cancelled = False
                MockProcess.return_value = mock_process

                result = await executor.execute(request)

        assert result.success is True
        assert result.output == '{"result": "done", "session_id": "abc-123"}'
        assert result.session_id == "abc-123"

    @pytest.mark.asyncio
    async def test_execution_with_registry(self) -> None:
        """Execution registers and unregisters session."""
        registry = SessionRegistry()
        executor = ClaudeExecutor(session_registry=registry)
        request = ExecutionRequest(prompt="Test", model="sonnet", timeout_seconds=10)

        mock_result = ProcessResult(stdout="{}", stderr="", returncode=0)

        with patch("prism.core.executor.CancellableProcess") as MockProcess:
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = False
            MockProcess.return_value = mock_process

            await executor.execute(request, session_id="test-session")

        # Session should be unregistered after execution
        session = await registry.get("test-session")
        assert session is None

    @pytest.mark.asyncio
    async def test_execution_failure(self) -> None:
        """Failed execution returns error result."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(prompt="Test", model="sonnet", timeout_seconds=10)

        mock_result = ProcessResult(
            stdout="",
            stderr="Error: API rate limit",
            returncode=1,
        )

        with patch("prism.core.executor.CancellableProcess") as MockProcess:
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = False
            MockProcess.return_value = mock_process

            result = await executor.execute(request)

        assert result.success is False
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execution_timeout(self) -> None:
        """Timeout returns timeout result."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(prompt="Test", model="sonnet", timeout_seconds=1)

        with patch("prism.core.executor.CancellableProcess") as MockProcess:
            mock_process = AsyncMock()
            mock_process.run.side_effect = TimeoutError()
            MockProcess.return_value = mock_process

            result = await executor.execute(request)

        assert result.success is False
        assert result.is_timeout is True

    @pytest.mark.asyncio
    async def test_execution_cancelled(self) -> None:
        """Cancelled execution returns cancelled result."""
        executor = ClaudeExecutor()
        request = ExecutionRequest(prompt="Test", model="sonnet", timeout_seconds=10)

        mock_result = ProcessResult(stdout="", stderr="", returncode=-15)

        with patch("prism.core.executor.CancellableProcess") as MockProcess:
            mock_process = AsyncMock()
            mock_process.run.return_value = mock_result
            mock_process.is_cancelled = True
            MockProcess.return_value = mock_process

            result = await executor.execute(request)

        assert result.success is False
        assert result.is_cancelled is True


class TestSessionIdExtraction:
    """Test session ID extraction from output."""

    def test_extract_from_json(self) -> None:
        """Extract session_id from valid JSON."""
        executor = ClaudeExecutor()
        output = '{"result": "x", "session_id": "sess-abc-123"}'

        session_id = executor._extract_session_id(output)

        assert session_id == "sess-abc-123"

    def test_extract_from_invalid_json_regex(self) -> None:
        """Fall back to regex for partial JSON."""
        executor = ClaudeExecutor()
        output = 'partial {"session_id": "sess-xyz"} more stuff'

        session_id = executor._extract_session_id(output)

        assert session_id == "sess-xyz"

    def test_no_session_id(self) -> None:
        """Return None when no session_id present."""
        executor = ClaudeExecutor()
        output = '{"result": "done"}'

        session_id = executor._extract_session_id(output)

        assert session_id is None

    def test_invalid_output(self) -> None:
        """Handle completely invalid output."""
        executor = ClaudeExecutor()
        output = "not json at all"

        session_id = executor._extract_session_id(output)

        assert session_id is None
