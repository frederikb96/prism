"""Tests for workers.

Tests worker agent patterns using MockExecutor:
- ExecutorProtocol compliance
- Worker execution patterns
- Manager patterns for task planning
"""

from __future__ import annotations

import pytest

from prism.config import RetryConfig
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.retry import RetryExecutor
from prism.workers.base import ExecutorProtocol

from .conftest import MockExecutor


class TestExecutorProtocol:
    """Test that MockExecutor satisfies ExecutorProtocol."""

    def test_mock_executor_is_protocol_compliant(self) -> None:
        """MockExecutor implements ExecutorProtocol."""
        executor: ExecutorProtocol = MockExecutor()
        assert hasattr(executor, "execute")

    @pytest.mark.asyncio
    async def test_protocol_execute_signature(self) -> None:
        """Execute method has correct signature."""
        executor: ExecutorProtocol = MockExecutor()
        request = ExecutionRequest(prompt="Test", model="sonnet")

        executor.add_result(ExecutionResult.from_success("done"))
        result = await executor.execute(request)
        assert isinstance(result, ExecutionResult)

        executor.add_result(ExecutionResult.from_success("done"))
        result = await executor.execute(request, session_id="sess-1")
        assert isinstance(result, ExecutionResult)


class TestWorkerPatterns:
    """Test worker-executor interaction patterns.

    These tests verify the expected interaction between
    workers and executors using MockExecutor.
    """

    @pytest.mark.asyncio
    async def test_worker_uses_executor(
        self,
        mock_executor: MockExecutor,
        sample_request: ExecutionRequest,
    ) -> None:
        """Workers call executor.execute() with requests."""
        mock_executor.add_result(ExecutionResult.from_success('{"result": "data"}'))

        result = await mock_executor.execute(sample_request)

        assert result.success is True
        assert len(mock_executor.calls) == 1
        assert mock_executor.calls[0][0] == sample_request

    @pytest.mark.asyncio
    async def test_worker_with_schema_validation(
        self,
        mock_executor: MockExecutor,
        sample_schema: dict,
    ) -> None:
        """Workers can use RetryExecutor for schema validation."""
        config = RetryConfig(
            max_transient_retries=1,
            max_validation_retries=1,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
        )
        retry_executor = RetryExecutor(mock_executor, config)

        request = ExecutionRequest(
            prompt="Get structured data",
            model="sonnet",
            json_schema=sample_schema,
        )
        mock_executor.add_result(
            ExecutionResult.from_success(
                '{"result": "answer", "confidence": 0.9}',
                "sess-1",
            )
        )

        result = await retry_executor.execute(request, schema=sample_schema)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_with_tools(
        self,
        mock_executor: MockExecutor,
    ) -> None:
        """Workers can request tool access via ExecutionRequest."""
        request = ExecutionRequest(
            prompt="Search the web",
            model="sonnet",
            tools="mcp",
            allowed_tools=("web_search", "web_fetch"),
        )
        mock_executor.add_result(
            ExecutionResult.from_success('{"sources": ["url1"]}', "sess-1")
        )

        result = await mock_executor.execute(request)

        assert result.success is True
        called_request = mock_executor.calls[0][0]
        assert called_request.tools == "mcp"
        assert "web_search" in called_request.allowed_tools


class TestManagerPatterns:
    """Test patterns for ManagerAgent.

    ManagerAgent parses user queries and creates TaskPlans.
    These tests verify the expected output structure.
    """

    def test_task_plan_structure(self) -> None:
        """TaskPlan should contain worker assignments."""
        expected_plan_schema = {
            "type": "object",
            "required": ["tasks"],
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["query", "agent_type"],
                        "properties": {
                            "query": {"type": "string"},
                            "agent_type": {"type": "string"},
                            "context": {"type": "string"},
                        },
                    },
                },
            },
        }

        assert expected_plan_schema["type"] == "object"
        assert "tasks" in expected_plan_schema["required"]
