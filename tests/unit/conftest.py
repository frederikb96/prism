"""Shared fixtures for unit tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prism.config import RetryConfig, reset_config
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.session import SessionRegistry
from prism.workers.base import ExecutorProtocol


class MockExecutor:
    """Mock executor that returns predefined results."""

    def __init__(self) -> None:
        self.calls: list[tuple[ExecutionRequest, str | None]] = []
        self.results: list[ExecutionResult] = []
        self._call_index = 0

    def add_result(self, result: ExecutionResult) -> None:
        """Queue a result to be returned on next execute call."""
        self.results.append(result)

    async def execute(
        self,
        request: ExecutionRequest,
        schema: dict | None = None,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """Record call and return next queued result."""
        self.calls.append((request, session_id))
        if self._call_index < len(self.results):
            result = self.results[self._call_index]
            self._call_index += 1
            return result
        return ExecutionResult.from_error("No result queued")


_: ExecutorProtocol = MockExecutor()


@pytest.fixture(autouse=True)
def reset_config_fixture() -> None:
    """Reset config before each test to ensure clean state."""
    reset_config()
    os.environ["PRISM_CONFIG_PATH"] = str(
        Path(__file__).parent.parent.parent / "config" / "config.yaml"
    )


@pytest.fixture
def mock_executor() -> MockExecutor:
    """Create a fresh mock executor."""
    return MockExecutor()


@pytest.fixture
def session_registry() -> SessionRegistry:
    """Create a fresh session registry."""
    return SessionRegistry()


@pytest.fixture
def retry_config() -> RetryConfig:
    """Create retry config with minimal delays for fast tests."""
    return RetryConfig(
        max_transient_retries=2,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        exponential_base=2.0,
        max_validation_retries=2,
    )


@pytest.fixture
def sample_request() -> ExecutionRequest:
    """Create a sample execution request."""
    return ExecutionRequest(
        prompt="Test prompt",
        model="sonnet",
        timeout_seconds=30,
    )


@pytest.fixture
def sample_schema() -> dict:
    """Sample JSON schema for validation tests."""
    return {
        "type": "object",
        "required": ["result", "confidence"],
        "properties": {
            "result": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }


@pytest.fixture
def valid_json_output() -> str:
    """Valid JSON matching sample_schema."""
    return '{"result": "test answer", "confidence": 0.95}'


@pytest.fixture
def invalid_json_output() -> str:
    """Invalid JSON (missing required field)."""
    return '{"result": "test answer"}'
