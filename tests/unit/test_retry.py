"""Tests for RetryExecutor."""

from __future__ import annotations

import pytest

from prism.config import RetryConfig
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.retry import RetryExecutor, build_validation_retry_prompt

from .conftest import MockExecutor


class TestBuildValidationRetryPrompt:
    """Test retry prompt construction."""

    def test_includes_error(self) -> None:
        """Prompt includes validation error."""
        prompt = build_validation_retry_prompt(
            original_output='{"bad": "data"}',
            schema={"type": "object"},
            validation_error="missing required field 'name'",
        )

        assert "missing required field 'name'" in prompt

    def test_includes_schema(self) -> None:
        """Prompt includes expected schema."""
        schema = {"type": "object", "required": ["x"]}
        prompt = build_validation_retry_prompt(
            original_output="{}",
            schema=schema,
            validation_error="error",
        )

        assert '"required": [\n    "x"\n  ]' in prompt or '"required": ["x"]' in prompt

    def test_truncates_long_output(self) -> None:
        """Long output is truncated to 2000 chars."""
        long_output = "x" * 3000
        prompt = build_validation_retry_prompt(
            original_output=long_output,
            schema={},
            validation_error="error",
        )

        assert "..." in prompt
        assert "x" * 2000 in prompt
        assert "x" * 2001 not in prompt


class TestRetryExecutorTransientRetries:
    """Test inner loop (transient failure retries)."""

    @pytest.mark.asyncio
    async def test_success_no_retry(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Successful execution doesn't trigger retries."""
        mock_executor.add_result(ExecutionResult.from_success("done", "sess-1"))

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request)

        assert result.success is True
        assert len(mock_executor.calls) == 1

    @pytest.mark.asyncio
    async def test_timeout_not_retried(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Timeout returns immediately without retry."""
        mock_executor.add_result(ExecutionResult.from_timeout(30))
        mock_executor.add_result(ExecutionResult.from_success("done", "sess-1"))

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request)

        assert result.success is False
        assert result.is_timeout is True
        assert len(mock_executor.calls) == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Connection error triggers transient retry."""
        mock_executor.add_result(
            ExecutionResult.from_error("connection refused", exit_code=1)
        )
        mock_executor.add_result(ExecutionResult.from_success("done", "sess-1"))

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request)

        assert result.success is True
        assert len(mock_executor.calls) == 2

    @pytest.mark.asyncio
    async def test_max_transient_retries_exhausted(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Retries stop after max_transient_retries."""
        for _ in range(retry_config.max_transient_retries + 2):
            mock_executor.add_result(
                ExecutionResult.from_error("connection refused", exit_code=1)
            )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request)

        assert result.success is False
        assert len(mock_executor.calls) == retry_config.max_transient_retries + 1

    @pytest.mark.asyncio
    async def test_cancelled_not_retried(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Cancellation is intentional - don't retry."""
        mock_executor.add_result(ExecutionResult.from_cancelled())
        mock_executor.add_result(ExecutionResult.from_success("done"))

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request)

        assert result.is_cancelled is True
        assert len(mock_executor.calls) == 1


class TestRetryExecutorValidationRetries:
    """Test outer loop (validation failure retries)."""

    @pytest.mark.asyncio
    async def test_valid_output_no_retry(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
        sample_schema: dict,
        valid_json_output: str,
    ) -> None:
        """Valid schema output doesn't trigger retry."""
        mock_executor.add_result(ExecutionResult.from_success(valid_json_output, "sess-1"))

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=sample_schema)

        assert result.success is True
        assert len(mock_executor.calls) == 1

    @pytest.mark.asyncio
    async def test_invalid_output_retried(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
        sample_schema: dict,
        invalid_json_output: str,
        valid_json_output: str,
    ) -> None:
        """Invalid schema triggers validation retry with --resume."""
        mock_executor.add_result(
            ExecutionResult.from_success(invalid_json_output, "sess-1")
        )
        mock_executor.add_result(
            ExecutionResult.from_success(valid_json_output, "sess-1")
        )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=sample_schema)

        assert result.success is True
        assert len(mock_executor.calls) == 2

        retry_request = mock_executor.calls[1][0]
        assert retry_request.resume_session == "sess-1"
        assert "required property missing" in retry_request.prompt.lower()

    @pytest.mark.asyncio
    async def test_max_validation_retries_exhausted(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
        sample_schema: dict,
        invalid_json_output: str,
    ) -> None:
        """Retries stop after max_validation_retries."""
        for _ in range(retry_config.max_validation_retries + 2):
            mock_executor.add_result(
                ExecutionResult.from_success(invalid_json_output, "sess-1")
            )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=sample_schema)

        assert result.success is False
        assert "validation failed" in result.error_message.lower()
        assert len(mock_executor.calls) == retry_config.max_validation_retries + 1

    @pytest.mark.asyncio
    async def test_invalid_json_triggers_retry(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
        sample_schema: dict,
        valid_json_output: str,
    ) -> None:
        """Non-JSON output triggers validation retry."""
        mock_executor.add_result(
            ExecutionResult.from_success("This is not JSON", "sess-1")
        )
        mock_executor.add_result(
            ExecutionResult.from_success(valid_json_output, "sess-1")
        )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=sample_schema)

        assert result.success is True
        assert len(mock_executor.calls) == 2


class TestRetryExecutorTwoTierInteraction:
    """Test interaction between inner and outer retry loops."""

    @pytest.mark.asyncio
    async def test_transient_then_validation_retry(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
        sample_schema: dict,
        invalid_json_output: str,
        valid_json_output: str,
    ) -> None:
        """Transient retries don't count against validation budget."""
        mock_executor.add_result(
            ExecutionResult.from_error("connection refused", exit_code=1)
        )
        mock_executor.add_result(
            ExecutionResult.from_success(invalid_json_output, "sess-1")
        )
        mock_executor.add_result(
            ExecutionResult.from_success(valid_json_output, "sess-1")
        )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=sample_schema)

        assert result.success is True
        assert len(mock_executor.calls) == 3


class TestSchemaValidation:
    """Test built-in schema validation logic."""

    @pytest.mark.asyncio
    async def test_validates_object_type(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Validates object type correctly."""
        schema = {"type": "object", "required": ["name"]}
        mock_executor.add_result(
            ExecutionResult.from_success('{"name": "test"}', "sess-1")
        )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=schema)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_validates_array_type(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Validates array type correctly."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        mock_executor.add_result(
            ExecutionResult.from_success('["a", "b", "c"]', "sess-1")
        )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=schema)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_validates_nested_object(
        self,
        mock_executor: MockExecutor,
        retry_config: RetryConfig,
        sample_request: ExecutionRequest,
    ) -> None:
        """Validates nested object structure."""
        schema = {
            "type": "object",
            "required": ["data"],
            "properties": {
                "data": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
            },
        }
        mock_executor.add_result(
            ExecutionResult.from_success('{"data": {"value": 42}}', "sess-1")
        )

        retry_executor = RetryExecutor(mock_executor, retry_config)
        result = await retry_executor.execute(sample_request, schema=schema)

        assert result.success is True


class TestRetryConfig:
    """Test RetryConfig behavior."""

    def test_exponential_backoff(self) -> None:
        """Delay grows exponentially."""
        config = RetryConfig(
            max_transient_retries=3,
            base_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=100.0,
            max_validation_retries=5,
        )

        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0

    def test_delay_capped_at_max(self) -> None:
        """Delay is capped at max_delay_seconds."""
        config = RetryConfig(
            max_transient_retries=3,
            base_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=5.0,
            max_validation_retries=5,
        )

        assert config.get_delay(10) == 5.0
