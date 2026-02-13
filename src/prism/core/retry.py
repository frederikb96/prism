"""
Two-tier retry executor for Claude CLI.

Inner loop: Transient failures (timeouts, crashes) with exponential backoff
Outer loop: Validation failures (schema mismatch) with --resume
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from prism.config import RetryConfig, get_config
from prism.core.response import ExecutionRequest, ExecutionResult

if TYPE_CHECKING:
    from prism.workers.base import ExecutorProtocol

logger = logging.getLogger(__name__)


def build_validation_retry_prompt(
    original_output: str,
    schema: dict[str, Any],
    validation_error: str,
) -> str:
    """
    Build a prompt for validation retry using --resume.

    Args:
        original_output: The output that failed validation
        schema: Expected JSON schema
        validation_error: Description of validation failure
    """
    return f"""Your previous output did not match the required JSON schema.

VALIDATION ERROR:
{validation_error}

EXPECTED SCHEMA:
{json.dumps(schema, indent=2)}

YOUR PREVIOUS OUTPUT:
{original_output[:2000]}{"..." if len(original_output) > 2000 else ""}

Please provide a corrected response that matches the schema exactly.
Output ONLY the valid JSON, no additional text."""


class RetryExecutor:
    """
    Executor with two-tier retry logic.

    Inner Loop (Transient Retries):
    - Catches: subprocess crash, connection issues
    - Timeouts and cancellations return immediately (no retry)
    - Exponential backoff: 1s, 2s, 4s (configurable)
    - Does NOT count against validation budget
    - Max 3 retries by default

    Outer Loop (Validation Retries):
    - Triggered when output doesn't match schema
    - Uses --resume with the session to continue context
    - Includes validation error in retry prompt
    - Max 5 retries by default
    """

    def __init__(
        self,
        executor: ExecutorProtocol,
        config: RetryConfig | None = None,
    ) -> None:
        """
        Initialize retry executor.

        Args:
            executor: Underlying executor (typically ClaudeExecutor)
            config: Retry configuration (uses global config if not provided)
        """
        self._executor = executor
        self._config = config or get_config().retry

    async def execute(
        self,
        request: ExecutionRequest,
        schema: dict[str, Any] | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute request with two-tier retry logic.

        Args:
            request: Execution request
            schema: Optional JSON schema for output validation
            session_id: Optional session ID for tracking
            parent_session_id: Optional parent session for cancel tracking
        """
        current_request = request
        last_result: ExecutionResult | None = None
        last_session_id: str | None = session_id

        for validation_attempt in range(self._config.max_validation_retries + 1):
            is_retry = validation_attempt > 0

            if is_retry:
                logger.info(
                    "Validation retry attempt %d/%d",
                    validation_attempt,
                    self._config.max_validation_retries,
                )

            result = await self._execute_with_transient_retries(
                current_request,
                session_id=last_session_id,
                parent_session_id=parent_session_id,
            )

            if result.session_id:
                last_session_id = result.session_id

            if not result.success:
                return result

            if schema is None:
                return result

            validation_error = self._validate_output(result.output, schema)
            if validation_error is None:
                return result

            last_result = result

            if validation_attempt < self._config.max_validation_retries:
                retry_prompt = build_validation_retry_prompt(
                    original_output=result.output,
                    schema=schema,
                    validation_error=validation_error,
                )

                current_request = request.with_changes(
                    prompt=retry_prompt,
                    resume_session=last_session_id,
                )

                logger.warning(
                    "Schema validation failed, will retry with --resume",
                    extra={"validation_error": validation_error},
                )

        return ExecutionResult.from_error(
            message=f"Schema validation failed after {self._config.max_validation_retries} retries",
            output=last_result.output if last_result else "",
        )

    async def _execute_with_transient_retries(
        self,
        request: ExecutionRequest,
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute with retries for transient failures only.

        Args:
            request: Execution request
            session_id: Optional session ID
            parent_session_id: Optional parent session for cancel tracking
        """
        last_error: Exception | None = None

        for attempt in range(self._config.max_transient_retries + 1):
            is_retry = attempt > 0

            if is_retry:
                delay = self._config.get_delay(attempt - 1)
                logger.info(
                    "Transient retry attempt %d/%d after %.1fs delay",
                    attempt,
                    self._config.max_transient_retries,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                result = await self._executor.execute(
                    request,
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )

                if result.is_timeout or result.is_cancelled:
                    return result

                if not result.success and result.is_transient_error():
                    if attempt < self._config.max_transient_retries:
                        logger.warning(
                            "Transient error: %s, will retry",
                            result.error_message,
                        )
                        continue

                return result

            except Exception as e:
                last_error = e
                if attempt < self._config.max_transient_retries:
                    logger.warning("Exception during execution: %s, will retry", e)
                    continue
                raise

        if last_error:
            return ExecutionResult.from_error(str(last_error))

        return ExecutionResult.from_error("Unknown error during retry")

    def _validate_output(
        self,
        output: str,
        schema: dict[str, Any],
    ) -> str | None:
        """
        Validate output against JSON schema.

        Handles Claude CLI wrapper format automatically:
        - With --json-schema: uses "structured_output" field
        - Without: uses "result" field

        Args:
            output: Raw output string
            schema: JSON schema to validate against
        """
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        if isinstance(data, dict) and data.get("type") == "result":
            if "structured_output" in data:
                data = data["structured_output"]
            elif "result" in data:
                result = data["result"]
                if isinstance(result, str):
                    try:
                        data = json.loads(result)
                    except json.JSONDecodeError as e:
                        return f"Inner result is not valid JSON: {e}"
                else:
                    data = result

        return self._validate_against_schema(data, schema)

    def _validate_against_schema(
        self,
        data: Any,
        schema: dict[str, Any],
        path: str = "",
    ) -> str | None:
        """
        Simple recursive schema validation.

        Args:
            data: Data to validate
            schema: JSON schema
            path: Current path for error messages
        """
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                return f"{path or 'root'}: expected object, got {type(data).__name__}"

            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    return f"{path}.{prop}: required property missing"

            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    error = self._validate_against_schema(
                        data[prop],
                        prop_schema,
                        f"{path}.{prop}" if path else prop,
                    )
                    if error:
                        return error

        elif schema_type == "array":
            if not isinstance(data, list):
                return f"{path or 'root'}: expected array, got {type(data).__name__}"

            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    error = self._validate_against_schema(
                        item,
                        items_schema,
                        f"{path}[{i}]",
                    )
                    if error:
                        return error

        elif schema_type == "string":
            if not isinstance(data, str):
                return f"{path or 'root'}: expected string, got {type(data).__name__}"

        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                return f"{path or 'root'}: expected number, got {type(data).__name__}"

        elif schema_type == "integer":
            if not isinstance(data, int) or isinstance(data, bool):
                return f"{path or 'root'}: expected integer, got {type(data).__name__}"

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                return f"{path or 'root'}: expected boolean, got {type(data).__name__}"

        return None
