"""
Gemini CLI executor.

THE single place where `gemini -p` is invoked. All Gemini CLI calls
must go through GeminiExecutor.execute() to maintain DRY principle.

Includes built-in transient retry (no outer-tier schema validation retry).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from prism.config import RetryConfig, get_config
from prism.core.hooks import build_gemini_settings_file
from prism.core.process import CancellableProcess
from prism.core.response import ExecutionRequest, ExecutionResult

if TYPE_CHECKING:
    from prism.core.session import SessionRegistry

logger = logging.getLogger(__name__)


class GeminiExecutor:
    """
    Executor for Gemini CLI commands.

    Builds and executes `gemini -p` commands with proper argument
    handling, environment setup, temp file management, and result parsing.

    Includes built-in transient retry logic (inner-tier only).
    Gemini does not support --resume or --json-schema, so no outer-tier retry.
    """

    def __init__(
        self,
        session_registry: SessionRegistry | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._session_registry = session_registry
        self._retry_config = retry_config

    def _get_retry_config(self) -> RetryConfig:
        if self._retry_config is not None:
            return self._retry_config
        return get_config().retry

    def build_command(self, request: ExecutionRequest) -> list[str]:
        """
        Build Gemini CLI command from request.

        Args:
            request: Execution request with prompt and options

        Returns:
            List of command arguments for subprocess
        """
        cmd = [
            "gemini",
            "-p", request.prompt,
            "--model", request.model,
            "--allowed-tools", "google_web_search",
            "-o", "json",
            "--yolo",
        ]
        return cmd

    def _build_env(self, request: ExecutionRequest) -> tuple[dict[str, str], list[str]]:
        """
        Build environment variables and collect temp file paths for cleanup.

        Returns:
            Tuple of (env dict, list of temp file paths to clean up)
        """
        env = dict(request.env_vars) if request.env_vars else {}
        temp_files: list[str] = []

        # System prompt via temp file
        if request.system_prompt:
            sys_md_path = f"/tmp/prism-gemini-{uuid.uuid4()}.md"
            with open(sys_md_path, "w") as f:
                f.write(request.system_prompt)
            env["GEMINI_SYSTEM_MD"] = sys_md_path
            temp_files.append(sys_md_path)

        # Gemini settings file for hooks
        settings_path = build_gemini_settings_file()
        env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = settings_path
        temp_files.append(settings_path)

        # Map GOOGLE_API_KEY to GEMINI_API_KEY if needed
        full_env = os.environ.copy()
        full_env.update(env)
        if "GEMINI_API_KEY" not in full_env and "GOOGLE_API_KEY" in full_env:
            env["GEMINI_API_KEY"] = full_env["GOOGLE_API_KEY"]

        return env, temp_files

    async def execute(
        self,
        request: ExecutionRequest,
        schema: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a Gemini CLI command with transient retry.

        Args:
            request: Execution request with prompt and options
            schema: Ignored (Gemini has no --json-schema support)
            session_id: Optional session ID for tracking

        Returns:
            ExecutionResult with output and metadata
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        retry_cfg = self._get_retry_config()
        last_result: ExecutionResult | None = None

        for attempt in range(retry_cfg.max_transient_retries + 1):
            if attempt > 0:
                delay = retry_cfg.get_delay(attempt - 1)
                logger.info(
                    "Gemini transient retry %d/%d after %.1fs",
                    attempt, retry_cfg.max_transient_retries, delay,
                )
                await asyncio.sleep(delay)

            result = await self._execute_once(request, session_id)

            if result.is_cancelled:
                return result

            if result.is_timeout and attempt < retry_cfg.max_transient_retries:
                logger.warning("Gemini execution timed out, will retry")
                continue

            if not result.success and result.is_transient_error():
                if attempt < retry_cfg.max_transient_retries:
                    logger.warning("Gemini transient error: %s, will retry", result.error_message)
                    last_result = result
                    continue

            return result

        return last_result or ExecutionResult.from_error("Unknown error during Gemini retry")

    async def _execute_once(
        self,
        request: ExecutionRequest,
        session_id: str,
    ) -> ExecutionResult:
        """Execute a single Gemini CLI invocation."""
        cmd = self.build_command(request)
        env, temp_files = self._build_env(request)

        logger.debug(
            "Executing Gemini CLI",
            extra={
                "session_id": session_id,
                "model": request.model,
                "timeout": request.timeout_seconds,
            },
        )

        config = get_config()
        process = CancellableProcess(
            cmd=cmd,
            env=env,
            timeout_seconds=request.timeout_seconds,
            sigterm_timeout=config.process.sigterm_timeout_seconds,
            sigkill_timeout=config.process.sigkill_timeout_seconds,
        )

        if self._session_registry:
            await self._session_registry.register(session_id, process)

        try:
            result = await process.run()

            if process.is_cancelled:
                return ExecutionResult.from_cancelled()

            if not result.success:
                return ExecutionResult.from_error(
                    message=result.stderr or f"Gemini CLI exited with code {result.returncode}",
                    exit_code=result.returncode,
                    output=result.stdout,
                )

            return ExecutionResult.from_success(
                output=result.stdout,
                session_id=session_id,
            )

        except TimeoutError:
            return ExecutionResult.from_timeout(request.timeout_seconds)

        except Exception as e:
            logger.exception("Unexpected error during Gemini CLI execution")
            return ExecutionResult.from_error(
                message=str(e),
                exit_code=-1,
            )

        finally:
            if self._session_registry:
                await self._session_registry.unregister(session_id)
            self._cleanup_temp_files(temp_files)

    @staticmethod
    def _cleanup_temp_files(paths: list[str]) -> None:
        """Remove temporary files, ignoring errors."""
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def parse_gemini_output(raw_output: str) -> dict[str, Any]:
        """
        Parse Gemini CLI JSON output.

        Extracts response content and stats from the JSON output format:
        {"session_id": ..., "response": ..., "stats": {...}}

        Args:
            raw_output: Raw stdout from Gemini CLI

        Returns:
            Dict with 'response', 'tool_usage', and 'token_counts' keys
        """
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            return {"response": raw_output, "tool_usage": {}, "token_counts": {}}

        if not isinstance(data, dict):
            return {"response": raw_output, "tool_usage": {}, "token_counts": {}}

        response = data.get("response", "")
        stats = data.get("stats", {})

        tool_usage = {}
        if isinstance(stats, dict):
            tools = stats.get("tools", {})
            if isinstance(tools, dict):
                tool_usage = tools.get("byName", {})

        token_counts = {}
        if isinstance(stats, dict):
            models = stats.get("models", {})
            if isinstance(models, dict):
                for model_name, model_stats in models.items():
                    if isinstance(model_stats, dict):
                        token_counts[model_name] = model_stats.get("tokens", {})

        return {
            "response": response,
            "tool_usage": tool_usage,
            "token_counts": token_counts,
        }
