"""
Claude CLI executor.

THE single place where `claude -p` is invoked. All Claude CLI calls
must go through ClaudeExecutor.execute() to maintain DRY principle.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from prism.config import get_config
from prism.core.process import CancellableProcess
from prism.core.response import ExecutionRequest, ExecutionResult

if TYPE_CHECKING:
    from prism.core.session import SessionRegistry

logger = logging.getLogger(__name__)


class ClaudeExecutor:
    """
    Executor for Claude CLI commands.

    Builds and executes `claude -p` commands with proper argument
    handling, environment setup, and result parsing.

    OAuth authentication is handled via CLAUDE_CODE_OAUTH_TOKEN
    environment variable which Claude CLI reads automatically.
    """

    def __init__(
        self,
        session_registry: SessionRegistry | None = None,
    ) -> None:
        """
        Initialize executor.

        Args:
            session_registry: Optional registry for tracking sessions
        """
        self._session_registry = session_registry

    def build_command(self, request: ExecutionRequest) -> list[str]:
        """
        Build Claude CLI command from request.

        Args:
            request: Execution request with prompt and options

        Returns:
            List of command arguments for subprocess
        """
        cmd = [
            "claude",
            "-p", request.prompt,
            "--model", request.model,
            "--output-format", "json",
        ]

        if request.tools:
            cmd.extend(["--tools", request.tools])

        for tool in request.allowed_tools:
            cmd.extend(["--allowedTools", tool])

        if request.json_schema:
            schema_str = json.dumps(request.json_schema)
            cmd.extend(["--json-schema", schema_str])

        if request.system_prompt:
            cmd.extend(["--system-prompt", request.system_prompt])

        if request.resume_session:
            cmd.extend(["--resume", request.resume_session])

        if request.hooks_config:
            settings_json = json.dumps(request.hooks_config)
            cmd.extend(["--settings", settings_json])

        if request.mcp_config:
            cmd.extend(["--mcp-config", json.dumps(request.mcp_config)])

        if request.strict_mcp:
            cmd.append("--strict-mcp-config")

        if request.no_session_persistence:
            cmd.append("--no-session-persistence")

        return cmd

    async def execute(
        self,
        request: ExecutionRequest,
        schema: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a Claude CLI command.

        This is THE ONLY place where `claude -p` is invoked.

        Args:
            request: Execution request with prompt and options
            session_id: Optional session ID for tracking (generated if not provided)

        Returns:
            ExecutionResult with output and metadata
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        cmd = self.build_command(request)

        logger.debug(
            "Executing Claude CLI",
            extra={
                "session_id": session_id,
                "model": request.model,
                "timeout": request.timeout_seconds,
                "has_schema": request.json_schema is not None,
                "resume": request.resume_session,
            },
        )

        config = get_config()
        env_dict = dict(request.env_vars) if request.env_vars else {}

        if request.effort:
            env_dict["CLAUDE_CODE_EFFORT_LEVEL"] = request.effort

        env = env_dict or None

        process = CancellableProcess(
            cmd=cmd,
            env=env,
            timeout_seconds=request.timeout_seconds,
            sigterm_timeout=config.process.sigterm_timeout_seconds,
            sigkill_timeout=config.process.sigkill_timeout_seconds,
        )

        # Register session if we have a registry
        if self._session_registry:
            await self._session_registry.register(session_id, process)

        try:
            result = await process.run()

            if process.is_cancelled:
                return ExecutionResult.from_cancelled()

            if not result.success:
                return ExecutionResult.from_error(
                    message=result.stderr or f"Claude CLI exited with code {result.returncode}",
                    exit_code=result.returncode,
                    output=result.stdout,
                )

            # Parse session ID from output if present
            output_session_id = self._extract_session_id(result.stdout) or session_id

            return ExecutionResult.from_success(
                output=result.stdout,
                session_id=output_session_id,
            )

        except TimeoutError:
            return ExecutionResult.from_timeout(request.timeout_seconds)

        except Exception as e:
            logger.exception("Unexpected error during Claude CLI execution")
            return ExecutionResult.from_error(
                message=str(e),
                exit_code=-1,
            )

        finally:
            # Unregister session
            if self._session_registry:
                await self._session_registry.unregister(session_id)

    def _extract_session_id(self, output: str) -> str | None:
        """
        Extract session ID from Claude CLI JSON output.

        Args:
            output: Raw stdout from Claude CLI

        Returns:
            Session ID if found, None otherwise
        """
        try:
            # Claude CLI outputs JSON with session_id field
            data = json.loads(output)
            if isinstance(data, dict):
                return data.get("session_id")
        except json.JSONDecodeError:
            # Try regex fallback for partial output
            match = re.search(r'"session_id"\s*:\s*"([^"]+)"', output)
            if match:
                return match.group(1)

        return None
