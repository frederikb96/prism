"""
Claude Search agent using built-in WebSearch/WebFetch tools.

Uses Claude CLI with built-in web tools for research.
WebSearch and WebFetch are native Claude tools (NOT MCP tools).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from prism.core.parsing import extract_content_from_cli_output
from prism.core.response import ExecutionRequest
from prism.prompts import get_registry
from prism.workers.base import Agent, AgentResult

if TYPE_CHECKING:
    from prism.workers.base import ExecutorProtocol

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = ("WebSearch", "WebFetch")


class ClaudeSearchAgent(Agent):
    """
    Claude-based researcher with WebSearch and WebFetch tools.

    Uses built-in Claude tools (no --tools mcp flag needed).
    Prompt composed via registry: system.md + workers/websearch.md.
    """

    def __init__(
        self,
        executor: ExecutorProtocol,
        model: str,
        timeout: int,
        visible_timeout: int,
        hooks_config: dict[str, Any] | None = None,
        env_vars: tuple[tuple[str, str], ...] | None = None,
        effort: str | None = None,
    ) -> None:
        self._executor = executor
        self._model = model
        self._timeout = timeout
        self._visible_timeout = visible_timeout
        self._hooks_config = hooks_config
        self._env_vars = env_vars
        self._effort = effort

    @property
    def agent_type(self) -> str:
        return "claude_search"

    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout
        system_prompt = get_registry().build_system_prompt("websearch")
        user_prompt = get_registry().build_user_prompt(prompt, self._visible_timeout)

        request = ExecutionRequest(
            prompt=user_prompt,
            model=self._model,
            timeout_seconds=timeout,
            allowed_tools=ALLOWED_TOOLS,
            system_prompt=system_prompt,
            hooks_config=self._hooks_config,
            env_vars=self._env_vars,
            no_session_persistence=True,
            effort=self._effort,
        )

        logger.debug(
            "ClaudeSearch executing",
            extra={"prompt_length": len(prompt), "timeout": timeout},
        )

        result = await self._executor.execute(request)

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Claude search execution failed",
                raw_output=result.output,
            )

        content = extract_content_from_cli_output(result.output)

        return AgentResult.from_success(
            content=content,
            raw_output=result.output,
            session_id=result.session_id,
        )
