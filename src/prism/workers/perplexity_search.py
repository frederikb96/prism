"""
Perplexity Search agent using Claude with Perplexity MCP tools.

Full Claude worker with both mcp__perplexity__search and
mcp__perplexity__reason tools via MCP.
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

ALLOWED_TOOLS = ("mcp__perplexity__search", "mcp__perplexity__reason")

MCP_CONFIG: dict[str, Any] = {
    "mcpServers": {
        "perplexity": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "perplexity-mcp"],
            "env": {"PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"},
        },
    },
}


class PerplexitySearchAgent(Agent):
    """
    Claude-based agent with Perplexity MCP tools.

    Uses both perplexity search and reason tools via MCP.
    Requires --tools mcp flag for MCP tool access.
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
        return "perplexity_search"

    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout
        system_prompt = get_registry().build_system_prompt("perplexity")
        user_prompt = get_registry().build_user_prompt(prompt, self._visible_timeout)

        request = ExecutionRequest(
            prompt=user_prompt,
            model=self._model,
            timeout_seconds=timeout,
            tools="mcp",
            allowed_tools=ALLOWED_TOOLS,
            system_prompt=system_prompt,
            hooks_config=self._hooks_config,
            env_vars=self._env_vars,
            mcp_config=MCP_CONFIG,
            strict_mcp=True,
            no_session_persistence=True,
            effort=self._effort,
        )

        logger.debug(
            "PerplexitySearch executing",
            extra={"prompt_length": len(prompt), "timeout": timeout},
        )

        result = await self._executor.execute(request)

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Perplexity search execution failed",
                raw_output=result.output,
            )

        content = extract_content_from_cli_output(result.output)

        return AgentResult.from_success(
            content=content,
            raw_output=result.output,
            session_id=result.session_id,
        )
