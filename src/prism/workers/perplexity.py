"""
Perplexity agent using direct Perplexity API.

This agent does NOT use Claude - it calls Perplexity's API
directly via MCP for fast, factual answers. It is non-cancellable
because responses are instant.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prism.config import get_config
from prism.core.parsing import extract_content_from_cli_output
from prism.core.response import ExecutionRequest
from prism.workers.base import Agent, AgentResult

if TYPE_CHECKING:
    from prism.workers.base import ExecutorProtocol

logger = logging.getLogger(__name__)


class PerplexityAgent(Agent):
    """
    Direct Perplexity API agent.

    Uses mcp__perplexity__reason for fast, factual responses.
    This agent does NOT use Claude as an intermediary - it
    invokes Claude CLI only to access the Perplexity MCP tool.

    Non-cancellable because Perplexity responses are instant
    (no streaming, no long execution).
    """

    ALLOWED_TOOLS = ("mcp__perplexity__reason",)

    def __init__(
        self,
        executor: ExecutorProtocol,
        system_prompt: str | None = None,
        model: str | None = None,
        default_timeout: int | None = None,
    ) -> None:
        """
        Initialize Perplexity agent.

        Args:
            executor: Executor for Claude CLI calls (via DI)
            system_prompt: Custom system prompt for the invoking Claude
            model: Claude model (only used for MCP invocation)
            default_timeout: Default timeout in seconds
        """
        config = get_config()
        self._executor = executor
        self._system_prompt = system_prompt
        self._model = model or config.workers.perplexity.model
        self._default_timeout = default_timeout or config.workers.perplexity.default_timeout_seconds

    @property
    def agent_type(self) -> str:
        return "perplexity"

    @property
    def is_cancellable(self) -> bool:
        """Perplexity responses are instant - not cancellable."""
        return False

    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """
        Execute Perplexity query.

        The prompt is passed to Claude which will invoke the
        mcp__perplexity__reason tool with the query.

        Args:
            prompt: Query for Perplexity
            timeout_seconds: Optional timeout override
        """
        timeout = timeout_seconds or self._default_timeout

        wrapped_prompt = self._build_perplexity_prompt(prompt)

        request = ExecutionRequest(
            prompt=wrapped_prompt,
            model=self._model,
            timeout_seconds=timeout,
            tools="mcp",
            allowed_tools=self.ALLOWED_TOOLS,
            system_prompt=self._system_prompt,
        )

        logger.debug(
            "Perplexity agent executing",
            extra={"prompt_length": len(prompt), "timeout": timeout},
        )

        result = await self._executor.execute(request)

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Perplexity execution failed",
                raw_output=result.output,
            )

        content = self._extract_content(result.output)

        return AgentResult.from_success(
            content=content,
            raw_output=result.output,
            session_id=result.session_id,
        )

    def _build_perplexity_prompt(self, query: str) -> str:
        """
        Build prompt instructing Claude to use Perplexity tool.

        Args:
            query: User's search query
        """
        return f"""Use the mcp__perplexity__reason tool to answer this query.
Pass the query exactly as provided. Return the tool's response directly.

Query: {query}"""

    def _extract_content(self, raw_output: str) -> str:
        """Extract content from Claude CLI JSON output."""
        return extract_content_from_cli_output(raw_output)
