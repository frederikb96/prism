"""
Researcher agent using Claude with WebSearch/WebFetch tools.

This agent uses Claude CLI with built-in web tools to perform
research tasks. Output is unstructured text (no schema validation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from prism.config import get_config
from prism.core.parsing import extract_content_from_cli_output
from prism.core.response import ExecutionRequest
from prism.workers.base import Agent, AgentResult

if TYPE_CHECKING:
    from prism.workers.base import ExecutorProtocol

logger = logging.getLogger(__name__)


class ResearcherAgent(Agent):
    """
    Claude-based researcher with WebSearch and WebFetch tools.

    Uses Claude CLI with 'mcp' tools mode to access web search
    capabilities. Produces unstructured text output suitable for
    synthesis by downstream agents.
    """

    ALLOWED_TOOLS = (
        "WebSearch",
        "WebFetch",
    )

    def __init__(
        self,
        executor: ExecutorProtocol,
        system_prompt: str | None = None,
        model: str | None = None,
        default_timeout: int | None = None,
        hooks_config: dict[str, Any] | None = None,
        env_vars: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        """
        Initialize researcher agent.

        Args:
            executor: Executor for Claude CLI calls (via DI)
            system_prompt: Custom system prompt (loaded from prompts if None)
            model: Claude model to use
            default_timeout: Default timeout in seconds
            hooks_config: Claude hooks configuration for time awareness
            env_vars: Environment variables for hooks
        """
        config = get_config()
        self._executor = executor
        self._system_prompt = system_prompt
        self._model = model or config.workers.researcher.model
        self._default_timeout = default_timeout or config.workers.researcher.default_timeout_seconds
        self._hooks_config = hooks_config
        self._env_vars = env_vars

    @property
    def agent_type(self) -> str:
        return "researcher"

    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """
        Execute research task.

        Args:
            prompt: Research query/task
            timeout_seconds: Optional timeout override
        """
        timeout = timeout_seconds or self._default_timeout

        request = ExecutionRequest(
            prompt=prompt,
            model=self._model,
            timeout_seconds=timeout,
            tools="mcp",
            allowed_tools=self.ALLOWED_TOOLS,
            system_prompt=self._system_prompt,
            hooks_config=self._hooks_config,
            env_vars=self._env_vars,
        )

        logger.debug(
            "Researcher executing",
            extra={"prompt_length": len(prompt), "timeout": timeout},
        )

        result = await self._executor.execute(request)

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Researcher execution failed",
                raw_output=result.output,
            )

        content = self._extract_content(result.output)

        return AgentResult.from_success(
            content=content,
            raw_output=result.output,
            session_id=result.session_id,
        )

    def _extract_content(self, raw_output: str) -> str:
        """Extract text content from Claude CLI JSON output."""
        return extract_content_from_cli_output(raw_output)
