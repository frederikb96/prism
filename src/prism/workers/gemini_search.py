"""
Gemini Search agent using GeminiExecutor with google_web_search.

Uses Gemini CLI for web research. System prompt via temp file,
hooks via Gemini settings file (handled by GeminiExecutor).
"""

from __future__ import annotations

import logging

from prism.core.gemini import GeminiExecutor
from prism.core.response import ExecutionRequest
from prism.prompts import get_registry
from prism.workers.base import Agent, AgentResult

logger = logging.getLogger(__name__)


class GeminiSearchAgent(Agent):
    """
    Gemini-based web search agent.

    Uses GeminiExecutor which handles system prompt temp files,
    settings file for hooks, and environment setup.
    """

    def __init__(
        self,
        executor: GeminiExecutor,
        model: str,
        timeout: int,
        visible_timeout: int,
        env_vars: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        self._executor = executor
        self._model = model
        self._timeout = timeout
        self._visible_timeout = visible_timeout
        self._env_vars = env_vars

    @property
    def agent_type(self) -> str:
        return "gemini_search"

    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        timeout = timeout_seconds if timeout_seconds is not None else self._timeout
        system_prompt = get_registry().build_system_prompt("gemini")
        user_prompt = get_registry().build_user_prompt(prompt, self._visible_timeout)

        env_vars = self._env_vars or ()

        request = ExecutionRequest(
            prompt=user_prompt,
            model=self._model,
            timeout_seconds=timeout,
            system_prompt=system_prompt,
            env_vars=env_vars,
        )

        logger.debug(
            "GeminiSearch executing",
            extra={"prompt_length": len(prompt), "timeout": timeout},
        )

        result = await self._executor.execute(request)

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Gemini search execution failed",
                raw_output=result.output,
            )

        parsed = GeminiExecutor.parse_gemini_output(result.output)
        content = parsed.get("response", result.output)
        if not content:
            content = result.output

        return AgentResult.from_success(
            content=content,
            raw_output=result.output,
            session_id=result.session_id,
        )
