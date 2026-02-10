"""
Result synthesis.

Combines worker results into coherent final response.
Uses Claude CLI with --resume for context continuity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prism.config import get_config
from prism.core.parsing import extract_content_from_cli_output
from prism.core.response import ExecutionRequest

if TYPE_CHECKING:
    from prism.workers.base import AgentResult, ExecutorProtocol

logger = logging.getLogger(__name__)

# Synthesis system prompt
SYNTHESIS_SYSTEM_PROMPT = """\
You are a research synthesis assistant. Your job is to combine multiple \
search results into a clear, comprehensive response.

Guidelines:
- Synthesize information, don't just list results
- Highlight key findings and consensus
- Note any contradictions or uncertainties
- Maintain factual accuracy
- Be concise but thorough
- Use clear structure (headers, bullets) when appropriate"""


class ResultSynthesizer:
    """
    Synthesizes multiple worker results into a final response.

    Uses Claude CLI to intelligently combine results,
    optionally with --resume for context continuity.
    """

    def __init__(
        self,
        executor: ExecutorProtocol,
        model: str | None = None,
        default_timeout: int | None = None,
    ) -> None:
        """
        Initialize synthesizer.

        Args:
            executor: Executor for Claude CLI calls
            model: Claude model to use (from config if None)
            default_timeout: Default timeout in seconds (from config if None)
        """
        config = get_config()
        self._executor = executor
        self._model = model or config.synthesizer.model
        self._default_timeout = default_timeout or config.synthesizer.default_timeout_seconds

    async def synthesize(
        self,
        original_query: str,
        results: list[AgentResult],
        resume_session: str | None = None,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """
        Synthesize worker results into final response.

        Args:
            original_query: User's original search query
            results: List of AgentResults from workers
            resume_session: Optional session ID for context continuity
            timeout_seconds: Optional timeout override

        Returns:
            AgentResult with synthesized content
        """
        from prism.workers.base import AgentResult

        timeout = timeout_seconds or self._default_timeout

        # Filter successful results
        successful = [r for r in results if r.success]

        if not successful:
            logger.warning("No successful results to synthesize")
            return AgentResult.from_error(
                error="All worker results failed - nothing to synthesize",
            )

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(original_query, successful)

        request = ExecutionRequest(
            prompt=prompt,
            model=self._model,
            timeout_seconds=timeout,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            resume_session=resume_session,
        )

        logger.debug(
            "Synthesizing results",
            extra={
                "result_count": len(successful),
                "resume": resume_session is not None,
            },
        )

        result = await self._executor.execute(request)

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Synthesis failed",
                raw_output=result.output,
            )

        # Extract content from Claude CLI output
        content = self._extract_content(result.output)

        return AgentResult.from_success(
            content=content,
            raw_output=result.output,
            session_id=result.session_id,
            synthesized_count=len(successful),
        )

    def _build_synthesis_prompt(
        self,
        query: str,
        results: list[AgentResult],
    ) -> str:
        """
        Build synthesis prompt from query and results.

        Args:
            query: Original search query
            results: Successful worker results

        Returns:
            Prompt for synthesis
        """
        sections = [f"Original Query: {query}", "", "Research Results:"]

        for i, result in enumerate(results, 1):
            content = result.content if isinstance(result.content, str) else str(result.content)
            sections.append(f"\n--- Source {i} ---")
            sections.append(content[:4000])  # Truncate very long results

        sections.append("\n---")
        sections.append(
            "\nSynthesize these results into a comprehensive response to the original query."
        )

        return "\n".join(sections)

    def _extract_content(self, raw_output: str) -> str:
        """Extract content from Claude CLI JSON output."""
        return extract_content_from_cli_output(raw_output)
