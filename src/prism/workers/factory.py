"""
Worker factory for creating configured agent instances.

Shared factory used by both L0 flow and L1-3 dispatcher.
Maps agent_type string to the correct Agent with model from config.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from prism.config import get_config
from prism.core.hooks import build_claude_hooks, build_time_env_vars
from prism.workers.base import Agent
from prism.workers.claude_search import ClaudeSearchAgent
from prism.workers.gemini_search import GeminiSearchAgent
from prism.workers.perplexity_search import PerplexitySearchAgent
from prism.workers.tavily_search import TavilySearchAgent

if TYPE_CHECKING:
    from prism.core.gemini import GeminiExecutor
    from prism.core.retry import RetryExecutor

VALID_AGENT_TYPES = frozenset(
    {"claude_search", "tavily_search", "perplexity_search", "gemini_search"}
)

_CLAUDE_AGENT_CLASSES: dict[
    str, type[ClaudeSearchAgent | TavilySearchAgent | PerplexitySearchAgent]
] = {
    "claude_search": ClaudeSearchAgent,
    "tavily_search": TavilySearchAgent,
    "perplexity_search": PerplexitySearchAgent,
}


def create_worker(
    agent_type: str,
    claude_executor: RetryExecutor,
    gemini_executor: GeminiExecutor,
    level: int,
    timeout: int,
    visible_timeout: int,
) -> Agent:
    """
    Create a configured worker agent for the given type and level.

    Sets hook_log_path, worker_start_time, and worker_model on the returned
    agent for post-execution hook log parsing and completion logging.

    Args:
        agent_type: Worker type (claude_search, tavily_search, perplexity_search, gemini_search)
        claude_executor: RetryExecutor for Claude-based workers
        gemini_executor: GeminiExecutor for Gemini workers
        level: Search level (0-3) for model selection
        timeout: Hard timeout in seconds (process kill)
        visible_timeout: Tool budget in seconds (shown to agent)

    Returns:
        Configured Agent instance

    Raises:
        ValueError: Unknown agent type
    """
    if agent_type not in VALID_AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type: {agent_type!r}. "
            f"Valid types: {', '.join(sorted(VALID_AGENT_TYPES))}"
        )

    config = get_config()
    start = time.time()
    log_path = f"/tmp/prism-hook-{uuid.uuid4()}.log"

    if agent_type == "gemini_search":
        model_cfg = config.models.gemini_workers[level]
        env_vars = build_time_env_vars(
            start_time=start,
            tool_timeout=visible_timeout,
            hook_format="gemini",
            log_path=log_path,
        )
        worker: Agent = GeminiSearchAgent(
            executor=gemini_executor,
            model=model_cfg.model,
            timeout=timeout,
            visible_timeout=visible_timeout,
            env_vars=env_vars,
        )
    else:
        model_cfg = config.models.claude_workers[level]
        hooks_config = build_claude_hooks()
        env_vars = build_time_env_vars(
            start_time=start,
            tool_timeout=visible_timeout,
            hook_format="claude",
            log_path=log_path,
        )

        cls = _CLAUDE_AGENT_CLASSES[agent_type]
        worker = cls(
            executor=claude_executor,
            model=model_cfg.model,
            timeout=timeout,
            visible_timeout=visible_timeout,
            hooks_config=hooks_config,
            env_vars=env_vars,
            effort=model_cfg.effort,
        )

    worker.hook_log_path = log_path  # type: ignore[attr-defined]
    worker.worker_start_time = start  # type: ignore[attr-defined]
    worker.worker_model = model_cfg.model  # type: ignore[attr-defined]
    return worker
