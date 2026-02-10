"""
Full search flow coordination.

Orchestrates the complete search process:
- Level 0: Direct Perplexity (no orchestration)
- Level 1-3: Manager -> Dispatch workers -> Synthesize
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from prism.config import get_config
from prism.database import SessionStatus
from prism.workers import ManagerAgent, PerplexityAgent

if TYPE_CHECKING:
    from prism.core.retry import RetryExecutor
    from prism.core.session import SessionRegistry
    from prism.database import SearchSessionRepository
    from prism.orchestrator.dispatcher import WorkerDispatcher
    from prism.orchestrator.synthesizer import ResultSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Result of a search operation.

    Human-readable content with metadata.
    """

    success: bool
    content: str
    session_id: str | None = None
    level: int = 0
    query: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP response."""
        return {
            "success": self.success,
            "content": self.content,
            "session_id": self.session_id,
            "level": self.level,
            "query": self.query,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class SearchFlow:
    """
    Coordinates full search flow.

    Level 0: Direct Perplexity call (instant)
    Level 1-3: Manager planning -> Worker dispatch -> Result synthesis
    """

    def __init__(
        self,
        retry_executor: RetryExecutor,
        dispatcher: WorkerDispatcher,
        synthesizer: ResultSynthesizer,
        session_registry: SessionRegistry,
        session_repository: SearchSessionRepository,
        user_id: str,
    ) -> None:
        """
        Initialize search flow.

        Args:
            retry_executor: Retry executor for all agents (includes transient retry)
            dispatcher: Worker dispatcher (already configured with retry_executor)
            synthesizer: Result synthesizer (already configured with retry_executor)
            session_registry: Session registry for tracking
            session_repository: Repository for session persistence
            user_id: User ID for multi-tenancy
        """
        self._retry_executor = retry_executor
        self._dispatcher = dispatcher
        self._synthesizer = synthesizer
        self._session_registry = session_registry
        self._session_repository = session_repository
        self._user_id = user_id

    async def execute_search(
        self,
        query: str,
        level: int = 1,
    ) -> SearchResult:
        """
        Execute a search at the specified level.

        Args:
            query: Search query
            level: Search depth (0-3)

        Returns:
            SearchResult with content and metadata
        """
        config = get_config()

        if level < 0 or level > 3:
            return SearchResult(
                success=False,
                content="",
                error=f"Invalid level: {level}. Must be 0-3.",
                level=level,
                query=query,
            )

        if not query or not query.strip():
            return SearchResult(
                success=False,
                content="",
                error="Query cannot be empty",
                level=level,
                query=query,
            )

        max_query_length = config.search.max_query_length
        if len(query) > max_query_length:
            return SearchResult(
                success=False,
                content="",
                error=f"Query too long: {len(query)} chars (max {max_query_length})",
                level=level,
                query=query[:100] + "...",
            )

        session_uuid = uuid.uuid4()
        session_id = str(session_uuid)
        start_time = time.monotonic()

        logger.info(
            "Starting search",
            extra={"query_length": len(query), "level": level, "session_id": session_id},
        )

        await self._session_repository.create(
            user_id=self._user_id,
            query=query,
            prompt=query,
            level=level,
            session_id=session_uuid,
        )

        await self._session_repository.update(
            session_uuid,
            status=SessionStatus.RUNNING,
        )

        try:
            await self._session_registry.register(session_id)

            if level == 0:
                result = await self._execute_level_0(query, session_id)
            else:
                result = await self._execute_level_1_3(query, level, session_id)

            duration_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = datetime.now(timezone.utc)

            await self._session_repository.update(
                session_uuid,
                status=SessionStatus.COMPLETED if result.success else SessionStatus.FAILED,
                result=result.to_dict(),
                summary=self._extract_summary(result),
                error_message=result.error,
                completed_at=completed_at,
                duration_ms=duration_ms,
                claude_session_id=result.session_id if level > 0 else None,
            )

            return result

        except Exception as e:
            logger.exception("Search failed", extra={"session_id": session_id})
            duration_ms = int((time.monotonic() - start_time) * 1000)

            await self._session_repository.update(
                session_uuid,
                status=SessionStatus.FAILED,
                error_message=str(e),
                completed_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
            )

            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error=f"Search failed: {e}",
            )

        finally:
            await self._session_registry.unregister(session_id)

    def _extract_summary(self, result: SearchResult) -> str | None:
        """
        Extract a short summary from search result.

        Args:
            result: SearchResult to summarize

        Returns:
            Summary string or None if extraction fails
        """
        if not result.success or not result.content:
            return None

        content = result.content
        if len(content) <= 200:
            return content

        first_para_end = content.find("\n\n")
        if first_para_end > 0 and first_para_end <= 300:
            return content[:first_para_end]

        return content[:200] + "..."

    async def _execute_level_0(
        self,
        query: str,
        session_id: str,
    ) -> SearchResult:
        """
        Execute Level 0 search (direct Perplexity).

        Args:
            query: Search query
            session_id: Session ID for tracking

        Returns:
            SearchResult from Perplexity
        """
        config = get_config()
        level_config = config.levels[0]

        perplexity = PerplexityAgent(
            executor=self._retry_executor,
            default_timeout=level_config.worker_timeout_seconds,
        )

        timeout = level_config.worker_timeout_seconds
        result = await perplexity.execute(query, timeout_seconds=timeout)

        if not result.success:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=0,
                query=query,
                error=result.error or "Perplexity query failed",
            )

        content = result.content if isinstance(result.content, str) else str(result.content)

        return SearchResult(
            success=True,
            content=content,
            session_id=result.session_id or session_id,
            level=0,
            query=query,
            metadata={"agent": "perplexity"},
        )

    async def _execute_level_1_3(
        self,
        query: str,
        level: int,
        session_id: str,
    ) -> SearchResult:
        """
        Execute Level 1-3 search (orchestrated).

        Flow: Manager -> Dispatch -> Synthesize

        Args:
            query: Search query
            level: Search level (1, 2, or 3)
            session_id: Session ID for tracking

        Returns:
            SearchResult with synthesized content
        """
        config = get_config()
        level_config = config.levels[level]

        manager = ManagerAgent(
            executor=self._retry_executor,
            default_timeout=level_config.manager_timeout_seconds,
        )

        logger.debug("Manager creating task plan", extra={"level": level})

        manager_timeout = level_config.manager_timeout_seconds
        plan_result = await manager.execute(query, timeout_seconds=manager_timeout)

        if not plan_result.success:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error=f"Manager planning failed: {plan_result.error}",
            )

        from prism.workers.manager import TaskPlan

        if isinstance(plan_result.content, dict):
            task_plan = TaskPlan.from_dict(plan_result.content)
        else:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error="Manager returned invalid task plan",
            )

        if not task_plan.tasks:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error="Manager produced empty task plan",
            )

        logger.info(
            "Task plan created",
            extra={"task_count": len(task_plan.tasks), "reasoning": task_plan.reasoning[:100]},
        )

        worker_results = await self._dispatcher.dispatch(
            task_plan=task_plan,
            worker_timeout=level_config.worker_timeout_seconds,
            visible_timeout=level_config.worker_visible_timeout,
        )

        successful_count = sum(1 for r in worker_results if r.success)
        if successful_count == 0:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error="All workers failed",
                metadata={"task_count": len(task_plan.tasks)},
            )

        synthesis_timeout = level_config.manager_timeout_seconds
        synthesis_result = await self._synthesizer.synthesize(
            original_query=query,
            results=worker_results,
            resume_session=plan_result.session_id,
            timeout_seconds=synthesis_timeout,
        )

        if not synthesis_result.success:
            content = self._fallback_combine(worker_results)
            return SearchResult(
                success=True,
                content=content,
                session_id=session_id,
                level=level,
                query=query,
                metadata={
                    "fallback": True,
                    "task_count": len(task_plan.tasks),
                    "successful_workers": successful_count,
                },
            )

        content = (
            synthesis_result.content
            if isinstance(synthesis_result.content, str)
            else str(synthesis_result.content)
        )

        return SearchResult(
            success=True,
            content=content,
            session_id=synthesis_result.session_id or session_id,
            level=level,
            query=query,
            metadata={
                "task_count": len(task_plan.tasks),
                "successful_workers": successful_count,
                "reasoning": task_plan.reasoning,
            },
        )

    def _fallback_combine(self, results: list) -> str:
        """
        Fallback: concatenate successful results.

        Args:
            results: List of AgentResults
        """
        sections = []
        for i, result in enumerate(results, 1):
            if result.success:
                content = result.content if isinstance(result.content, str) else str(result.content)
                sections.append(f"## Source {i}\n\n{content}")

        return "\n\n---\n\n".join(sections)
