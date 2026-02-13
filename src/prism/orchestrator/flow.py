"""
Full search flow coordination.

Orchestrates the complete search process:
- Level 0: Multi-provider via factory (default, mix, or explicit list)
- Level 1-3: Manager -> Dispatch workers -> Synthesize
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from prism.config import get_config
from prism.core.logging import log_manager_phase, log_worker_completion, parse_hook_log_detailed
from prism.database import SessionStatus
from prism.workers import ManagerAgent
from prism.workers.factory import VALID_AGENT_TYPES, create_worker

if TYPE_CHECKING:
    from prism.core.gemini import GeminiExecutor
    from prism.core.retry import RetryExecutor
    from prism.core.session import SessionRegistry
    from prism.database import SearchSessionRepository
    from prism.orchestrator.dispatcher import WorkerDispatcher

logger = logging.getLogger(__name__)

ALL_PROVIDERS = ["claude_search", "tavily_search", "perplexity_search", "gemini_search"]


def _build_worker_details(worker_results: list) -> list[dict[str, Any]]:
    """Build per-worker detail list from dispatch results."""
    details: list[dict[str, Any]] = []
    for r in worker_results:
        detail: dict[str, Any] = {
            "type": r.metadata.get("agent_type", "unknown"),
            "key": r.metadata.get("agent_key"),
            "success": r.success,
            "wall_time_s": r.metadata.get("wall_time_s"),
        }
        if r.error:
            detail["error"] = r.error
        details.append(detail)
    return details


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
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class SearchFlow:
    """
    Coordinates full search flow.

    Level 0: Multi-provider worker execution via factory
    Level 1-3: Manager planning -> Worker dispatch -> Result synthesis
    """

    def __init__(
        self,
        retry_executor: RetryExecutor,
        gemini_executor: GeminiExecutor,
        dispatcher: WorkerDispatcher,
        session_registry: SessionRegistry,
        session_repository: SearchSessionRepository,
    ) -> None:
        self._retry_executor = retry_executor
        self._gemini_executor = gemini_executor
        self._dispatcher = dispatcher
        self._session_registry = session_registry
        self._session_repository = session_repository

    async def execute_search(
        self,
        query: str,
        level: int = 0,
        providers: list[str] | None = None,
        *,
        user_id: str,
    ) -> SearchResult:
        """
        Execute a search at the specified level.

        Args:
            query: Search query
            level: Search depth (0-3)
            providers: L0 provider selection (None=config default, ["mix"]=all 4)
            user_id: Requesting user's identifier
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
            extra={"query_length": len(query), "search_level": level, "session_id": session_id},
        )

        await self._session_repository.create(
            user_id=user_id,
            query=query,
            prompt=query,
            level=level,
            session_id=session_uuid,
        )

        await self._session_repository.update(
            session_uuid,
            user_id,
            status=SessionStatus.RUNNING,
        )

        try:
            await self._session_registry.register(session_id, user_id=user_id)

            if level == 0:
                result = await self._execute_level_0(query, session_id, providers)
            else:
                result = await self._execute_level_1_3(query, level, session_id)

            duration_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = datetime.now(timezone.utc)

            await self._session_repository.update(
                session_uuid,
                user_id,
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
                user_id,
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
        """Extract a short summary from search result."""
        if not result.success or not result.content:
            return None

        content = result.content
        if len(content) <= 200:
            return content

        first_para_end = content.find("\n\n")
        if first_para_end > 0 and first_para_end <= 300:
            return content[:first_para_end]

        return content[:200] + "..."

    def _resolve_providers(self, providers: list[str] | None) -> list[str]:
        """
        Resolve provider list from input.

        None -> config default. ["mix"] -> all 4. Otherwise pass through.
        """
        config = get_config()

        if providers is None:
            return list(config.level0.default_providers)

        if "mix" in providers:
            return list(ALL_PROVIDERS)

        return list(providers)

    async def _execute_level_0(
        self,
        query: str,
        session_id: str,
        providers: list[str] | None = None,
    ) -> SearchResult:
        """
        Execute Level 0 search via factory workers.

        Single provider returns content directly.
        Multiple providers return sectioned results.
        """
        config = get_config()
        level_config = config.levels[0]
        resolved = self._resolve_providers(providers)

        timeout = level_config.worker_timeout_seconds
        visible_timeout = level_config.worker_visible_timeout

        workers = []
        for agent_type in resolved:
            if agent_type not in VALID_AGENT_TYPES:
                return SearchResult(
                    success=False,
                    content="",
                    session_id=session_id,
                    level=0,
                    query=query,
                    error=f"Unknown provider: {agent_type}",
                )
            workers.append(
                create_worker(
                    agent_type=agent_type,
                    claude_executor=self._retry_executor,
                    gemini_executor=self._gemini_executor,
                    level=0,
                    timeout=timeout,
                    visible_timeout=visible_timeout,
                )
            )

        worker_starts = [time.monotonic()] * len(workers)
        results = await asyncio.gather(
            *[
                w.execute(query, timeout_seconds=timeout, parent_session_id=session_id)
                for w in workers
            ],
            return_exceptions=True,
        )

        successful_sections: list[tuple[str, str]] = []
        all_errors: list[str] = []
        worker_details: list[dict[str, Any]] = []

        for i, result in enumerate(results):
            provider_name = resolved[i]
            wall_time = round(time.monotonic() - worker_starts[i], 1)

            if isinstance(result, BaseException):
                all_errors.append(f"{provider_name}: {result}")
                worker_details.append({
                    "type": provider_name,
                    "success": False,
                    "wall_time_s": wall_time,
                    "error": str(result),
                })
                continue

            # Parse hook log and emit worker completion
            worker = workers[i]
            hook_data = None
            log_path = getattr(worker, "hook_log_path", None)
            w_start = getattr(worker, "worker_start_time", None)
            if log_path and w_start:
                hook_data = parse_hook_log_detailed(log_path, w_start)
                try:
                    os.unlink(log_path)
                except OSError:
                    pass

            model = getattr(worker, "worker_model", "unknown")
            content_len = len(result.content) if isinstance(result.content, str) else 0
            log_worker_completion(
                worker_type=provider_name,
                agent_key=f"{provider_name}_1",
                success=result.success,
                wall_time_s=wall_time,
                model=model,
                response_length=content_len,
                tool_calls=hook_data["tool_calls"] if hook_data else 0,
                hook_blocks=hook_data["hook_blocks"] if hook_data else 0,
                tool_call_details=hook_data["calls"] if hook_data else None,
            )

            detail: dict[str, Any] = {
                "type": provider_name,
                "success": result.success,
                "wall_time_s": wall_time,
            }

            if result.success:
                content = result.content if isinstance(result.content, str) else str(result.content)
                successful_sections.append((provider_name, content))
            else:
                error_msg = result.error or "failed"
                all_errors.append(f"{provider_name}: {error_msg}")
                detail["error"] = error_msg

            worker_details.append(detail)

        if not successful_sections:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=0,
                query=query,
                error=f"All providers failed: {'; '.join(all_errors)}",
                metadata={"workers": worker_details},
            )

        if len(resolved) == 1:
            content = successful_sections[0][1]
        else:
            parts = []
            for provider_name, content_text in successful_sections:
                parts.append(f"## {provider_name}\n\n{content_text}")
            content = "\n\n---\n\n".join(parts)

        return SearchResult(
            success=True,
            content=content,
            session_id=session_id,
            level=0,
            query=query,
            metadata={"workers": worker_details},
        )

    async def _execute_level_1_3(
        self,
        query: str,
        level: int,
        session_id: str,
    ) -> SearchResult:
        """
        Execute Level 1-3 search (orchestrated).

        Flow: Manager.plan() -> Dispatch -> Manager.synthesize() (--resume)
        """
        config = get_config()
        level_config = config.levels[level]
        manager_model_cfg = config.models.session_manager[level]

        if not level_config.agent_allocation:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error=f"No agent_allocation configured for level {level}",
            )

        manager = ManagerAgent(
            executor=self._retry_executor,
            model=manager_model_cfg.model,
            agent_allocation=level_config.agent_allocation,
            level=level,
            parent_session_id=session_id,
        )

        logger.debug("Manager creating task plan", extra={"search_level": level})

        plan_start = time.monotonic()
        plan_result = await manager.plan(query, timeout_seconds=None)
        plan_wall_time = round(time.monotonic() - plan_start, 2)

        log_manager_phase(
            phase="planning",
            level=level,
            wall_time_s=plan_wall_time,
            session_id=manager.session_id,
        )

        if not plan_result.success:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error=f"Manager planning failed: {plan_result.error}",
            )

        from prism.workers.manager import Task, TaskPlan

        if isinstance(plan_result.content, dict) and "tasks" in plan_result.content:
            tasks = [
                Task(query=t["query"], agent_type=t["agent_type"], key=t.get("key", ""))
                for t in plan_result.content["tasks"]
            ]
            task_plan = TaskPlan(tasks=tasks)
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
            extra={"task_count": len(task_plan.tasks)},
        )

        worker_results = await self._dispatcher.dispatch(
            task_plan=task_plan,
            worker_timeout=level_config.worker_timeout_seconds,
            visible_timeout=level_config.worker_visible_timeout,
            level=level,
            parent_session_id=session_id,
        )

        worker_details = _build_worker_details(worker_results)
        successful_count = sum(1 for w in worker_details if w["success"])

        if successful_count == 0:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                level=level,
                query=query,
                error="All workers failed",
                metadata={
                    "task_count": len(task_plan.tasks),
                    "workers": worker_details,
                },
            )

        synth_start = time.monotonic()
        synthesis_result = await manager.synthesize(
            results=worker_results,
            timeout_seconds=None,
        )
        synth_wall_time = round(time.monotonic() - synth_start, 2)

        log_manager_phase(
            phase="synthesis",
            level=level,
            wall_time_s=synth_wall_time,
            session_id=manager.session_id,
        )

        result_metadata: dict[str, Any] = {
            "task_count": len(task_plan.tasks),
            "workers": worker_details,
        }

        if not synthesis_result.success:
            content = self._fallback_combine(worker_results)
            result_metadata["fallback"] = True
            return SearchResult(
                success=True,
                content=content,
                session_id=session_id,
                level=level,
                query=query,
                metadata=result_metadata,
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
            metadata=result_metadata,
        )

    async def resume_session(
        self,
        claude_session_id: str,
        follow_up: str,
        session_id: str,
    ) -> SearchResult:
        """
        Resume a previous L1-3 session with a follow-up query.

        Uses ManagerAgent.follow_up_chat() which handles --resume
        internally. Model and system prompt are inherited from the session.

        Args:
            claude_session_id: Claude CLI session ID to resume
            follow_up: Follow-up question or instruction
            session_id: Our DB session ID for tracking

        Returns:
            SearchResult with the follow-up response
        """
        logger.info(
            "Resuming session",
            extra={"claude_session_id": claude_session_id, "session_id": session_id},
        )

        manager = ManagerAgent(
            executor=self._retry_executor,
            model="sonnet",
            agent_allocation={},
            level=0,
            session_id=claude_session_id,
        )

        chat_result = await manager.follow_up_chat(follow_up, timeout_seconds=None)

        if not chat_result.success:
            return SearchResult(
                success=False,
                content="",
                session_id=session_id,
                query=follow_up,
                error=chat_result.error or "Resume failed",
            )

        content = (
            chat_result.content
            if isinstance(chat_result.content, str)
            else str(chat_result.content)
        )

        return SearchResult(
            success=True,
            content=content,
            session_id=session_id,
            query=follow_up,
            metadata={"resumed_from": claude_session_id},
        )

    def _fallback_combine(self, results: list) -> str:
        """Fallback: concatenate successful results."""
        sections = []
        for i, result in enumerate(results, 1):
            if result.success:
                content = result.content if isinstance(result.content, str) else str(result.content)
                sections.append(f"## Source {i}\n\n{content}")

        return "\n\n---\n\n".join(sections)
