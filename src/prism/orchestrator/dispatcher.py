"""
Parallel worker dispatch.

Routes tasks to appropriate workers via factory and executes in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from prism.core.logging import log_worker_completion, parse_hook_log_detailed
from prism.workers.factory import create_worker

if TYPE_CHECKING:
    from prism.core.gemini import GeminiExecutor
    from prism.core.retry import RetryExecutor
    from prism.workers.base import AgentResult
    from prism.workers.manager import Task, TaskPlan

logger = logging.getLogger(__name__)


def _collect_hook_data(worker: object, start: float) -> dict | None:
    """Parse hook log from a worker after execution, if available."""
    log_path = getattr(worker, "hook_log_path", None)
    start_time = getattr(worker, "worker_start_time", None)
    if log_path and start_time:
        data = parse_hook_log_detailed(log_path, start_time)
        try:
            os.unlink(log_path)
        except OSError:
            pass
        return data
    return None


class WorkerDispatcher:
    """
    Dispatches tasks to worker agents in parallel.

    Creates workers via factory on-demand based on task agent_type.
    Uses asyncio.gather for parallel execution.
    """

    def __init__(
        self,
        claude_executor: RetryExecutor,
        gemini_executor: GeminiExecutor,
    ) -> None:
        self._claude_executor = claude_executor
        self._gemini_executor = gemini_executor

    async def _execute_task(
        self,
        task: Task,
        timeout: int,
        visible_timeout: int,
        level: int,
        parent_session_id: str | None = None,
    ) -> AgentResult:
        """Execute a single task with its assigned worker. Tracks wall time."""
        from prism.workers.base import AgentResult

        agent_key = task.key or f"{task.agent_type}_0"
        start = time.monotonic()

        try:
            worker = create_worker(
                agent_type=task.agent_type,
                claude_executor=self._claude_executor,
                gemini_executor=self._gemini_executor,
                level=level,
                timeout=timeout,
                visible_timeout=visible_timeout,
            )

            logger.debug(
                "Dispatching task",
                extra={
                    "agent_key": agent_key,
                    "agent_type": task.agent_type,
                    "query_length": len(task.query),
                    "search_level": level,
                },
            )

            result = await worker.execute(
                task.query, timeout_seconds=timeout, parent_session_id=parent_session_id
            )
            wall_time = round(time.monotonic() - start, 1)
            result.metadata["agent_key"] = agent_key
            result.metadata["wall_time_s"] = wall_time

            hook_data = _collect_hook_data(worker, start)
            if hook_data:
                result.metadata["tool_calls"] = hook_data["tool_calls"]
                result.metadata["hook_blocks"] = hook_data["hook_blocks"]
                result.metadata["tool_call_details"] = hook_data["calls"]

            model = getattr(worker, "worker_model", "unknown")
            content_len = len(result.content) if isinstance(result.content, str) else 0
            log_worker_completion(
                worker_type=task.agent_type,
                agent_key=agent_key,
                success=result.success,
                wall_time_s=wall_time,
                model=model,
                response_length=content_len,
                tool_calls=hook_data["tool_calls"] if hook_data else 0,
                hook_blocks=hook_data["hook_blocks"] if hook_data else 0,
                tool_call_details=hook_data["calls"] if hook_data else None,
            )

            return result

        except Exception as e:
            wall_time = round(time.monotonic() - start, 1)
            logger.exception("Task execution failed", extra={"agent_key": agent_key})
            return AgentResult.from_error(
                error=f"Task failed: {e}",
                agent_key=agent_key,
                wall_time_s=wall_time,
            )

    async def dispatch(
        self,
        task_plan: TaskPlan,
        worker_timeout: int,
        visible_timeout: int,
        level: int,
        parent_session_id: str | None = None,
    ) -> list[AgentResult]:
        """Dispatch all tasks in parallel."""
        if not task_plan.tasks:
            logger.warning("Empty task plan received")
            return []

        logger.info(
            "Dispatching tasks",
            extra={
                "task_count": len(task_plan.tasks),
                "timeout": worker_timeout,
                "visible_timeout": visible_timeout,
                "search_level": level,
            },
        )

        results = await asyncio.gather(
            *[
                self._execute_task(
                    task, worker_timeout, visible_timeout, level, parent_session_id
                )
                for task in task_plan.tasks
            ],
            return_exceptions=True,
        )

        from prism.workers.base import AgentResult

        processed: list[AgentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                task = task_plan.tasks[i]
                agent_key = task.key or f"{task.agent_type}_{i}"
                processed.append(
                    AgentResult.from_error(
                        error=f"Dispatch error: {result}",
                        agent_key=agent_key,
                    )
                )
            else:
                processed.append(result)

        success_count = sum(1 for r in processed if r.success)
        logger.info(
            "Dispatch complete",
            extra={"success": success_count, "total": len(processed)},
        )

        return processed
