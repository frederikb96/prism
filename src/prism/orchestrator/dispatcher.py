"""
Parallel worker dispatch.

Routes tasks to appropriate workers via factory and executes in parallel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from prism.workers.factory import create_worker

if TYPE_CHECKING:
    from prism.core.gemini import GeminiExecutor
    from prism.core.retry import RetryExecutor
    from prism.workers.base import AgentResult
    from prism.workers.manager import Task, TaskPlan

logger = logging.getLogger(__name__)


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
    ) -> AgentResult:
        """
        Execute a single task with its assigned worker.

        Args:
            task: Task to execute
            timeout: Actual timeout in seconds
            visible_timeout: Timeout the agent sees
            level: Search level for model selection
        """
        from prism.workers.base import AgentResult

        try:
            worker = create_worker(
                agent_type=task.agent_type,
                claude_executor=self._claude_executor,
                gemini_executor=self._gemini_executor,
                level=level,
                timeout=timeout,
                visible_timeout=visible_timeout,
            )

            prompt = task.query
            if task.context:
                prompt = f"{task.context}\n\n{task.query}"

            logger.debug(
                "Dispatching task",
                extra={
                    "agent_type": task.agent_type,
                    "query_length": len(task.query),
                    "visible_timeout": visible_timeout,
                    "level": level,
                },
            )

            return await worker.execute(prompt, timeout_seconds=timeout)

        except Exception as e:
            logger.exception("Task execution failed", extra={"agent_type": task.agent_type})
            return AgentResult.from_error(
                error=f"Task failed: {e}",
                agent_type=task.agent_type,
            )

    async def dispatch(
        self,
        task_plan: TaskPlan,
        worker_timeout: int,
        visible_timeout: int,
        level: int,
    ) -> list[AgentResult]:
        """
        Dispatch all tasks in parallel.

        Args:
            task_plan: Plan containing tasks to dispatch
            worker_timeout: Actual timeout per worker in seconds
            visible_timeout: Timeout the agents see
            level: Search level for model selection

        Returns:
            List of AgentResults (one per task)
        """
        if not task_plan.tasks:
            logger.warning("Empty task plan received")
            return []

        logger.info(
            "Dispatching tasks",
            extra={
                "task_count": len(task_plan.tasks),
                "timeout": worker_timeout,
                "visible_timeout": visible_timeout,
                "level": level,
            },
        )

        results = await asyncio.gather(
            *[
                self._execute_task(task, worker_timeout, visible_timeout, level)
                for task in task_plan.tasks
            ],
            return_exceptions=True,
        )

        from prism.workers.base import AgentResult

        processed: list[AgentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                processed.append(
                    AgentResult.from_error(
                        error=f"Dispatch error: {result}",
                        task_index=i,
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
