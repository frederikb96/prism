"""
Parallel worker dispatch.

Routes tasks to appropriate workers and executes in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from prism.core.hooks import build_time_awareness_hooks, build_time_env_vars
from prism.workers import PerplexityAgent, ResearcherAgent, TavilyAgent

if TYPE_CHECKING:
    from prism.workers.base import Agent, AgentResult, ExecutorProtocol
    from prism.workers.manager import Task, TaskPlan

logger = logging.getLogger(__name__)


class WorkerDispatcher:
    """
    Dispatches tasks to worker agents in parallel.

    Creates workers on-demand based on task agent_type.
    Uses asyncio.gather for parallel execution.
    """

    def __init__(
        self,
        executor: ExecutorProtocol,
    ) -> None:
        """
        Initialize dispatcher.

        Args:
            executor: Executor for worker agents (via DI)
        """
        self._executor = executor

    def _create_worker(
        self,
        agent_type: str,
        timeout: int,
        hooks_config: dict[str, Any] | None = None,
        env_vars: tuple[tuple[str, str], ...] | None = None,
    ) -> Agent:
        """
        Create worker agent based on type.

        Args:
            agent_type: Type of agent (researcher, tavily, perplexity)
            timeout: Timeout in seconds for this worker
            hooks_config: Claude hooks configuration for time awareness
            env_vars: Environment variables for hooks

        Returns:
            Configured Agent instance

        Raises:
            ValueError: Unknown agent type
        """
        if agent_type == "researcher":
            return ResearcherAgent(
                executor=self._executor,
                default_timeout=timeout,
                hooks_config=hooks_config,
                env_vars=env_vars,
            )
        elif agent_type == "tavily":
            return TavilyAgent(
                executor=self._executor,
                default_timeout=timeout,
                hooks_config=hooks_config,
                env_vars=env_vars,
            )
        elif agent_type == "perplexity":
            return PerplexityAgent(
                executor=self._executor,
                default_timeout=timeout,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    async def _execute_task(
        self,
        task: Task,
        timeout: int,
        visible_timeout: int,
    ) -> AgentResult:
        """
        Execute a single task with its assigned worker.

        Args:
            task: Task to execute
            timeout: Actual timeout in seconds
            visible_timeout: Timeout the agent sees

        Returns:
            AgentResult from worker
        """
        from prism.workers.base import AgentResult

        try:
            hooks_config = None
            env_vars = None

            if task.agent_type in ("researcher", "tavily"):
                hooks_config = build_time_awareness_hooks()
                env_vars = build_time_env_vars(
                    start_time=time.time(),
                    visible_timeout=visible_timeout,
                )

            worker = self._create_worker(
                task.agent_type,
                timeout,
                hooks_config=hooks_config,
                env_vars=env_vars,
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
    ) -> list[AgentResult]:
        """
        Dispatch all tasks in parallel.

        Args:
            task_plan: Plan containing tasks to dispatch
            worker_timeout: Actual timeout per worker in seconds
            visible_timeout: Timeout the agents see

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
            },
        )

        results = await asyncio.gather(
            *[
                self._execute_task(task, worker_timeout, visible_timeout)
                for task in task_plan.tasks
            ],
            return_exceptions=True,
        )

        # Convert exceptions to error results
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
