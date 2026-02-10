"""
Manager agent for search planning.

Uses RetryExecutor with schema validation to produce
structured task plans for worker dispatch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from prism.config import get_config
from prism.core.response import ExecutionRequest
from prism.prompts import PromptRegistry
from prism.workers.base import Agent, AgentResult

if TYPE_CHECKING:
    from prism.core.retry import RetryExecutor

logger = logging.getLogger(__name__)


@dataclass
class TaskPlan:
    """
    Structured output from manager agent.

    Contains list of tasks to dispatch to workers.
    """

    tasks: list[Task] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPlan:
        """Create TaskPlan from dictionary."""
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class Task:
    """
    A single task for a worker agent.

    Attributes:
        query: The search query for this task
        agent_type: Which agent to use (researcher, tavily, perplexity)
        context: Additional context for the agent
    """

    query: str
    agent_type: str = "researcher"
    context: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create Task from dictionary."""
        return cls(
            query=data.get("query", ""),
            agent_type=data.get("agent_type", "researcher"),
            context=data.get("context", ""),
        )


_registry = PromptRegistry()


def _get_task_plan_schema() -> dict[str, Any]:
    """Load task plan schema from prompts/search_manager/task_schema.json."""
    schema = _registry.get_schema("search_manager/task_schema")
    if schema is None:
        raise RuntimeError("Task plan schema not found at prompts/search_manager/task_schema.json")
    return schema


class ManagerAgent(Agent):
    """
    Search manager that creates task plans for workers.

    Uses RetryExecutor for schema-validated JSON output.
    Produces TaskPlan with list of tasks to dispatch.
    """

    def __init__(
        self,
        executor: RetryExecutor,
        system_prompt: str | None = None,
        level_prompt: str | None = None,
        model: str | None = None,
        default_timeout: int | None = None,
    ) -> None:
        """
        Initialize manager agent.

        Args:
            executor: RetryExecutor for schema-validated execution
            system_prompt: Custom system prompt
            level_prompt: Level-specific instructions (L1/L2/L3)
            model: Claude model to use
            default_timeout: Default timeout in seconds
        """
        config = get_config()
        self._executor = executor
        self._system_prompt = system_prompt
        self._level_prompt = level_prompt
        self._model = model or config.workers.manager.model
        self._default_timeout = default_timeout or config.levels[1].manager_timeout_seconds
        self._schema = _get_task_plan_schema()

    @property
    def agent_type(self) -> str:
        return "manager"

    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """
        Create a task plan for the given query.

        Args:
            prompt: User's search query
            timeout_seconds: Optional timeout override
        """
        timeout = timeout_seconds or self._default_timeout

        full_prompt = self._build_planning_prompt(prompt)

        request = ExecutionRequest(
            prompt=full_prompt,
            model=self._model,
            timeout_seconds=timeout,
            json_schema=self._schema,
            system_prompt=self._system_prompt,
        )

        logger.debug(
            "Manager creating task plan",
            extra={"query_length": len(prompt), "timeout": timeout},
        )

        result = await self._executor.execute(
            request,
            schema=self._schema,
        )

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Manager planning failed",
                raw_output=result.output,
            )

        try:
            task_plan = self._parse_task_plan(result.output)
            return AgentResult.from_success(
                content=asdict(task_plan),
                raw_output=result.output,
                session_id=result.session_id,
                task_count=len(task_plan.tasks),
            )
        except Exception as e:
            logger.exception("Failed to parse task plan")
            return AgentResult.from_error(
                error=f"Failed to parse task plan: {e}",
                raw_output=result.output,
            )

    def _build_planning_prompt(self, query: str) -> str:
        """
        Build the planning prompt with level-specific instructions.

        Args:
            query: User's search query
        """
        parts = [f"Create a search plan for the following query:\n\n{query}"]

        if self._level_prompt:
            parts.append(f"\n\n{self._level_prompt}")

        return "\n".join(parts)

    def _parse_task_plan(self, raw_output: str) -> TaskPlan:
        """
        Parse raw output into TaskPlan.

        Args:
            raw_output: JSON output from Claude CLI
        """
        data = json.loads(raw_output)

        if isinstance(data, dict):
            if "structured_output" in data:
                data = data["structured_output"]
            elif "result" in data:
                result_str = data["result"]
                if isinstance(result_str, str):
                    data = json.loads(result_str)
                else:
                    data = result_str

        return TaskPlan.from_dict(data)
