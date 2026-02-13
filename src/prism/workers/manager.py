"""
Manager agent for search session lifecycle.

Handles planning, synthesis, and follow-ups using RetryExecutor
with schema validation for structured JSON output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prism.core.logging import log_prompt
from prism.core.response import ExecutionRequest
from prism.prompts import get_registry

if TYPE_CHECKING:
    from prism.core.retry import RetryExecutor
    from prism.workers.base import AgentResult

logger = logging.getLogger(__name__)

FOLLOW_UP_SEARCH_PREAMBLE = (
    "This is a follow-up search issued by the user. "
    "Below you see again the search prompt with the new follow-up query.\n\n"
)


@dataclass
class TaskPlan:
    """Structured output from manager agent."""

    tasks: list[Task] = field(default_factory=list)

    @classmethod
    def from_keyed_dict(cls, data: dict[str, Any]) -> TaskPlan:
        """Parse task plan from JSON output.

        Supports two formats:
        - New: {"claude_search_1": "query string", ...}
        - Legacy: {"claude_search_1": {"query": "...", "context": "..."}, ...}
        """
        tasks = []
        for key, value in data.items():
            parts = key.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue

            agent_type = parts[0]
            if isinstance(value, str):
                tasks.append(Task(query=value, agent_type=agent_type, key=key))
            elif isinstance(value, dict):
                tasks.append(Task(
                    query=value.get("query", ""),
                    agent_type=agent_type,
                    key=key,
                ))
        return cls(tasks=tasks)


@dataclass
class Task:
    """A single task for a worker agent."""

    query: str
    agent_type: str = "claude_search"
    key: str = ""


class ManagerAgent:
    """
    Manages multi-turn search session: planning, synthesis, and follow-ups.

    Tracks Claude CLI session_id internally so that plan -> synthesize ->
    follow-up all happen in the same conversational context via --resume.
    """

    def __init__(
        self,
        executor: RetryExecutor,
        model: str,
        agent_allocation: dict[str, int],
        level: int,
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        self._executor = executor
        self._model = model
        self._agent_allocation = agent_allocation
        self._level = level
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._registry = get_registry()
        self._task_schema = self._build_task_schema(agent_allocation)
        self._response_schema = self._load_response_schema()
        self._system_prompt = self._registry.get_content("search_manager/system")

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def plan(self, query: str, timeout_seconds: int | None = None) -> AgentResult:
        """Create initial task plan. Sets self._session_id."""
        from prism.workers.base import AgentResult

        user_prompt = self._render_search_prompt(query)

        log_prompt(
            prompt_type="search_planning",
            prompt=user_prompt,
            level=self._level,
            session_id=self._session_id,
        )

        request = ExecutionRequest(
            prompt=user_prompt,
            model=self._model,
            timeout_seconds=timeout_seconds,
            json_schema=self._task_schema,
            system_prompt=self._system_prompt,
        )

        logger.debug(
            "Manager creating task plan",
            extra={"query_length": len(query), "timeout": timeout_seconds},
        )

        result = await self._executor.execute(
            request, schema=self._task_schema, parent_session_id=self._parent_session_id
        )

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Manager planning failed",
                raw_output=result.output,
            )

        if result.session_id:
            self._session_id = result.session_id

        try:
            task_plan = self._parse_task_plan(result.output)
            return AgentResult.from_success(
                content={"tasks": [
                    {"query": t.query, "agent_type": t.agent_type, "key": t.key}
                    for t in task_plan.tasks
                ]},
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

    async def synthesize(
        self, results: list[AgentResult], timeout_seconds: int | None = None
    ) -> AgentResult:
        """Synthesize results via --resume. Requires self._session_id."""
        from prism.workers.base import AgentResult

        if not self._session_id:
            return AgentResult.from_error(error="No session_id for synthesis --resume")

        user_prompt = self._render_synthesis_prompt(results)

        log_prompt(
            prompt_type="synthesis",
            prompt=user_prompt,
            level=self._level,
            session_id=self._session_id,
        )

        request = ExecutionRequest(
            prompt=user_prompt,
            resume_session=self._session_id,
            timeout_seconds=timeout_seconds,
            json_schema=self._response_schema,
        )

        logger.debug(
            "Manager synthesizing results",
            extra={"result_count": len(results), "resume": self._session_id},
        )

        result = await self._executor.execute(
            request, schema=self._response_schema, parent_session_id=self._parent_session_id
        )

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Synthesis failed",
                raw_output=result.output,
            )

        if result.session_id:
            self._session_id = result.session_id

        response_text = self._parse_response(result.output)

        return AgentResult.from_success(
            content=response_text,
            raw_output=result.output,
            session_id=result.session_id,
            synthesized_count=sum(1 for r in results if r.success),
        )

    async def follow_up_search(
        self,
        follow_up: str,
        agent_allocation: dict[str, int] | None = None,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """Follow-up search via --resume. Reuses search prompt with preamble."""
        from prism.workers.base import AgentResult

        if not self._session_id:
            return AgentResult.from_error(error="No session_id for follow-up --resume")

        alloc = agent_allocation or self._agent_allocation
        schema = self._build_task_schema(alloc) if agent_allocation else self._task_schema

        user_prompt = FOLLOW_UP_SEARCH_PREAMBLE + self._render_search_prompt(follow_up)

        request = ExecutionRequest(
            prompt=user_prompt,
            resume_session=self._session_id,
            timeout_seconds=timeout_seconds,
            json_schema=schema,
        )

        result = await self._executor.execute(
            request, schema=schema, parent_session_id=self._parent_session_id
        )

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Follow-up search planning failed",
                raw_output=result.output,
            )

        if result.session_id:
            self._session_id = result.session_id

        try:
            task_plan = self._parse_task_plan(result.output)
            return AgentResult.from_success(
                content={"tasks": [
                    {"query": t.query, "agent_type": t.agent_type, "key": t.key}
                    for t in task_plan.tasks
                ]},
                raw_output=result.output,
                session_id=result.session_id,
                task_count=len(task_plan.tasks),
            )
        except Exception as e:
            logger.exception("Failed to parse follow-up task plan")
            return AgentResult.from_error(
                error=f"Failed to parse follow-up task plan: {e}",
                raw_output=result.output,
            )

    async def follow_up_chat(
        self, follow_up: str, timeout_seconds: int | None = None
    ) -> AgentResult:
        """Conversational follow-up via --resume."""
        from prism.workers.base import AgentResult

        if not self._session_id:
            return AgentResult.from_error(error="No session_id for follow-up --resume")

        template = self._registry.get_content("search_manager/user_follow_up_chat")
        if template is None:
            return AgentResult.from_error(error="Follow-up chat template not found")

        user_prompt = template.replace("{follow_up}", follow_up)

        request = ExecutionRequest(
            prompt=user_prompt,
            resume_session=self._session_id,
            timeout_seconds=timeout_seconds,
            json_schema=self._response_schema,
        )

        result = await self._executor.execute(
            request, schema=self._response_schema, parent_session_id=self._parent_session_id
        )

        if not result.success:
            return AgentResult.from_error(
                error=result.error_message or "Follow-up chat failed",
                raw_output=result.output,
            )

        if result.session_id:
            self._session_id = result.session_id

        response_text = self._parse_response(result.output)

        return AgentResult.from_success(
            content=response_text,
            raw_output=result.output,
            session_id=result.session_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task_schema(allocation: dict[str, int]) -> dict[str, Any]:
        """Dynamically generate JSON schema from agent allocation.

        Simple format: each key is agent_type_N, value is the search prompt string.
        """
        required: list[str] = []
        properties: dict[str, Any] = {}
        for agent_type, count in sorted(allocation.items()):
            if count <= 0:
                continue
            for i in range(1, count + 1):
                key = f"{agent_type}_{i}"
                required.append(key)
                properties[key] = {
                    "type": "string",
                    "description": f"Search prompt for {agent_type}",
                }
        return {
            "type": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": False,
        }

    @staticmethod
    def _build_schema_example(allocation: dict[str, int]) -> str:
        """Build a concrete JSON example for the task plan output format."""
        example: dict[str, str] = {}
        for agent_type, count in sorted(allocation.items()):
            if count <= 0:
                continue
            for i in range(1, count + 1):
                example[f"{agent_type}_{i}"] = "..."
        return json.dumps(example, indent=2)

    def _render_agent_section(self) -> str:
        """Render agent descriptions from .md files, conditional on allocation."""
        sections: list[str] = []
        for agent_type, count in sorted(self._agent_allocation.items()):
            if count <= 0:
                continue
            desc = self._registry.get_content(f"search_manager/agents/{agent_type}")
            if desc is None:
                desc = f"Search agent: {agent_type}"
            else:
                desc = desc.strip()
            slots = ", ".join(f"{agent_type}_{i}" for i in range(1, count + 1))
            slot_word = "slot" if count == 1 else "slots"
            sections.append(f"### {agent_type} ({count} {slot_word}: {slots})\n\n{desc}")
        return "\n\n".join(sections)

    def _render_search_prompt(self, query: str) -> str:
        """Render search planning prompt with agents, level guidance, and schema example."""
        template = self._registry.get_content("search_manager/user_search")
        if template is None:
            raise RuntimeError("Template not found: search_manager/user_search")

        agent_section = self._render_agent_section()
        schema_example = self._build_schema_example(self._agent_allocation)
        agent_count = sum(c for c in self._agent_allocation.values() if c > 0)

        level_guidance = self._registry.get_content(
            f"search_manager/levels/search_l{self._level}"
        ) or ""
        level_guidance = level_guidance.strip().replace(
            "{agent_count}", str(agent_count)
        )

        prompt = template.replace("{query}", query)
        prompt = prompt.replace("{agent_section}", agent_section)
        prompt = prompt.replace("{level_guidance}", level_guidance)
        prompt = prompt.replace("{schema_example}", schema_example)
        return prompt

    def _render_synthesis_prompt(self, results: list[AgentResult]) -> str:
        """Render synthesis prompt with worker results and level guidance."""
        template = self._registry.get_content("search_manager/user_synthesis")
        if template is None:
            raise RuntimeError("Template not found: search_manager/user_synthesis")

        worker_results = self._format_worker_results(results)
        level_guidance = self._registry.get_content(
            f"search_manager/levels/synthesis_l{self._level}"
        ) or ""

        prompt = template.replace("{worker_results}", worker_results)
        prompt = prompt.replace("{level_guidance}", level_guidance.strip())
        return prompt

    @staticmethod
    def _format_worker_results(results: list[AgentResult]) -> str:
        """Format worker results with timing info for synthesis prompt."""
        sections: list[str] = []
        for result in results:
            agent_key = result.metadata.get("agent_key", "unknown")
            wall_time = result.metadata.get("wall_time_s")
            timed_out = result.metadata.get("timed_out", False)

            timing = ""
            if timed_out:
                timing = f" (TIMED OUT after {wall_time:.1f}s)" if wall_time else " (TIMED OUT)"
            elif wall_time is not None:
                timing = f" ({wall_time:.1f}s)"

            if result.success:
                content = result.content if isinstance(result.content, str) else str(result.content)
                sections.append(f"### {agent_key}{timing}\n\n{content}")
            else:
                error = result.error or "failed"
                sections.append(f"### {agent_key}{timing}\n\nFAILED: {error}")
        return "\n\n".join(sections)

    def _load_response_schema(self) -> dict[str, Any]:
        """Load response schema from file."""
        schema = self._registry.get_schema("search_manager/response_schema")
        if schema is None:
            raise RuntimeError("Response schema not found")
        return schema

    def _parse_task_plan(self, raw_output: str) -> TaskPlan:
        """Parse raw output into TaskPlan."""
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

        return TaskPlan.from_keyed_dict(data)

    def _parse_response(self, raw_output: str) -> str:
        """Extract response text from {"response": "..."} format."""
        data: Any = json.loads(raw_output)
        if isinstance(data, dict):
            if "structured_output" in data:
                data = data["structured_output"]
            elif "result" in data:
                result_str = data["result"]
                data = json.loads(result_str) if isinstance(result_str, str) else result_str
        if isinstance(data, dict):
            return str(data.get("response", data))
        return str(data)
