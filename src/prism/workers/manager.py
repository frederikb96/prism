"""
Manager agent for search session lifecycle.

Handles planning, synthesis, and follow-ups using RetryExecutor
with schema validation for structured JSON output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from prism.core.response import ExecutionRequest
from prism.prompts import get_registry
from prism.workers.base import AgentResult

if TYPE_CHECKING:
    from prism.core.retry import RetryExecutor

logger = logging.getLogger(__name__)

# Brief descriptions for the user-prompt agent section.
# Full descriptions live in system.md; these are compact summaries with slot info.
_AGENT_DESCRIPTIONS: dict[str, str] = {
    "claude_search": "Thorough structured researcher with link-following and WebFetch.",
    "gemini_search": "Aggressive parallel searcher with Google grounding search.",
    "tavily_search": "Real original content extraction and multi-source validation.",
    "perplexity_search": "Quick factual lookups and efficient wide-range info gathering.",
}


@dataclass
class TaskPlan:
    """Structured output from manager agent."""

    tasks: list[Task] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPlan:
        """Create TaskPlan from legacy array format."""
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            reasoning=data.get("reasoning", ""),
        )

    @classmethod
    def from_keyed_dict(cls, data: dict[str, Any]) -> TaskPlan:
        """Parse fixed-dict format (claude_search_1, gemini_search_2, etc.) into TaskPlan."""
        tasks = []
        reasoning = data.get("reasoning", "")
        for key, value in data.items():
            if key == "reasoning":
                continue
            if not isinstance(value, dict):
                continue
            # key format: "claude_search_1" -> agent_type="claude_search"
            agent_type, _ = key.rsplit("_", 1)
            tasks.append(Task(
                query=value.get("query", ""),
                agent_type=agent_type,
                context=value.get("context", ""),
            ))
        return cls(tasks=tasks, reasoning=reasoning)


@dataclass
class Task:
    """A single task for a worker agent."""

    query: str
    agent_type: str = "claude_search"
    context: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create Task from dictionary."""
        return cls(
            query=data.get("query", ""),
            agent_type=data.get("agent_type", "claude_search"),
            context=data.get("context", ""),
        )


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
        session_id: str | None = None,
    ) -> None:
        self._executor = executor
        self._model = model
        self._agent_allocation = agent_allocation
        self._session_id = session_id
        self._registry = get_registry()
        self._task_schema = self._build_task_schema(agent_allocation)
        self._response_schema = self._load_response_schema()
        self._system_prompt = self._registry.get_content("search_manager/system")

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def plan(self, query: str, timeout_seconds: int | None = None) -> AgentResult:
        """Template A: Create initial task plan. Sets self._session_id."""
        user_prompt = self._render_search_prompt("user_search", query)

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

        result = await self._executor.execute(request, schema=self._task_schema)

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

    async def synthesize(
        self, query: str, results: list[AgentResult], timeout_seconds: int | None = None
    ) -> AgentResult:
        """Template B: Synthesize results via --resume. Requires self._session_id."""
        if not self._session_id:
            return AgentResult.from_error(error="No session_id for synthesis --resume")

        user_prompt = self._render_synthesis_prompt(query, results)

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

        result = await self._executor.execute(request, schema=self._response_schema)

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
        """Template C: Follow-up search via --resume. Optionally different allocation."""
        if not self._session_id:
            return AgentResult.from_error(error="No session_id for follow-up --resume")

        alloc = agent_allocation or self._agent_allocation
        schema = self._build_task_schema(alloc) if agent_allocation else self._task_schema

        user_prompt = self._render_search_prompt("user_follow_up_search", follow_up)

        request = ExecutionRequest(
            prompt=user_prompt,
            resume_session=self._session_id,
            timeout_seconds=timeout_seconds,
            json_schema=schema,
        )

        result = await self._executor.execute(request, schema=schema)

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
                content=asdict(task_plan),
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
        """Template D: Conversational follow-up via --resume."""
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

        result = await self._executor.execute(request, schema=self._response_schema)

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
        """Dynamically generate JSON schema from agent allocation."""
        required: list[str] = ["reasoning"]
        properties: dict[str, Any] = {
            "reasoning": {"type": "string", "description": "Brief strategy explanation"},
        }
        for agent_type, count in sorted(allocation.items()):
            if count <= 0:
                continue
            for i in range(1, count + 1):
                key = f"{agent_type}_{i}"
                required.append(key)
                properties[key] = {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"Search query for {agent_type}",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for the agent",
                        },
                    },
                }
        return {
            "type": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": False,
        }

    @staticmethod
    def _render_agent_section(allocation: dict[str, int]) -> str:
        """Render agent section for user prompts."""
        lines: list[str] = []
        for agent_type, count in sorted(allocation.items()):
            if count <= 0:
                continue
            slots = ", ".join(f"{agent_type}_{i}" for i in range(1, count + 1))
            desc = _AGENT_DESCRIPTIONS.get(agent_type, "")
            slot_word = "slot" if count == 1 else "slots"
            lines.append(f"- **{agent_type}** ({count} {slot_word}: {slots}): {desc}")
        return "\n".join(lines)

    def _render_search_prompt(self, template_name: str, query_or_follow_up: str) -> str:
        """Render Template A or C with agent section."""
        template = self._registry.get_content(f"search_manager/{template_name}")
        if template is None:
            raise RuntimeError(f"Template not found: search_manager/{template_name}")

        agent_section = self._render_agent_section(self._agent_allocation)
        prompt = template.replace("{query}", query_or_follow_up)
        prompt = prompt.replace("{follow_up}", query_or_follow_up)
        prompt = prompt.replace("{agent_section}", agent_section)
        return prompt

    def _render_synthesis_prompt(self, query: str, results: list[AgentResult]) -> str:
        """Render Template B with worker results."""
        template = self._registry.get_content("search_manager/user_synthesis")
        if template is None:
            raise RuntimeError("Template not found: search_manager/user_synthesis")

        worker_results = self._format_worker_results(results)
        prompt = template.replace("{query}", query)
        prompt = prompt.replace("{worker_results}", worker_results)
        return prompt

    @staticmethod
    def _format_worker_results(results: list[AgentResult]) -> str:
        """Format worker results as structured text for synthesis prompt."""
        sections: list[str] = []
        for result in results:
            if result.success:
                content = result.content if isinstance(result.content, str) else str(result.content)
                agent_type = result.metadata.get("agent_type", "unknown")
                sections.append(f"### {agent_type}\n{content}")
        return "\n\n".join(sections)

    def _load_response_schema(self) -> dict[str, Any]:
        """Load response schema from file."""
        schema = self._registry.get_schema("search_manager/response_schema")
        if schema is None:
            raise RuntimeError("Response schema not found")
        return schema

    def _parse_task_plan(self, raw_output: str) -> TaskPlan:
        """Parse raw output into TaskPlan via keyed-dict format."""
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
