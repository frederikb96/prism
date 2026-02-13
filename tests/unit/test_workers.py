"""Tests for workers.

Tests worker agent patterns, all 4 worker types, factory,
manager, and prompt composition.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from prism.config import RetryConfig
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.retry import RetryExecutor
from prism.prompts import PromptRegistry
from prism.workers.base import Agent, ExecutorProtocol
from prism.workers.claude_search import ClaudeSearchAgent
from prism.workers.factory import VALID_AGENT_TYPES, create_worker
from prism.workers.gemini_search import GeminiSearchAgent
from prism.workers.manager import ManagerAgent, Task, TaskPlan
from prism.workers.perplexity_search import PerplexitySearchAgent
from prism.workers.tavily_search import TavilySearchAgent

from .conftest import MockExecutor

_L1_ALLOCATION = {
    "claude_search": 1,
    "gemini_search": 1,
    "tavily_search": 1,
    "perplexity_search": 1,
}

_L2_ALLOCATION = {
    "claude_search": 2,
    "gemini_search": 2,
    "tavily_search": 1,
    "perplexity_search": 1,
}


def _retry_config() -> RetryConfig:
    return RetryConfig(
        max_transient_retries=1,
        max_validation_retries=1,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        exponential_base=2.0,
    )


# ---------------------------------------------------------------------------
# ExecutorProtocol compliance
# ---------------------------------------------------------------------------
class TestExecutorProtocol:
    """Test that MockExecutor satisfies ExecutorProtocol."""

    def test_mock_executor_is_protocol_compliant(self) -> None:
        executor: ExecutorProtocol = MockExecutor()
        assert hasattr(executor, "execute")

    @pytest.mark.asyncio
    async def test_protocol_execute_signature(self) -> None:
        executor: ExecutorProtocol = MockExecutor()
        request = ExecutionRequest(prompt="Test", model="sonnet")

        executor.add_result(ExecutionResult.from_success("done"))
        result = await executor.execute(request)
        assert isinstance(result, ExecutionResult)

        executor.add_result(ExecutionResult.from_success("done"))
        result = await executor.execute(request, session_id="sess-1")
        assert isinstance(result, ExecutionResult)


# ---------------------------------------------------------------------------
# Prompt Registry (build_system_prompt / build_user_prompt)
# ---------------------------------------------------------------------------
class TestPromptComposition:
    """Test prompt composition via PromptRegistry."""

    def test_build_system_prompt_websearch(self) -> None:
        registry = PromptRegistry()
        result = registry.build_system_prompt("websearch")
        assert "WebSearch" in result
        assert "WebFetch" in result
        assert "{worker_section}" not in result

    def test_build_system_prompt_tavily(self) -> None:
        registry = PromptRegistry()
        result = registry.build_system_prompt("tavily")
        assert "tavily_search" in result
        assert "tavily_extract" in result
        assert "{worker_section}" not in result

    def test_build_system_prompt_perplexity(self) -> None:
        registry = PromptRegistry()
        result = registry.build_system_prompt("perplexity")
        assert "perplexity" in result.lower()
        assert "search" in result.lower()
        assert "reason" in result.lower()
        assert "{worker_section}" not in result

    def test_build_system_prompt_gemini(self) -> None:
        registry = PromptRegistry()
        result = registry.build_system_prompt("gemini")
        assert "google_web_search" in result
        assert "{worker_section}" not in result

    def test_build_system_prompt_unknown_raises(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(RuntimeError, match="Worker section not found"):
            registry.build_system_prompt("nonexistent_worker")

    def test_build_user_prompt(self) -> None:
        registry = PromptRegistry()
        result = registry.build_user_prompt("What is Python?", 60)
        assert "What is Python?" in result
        assert "60" in result
        assert "{query}" not in result
        assert "{timeout_seconds}" not in result

    def test_build_system_prompt_contains_shared_sections(self) -> None:
        """All system prompts share the same base template."""
        registry = PromptRegistry()
        ws = registry.build_system_prompt("websearch")
        tv = registry.build_system_prompt("tavily")
        assert ws.startswith("You are a web research agent.")
        assert tv.startswith("You are a web research agent.")


# ---------------------------------------------------------------------------
# ClaudeSearchAgent
# ---------------------------------------------------------------------------
class TestClaudeSearchAgent:
    """Test ClaudeSearchAgent (ex-ResearcherAgent)."""

    def _make_agent(self, executor: MockExecutor) -> ClaudeSearchAgent:
        return ClaudeSearchAgent(
            executor=executor,
            model="sonnet",
            timeout=60,
            visible_timeout=30,
            hooks_config={"hooks": {}},
            env_vars=(("PRISM_START_TIME", "1000"),),
            effort="high",
        )

    def test_agent_type(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        assert agent.agent_type == "claude_search"

    def test_is_cancellable(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        assert agent.is_cancellable is True

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(
            ExecutionResult.from_success('{"result": "found it"}', "sess-1")
        )

        result = await agent.execute("test query")

        assert result.success is True
        assert result.content == "found it"
        assert result.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_execute_builds_correct_request(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("my query")

        req = mock_executor.calls[0][0]
        assert req.tools is None
        assert req.allowed_tools == ("WebSearch", "WebFetch")
        assert req.no_session_persistence is True
        assert req.hooks_config == {"hooks": {}}
        assert req.effort == "high"
        assert "my query" in req.prompt
        assert "30" in req.prompt

    @pytest.mark.asyncio
    async def test_execute_no_mcp_config(
        self, mock_executor: MockExecutor
    ) -> None:
        """ClaudeSearch does NOT use MCP config."""
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query")

        req = mock_executor.calls[0][0]
        assert req.mcp_config is None
        assert req.strict_mcp is False

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_error("API error"))

        result = await agent.execute("query")

        assert result.success is False
        assert "API error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_timeout_override(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query", timeout_seconds=120)

        req = mock_executor.calls[0][0]
        assert req.timeout_seconds == 120


# ---------------------------------------------------------------------------
# TavilySearchAgent
# ---------------------------------------------------------------------------
class TestTavilySearchAgent:
    """Test TavilySearchAgent."""

    def _make_agent(self, executor: MockExecutor) -> TavilySearchAgent:
        return TavilySearchAgent(
            executor=executor,
            model="sonnet",
            timeout=60,
            visible_timeout=30,
            hooks_config={"hooks": {}},
            env_vars=(("PRISM_START_TIME", "1000"),),
        )

    def test_agent_type(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        assert agent.agent_type == "tavily_search"

    @pytest.mark.asyncio
    async def test_execute_correct_tools(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query")

        req = mock_executor.calls[0][0]
        assert req.tools == "mcp"
        assert req.allowed_tools == (
            "mcp__tavily__tavily_search",
            "mcp__tavily__tavily_extract",
        )

    @pytest.mark.asyncio
    async def test_execute_has_mcp_config(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query")

        req = mock_executor.calls[0][0]
        assert req.mcp_config is not None
        assert "tavily" in req.mcp_config["mcpServers"]
        assert req.strict_mcp is True
        assert req.no_session_persistence is True

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(
            ExecutionResult.from_success('{"result": "tavily data"}', "sess-2")
        )

        result = await agent.execute("query")

        assert result.success is True
        assert result.content == "tavily data"


# ---------------------------------------------------------------------------
# PerplexitySearchAgent
# ---------------------------------------------------------------------------
class TestPerplexitySearchAgent:
    """Test PerplexitySearchAgent (full rewrite)."""

    def _make_agent(self, executor: MockExecutor) -> PerplexitySearchAgent:
        return PerplexitySearchAgent(
            executor=executor,
            model="sonnet",
            timeout=60,
            visible_timeout=30,
            hooks_config={"hooks": {}},
            env_vars=(("PRISM_START_TIME", "1000"),),
        )

    def test_agent_type(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        assert agent.agent_type == "perplexity_search"

    def test_is_cancellable(self, mock_executor: MockExecutor) -> None:
        """Perplexity IS cancellable now (full Claude worker)."""
        agent = self._make_agent(mock_executor)
        assert agent.is_cancellable is True

    @pytest.mark.asyncio
    async def test_execute_correct_tools(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query")

        req = mock_executor.calls[0][0]
        assert req.tools == "mcp"
        assert req.allowed_tools == (
            "mcp__perplexity__search",
            "mcp__perplexity__reason",
        )

    @pytest.mark.asyncio
    async def test_execute_has_mcp_config(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query")

        req = mock_executor.calls[0][0]
        assert req.mcp_config is not None
        assert "perplexity" in req.mcp_config["mcpServers"]
        assert req.strict_mcp is True
        assert req.no_session_persistence is True

    @pytest.mark.asyncio
    async def test_execute_has_hooks(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(ExecutionResult.from_success('{"result": "ok"}'))

        await agent.execute("query")

        req = mock_executor.calls[0][0]
        assert req.hooks_config == {"hooks": {}}
        assert req.env_vars == (("PRISM_START_TIME", "1000"),)

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_executor: MockExecutor) -> None:
        agent = self._make_agent(mock_executor)
        mock_executor.add_result(
            ExecutionResult.from_success('{"result": "perplexity answer"}')
        )

        result = await agent.execute("query")

        assert result.success is True
        assert result.content == "perplexity answer"


# ---------------------------------------------------------------------------
# GeminiSearchAgent
# ---------------------------------------------------------------------------
class TestGeminiSearchAgent:
    """Test GeminiSearchAgent (new)."""

    def _make_executor(self) -> MagicMock:
        executor = MagicMock()
        executor.execute = AsyncMock()
        executor.parse_gemini_output = GeminiSearchAgent.__class__
        return executor

    def _make_agent(self, executor: MagicMock) -> GeminiSearchAgent:
        return GeminiSearchAgent(
            executor=executor,
            model="gemini-2.5-flash",
            timeout=60,
            visible_timeout=30,
            env_vars=(("PRISM_HOOK_FORMAT", "gemini"),),
        )

    def test_agent_type(self) -> None:
        executor = self._make_executor()
        agent = self._make_agent(executor)
        assert agent.agent_type == "gemini_search"

    def test_is_cancellable(self) -> None:
        executor = self._make_executor()
        agent = self._make_agent(executor)
        assert agent.is_cancellable is True

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        executor = self._make_executor()
        executor.execute.return_value = ExecutionResult.from_success(
            '{"response": "gemini found it", "stats": {}}', "gem-sess"
        )
        agent = self._make_agent(executor)

        result = await agent.execute("test query")

        assert result.success is True
        assert result.content == "gemini found it"

    @pytest.mark.asyncio
    async def test_execute_builds_correct_request(self) -> None:
        executor = self._make_executor()
        executor.execute.return_value = ExecutionResult.from_success(
            '{"response": "ok", "stats": {}}'
        )
        agent = self._make_agent(executor)

        await agent.execute("my query")

        req = executor.execute.call_args[0][0]
        assert isinstance(req, ExecutionRequest)
        assert req.model == "gemini-2.5-flash"
        assert "my query" in req.prompt
        assert "30" in req.prompt
        assert req.system_prompt is not None
        assert "google_web_search" in req.system_prompt
        assert req.env_vars == (("PRISM_HOOK_FORMAT", "gemini"),)

    @pytest.mark.asyncio
    async def test_execute_failure(self) -> None:
        executor = self._make_executor()
        executor.execute.return_value = ExecutionResult.from_error("Gemini error")
        agent = self._make_agent(executor)

        result = await agent.execute("query")

        assert result.success is False
        assert "Gemini" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_unparseable_json(self) -> None:
        """Falls back to raw output when JSON parsing fails."""
        executor = self._make_executor()
        executor.execute.return_value = ExecutionResult.from_success(
            "plain text response"
        )
        agent = self._make_agent(executor)

        result = await agent.execute("query")

        assert result.success is True
        assert result.content == "plain text response"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class TestFactory:
    """Test create_worker factory."""

    def test_valid_agent_types_constant(self) -> None:
        assert "claude_search" in VALID_AGENT_TYPES
        assert "tavily_search" in VALID_AGENT_TYPES
        assert "perplexity_search" in VALID_AGENT_TYPES
        assert "gemini_search" in VALID_AGENT_TYPES

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_worker(
                "nonexistent",
                MagicMock(),
                MagicMock(),
                level=0,
                timeout=60,
                visible_timeout=30,
            )

    def test_create_claude_search(self) -> None:
        worker = create_worker(
            "claude_search",
            MagicMock(),
            MagicMock(),
            level=0,
            timeout=60,
            visible_timeout=30,
        )
        assert isinstance(worker, ClaudeSearchAgent)
        assert worker.agent_type == "claude_search"

    def test_create_tavily_search(self) -> None:
        worker = create_worker(
            "tavily_search",
            MagicMock(),
            MagicMock(),
            level=0,
            timeout=60,
            visible_timeout=30,
        )
        assert isinstance(worker, TavilySearchAgent)
        assert worker.agent_type == "tavily_search"

    def test_create_perplexity_search(self) -> None:
        worker = create_worker(
            "perplexity_search",
            MagicMock(),
            MagicMock(),
            level=0,
            timeout=60,
            visible_timeout=30,
        )
        assert isinstance(worker, PerplexitySearchAgent)
        assert worker.agent_type == "perplexity_search"

    def test_create_gemini_search(self) -> None:
        worker = create_worker(
            "gemini_search",
            MagicMock(),
            MagicMock(),
            level=0,
            timeout=60,
            visible_timeout=30,
        )
        assert isinstance(worker, GeminiSearchAgent)
        assert worker.agent_type == "gemini_search"

    def test_claude_worker_gets_hooks(self) -> None:
        worker = create_worker(
            "claude_search",
            MagicMock(),
            MagicMock(),
            level=0,
            timeout=60,
            visible_timeout=30,
        )
        assert worker._hooks_config is not None
        assert "hooks" in worker._hooks_config
        assert worker._env_vars is not None

    def test_gemini_worker_gets_env_vars(self) -> None:
        worker = create_worker(
            "gemini_search",
            MagicMock(),
            MagicMock(),
            level=0,
            timeout=60,
            visible_timeout=30,
        )
        assert worker._env_vars is not None
        env_dict = dict(worker._env_vars)
        assert "PRISM_HOOK_FORMAT" in env_dict
        assert env_dict["PRISM_HOOK_FORMAT"] == "gemini"

    def test_factory_uses_level_for_model(self) -> None:
        """Factory selects model from config based on level."""
        w0 = create_worker(
            "claude_search", MagicMock(), MagicMock(),
            level=0, timeout=60, visible_timeout=30,
        )
        w3 = create_worker(
            "claude_search", MagicMock(), MagicMock(),
            level=3, timeout=600, visible_timeout=480,
        )
        assert w0._model is not None
        assert w3._model is not None


# ---------------------------------------------------------------------------
# ManagerAgent
# ---------------------------------------------------------------------------
class TestManagerAgent:
    """Test ManagerAgent with new plan/synthesize interface."""

    def test_task_defaults_to_claude_search(self) -> None:
        task = Task(query="test")
        assert task.agent_type == "claude_search"

    def test_task_plan_from_keyed_dict_new_format(self) -> None:
        data = {
            "claude_search_1": "search query for claude",
            "gemini_search_1": "search query for gemini",
            "tavily_search_1": "search query for tavily",
        }
        plan = TaskPlan.from_keyed_dict(data)
        assert len(plan.tasks) == 3
        types = {t.agent_type for t in plan.tasks}
        assert types == {"claude_search", "gemini_search", "tavily_search"}
        claude_task = [t for t in plan.tasks if t.agent_type == "claude_search"][0]
        assert claude_task.query == "search query for claude"
        assert claude_task.key == "claude_search_1"

    def test_task_plan_from_keyed_dict_legacy_format(self) -> None:
        data = {
            "claude_search_1": {"query": "q1"},
            "gemini_search_1": {"query": "q2"},
        }
        plan = TaskPlan.from_keyed_dict(data)
        assert len(plan.tasks) == 2
        assert plan.tasks[0].query in ("q1", "q2")

    def test_task_plan_from_keyed_dict_skips_non_task_keys(self) -> None:
        data = {
            "claude_search_1": "q1",
            "not_a_task": "ignored",
        }
        plan = TaskPlan.from_keyed_dict(data)
        assert len(plan.tasks) == 1

    def test_build_task_schema_l1(self) -> None:
        schema = ManagerAgent._build_task_schema(_L1_ALLOCATION)
        assert schema["type"] == "object"
        assert "claude_search_1" in schema["required"]
        assert "gemini_search_1" in schema["required"]
        assert "tavily_search_1" in schema["required"]
        assert "perplexity_search_1" in schema["required"]
        assert len(schema["required"]) == 4
        assert schema["additionalProperties"] is False
        assert schema["properties"]["claude_search_1"]["type"] == "string"

    def test_build_task_schema_l2(self) -> None:
        schema = ManagerAgent._build_task_schema(_L2_ALLOCATION)
        assert "claude_search_1" in schema["required"]
        assert "claude_search_2" in schema["required"]
        assert "gemini_search_1" in schema["required"]
        assert "gemini_search_2" in schema["required"]
        assert "tavily_search_1" in schema["required"]
        assert "perplexity_search_1" in schema["required"]
        assert len(schema["required"]) == 6

    def test_build_task_schema_excludes_zero_count(self) -> None:
        alloc = {"claude_search": 2, "gemini_search": 0, "tavily_search": 1}
        schema = ManagerAgent._build_task_schema(alloc)
        assert "gemini_search_1" not in schema["required"]
        assert "claude_search_1" in schema["required"]
        assert "claude_search_2" in schema["required"]
        assert "tavily_search_1" in schema["required"]

    def test_render_agent_section(self, mock_executor: MockExecutor) -> None:
        manager = ManagerAgent(
            executor=mock_executor, model="s", agent_allocation=_L1_ALLOCATION, level=1,
        )
        section = manager._render_agent_section()
        assert "claude_search" in section
        assert "gemini_search" in section
        assert "tavily_search" in section
        assert "perplexity_search" in section
        assert "claude_search_1" in section
        assert "1 slot:" in section

    def test_render_agent_section_l2_plural(self, mock_executor: MockExecutor) -> None:
        manager = ManagerAgent(
            executor=mock_executor, model="s", agent_allocation=_L2_ALLOCATION, level=2,
        )
        section = manager._render_agent_section()
        assert "2 slots:" in section
        assert "claude_search_1, claude_search_2" in section

    def test_render_agent_section_excludes_zero(self, mock_executor: MockExecutor) -> None:
        alloc = {"claude_search": 1, "gemini_search": 0}
        manager = ManagerAgent(
            executor=mock_executor, model="s", agent_allocation=alloc, level=1,
        )
        section = manager._render_agent_section()
        assert "claude_search" in section
        assert "gemini_search" not in section

    def test_manager_system_prompt_is_lean(self) -> None:
        """Manager system prompt is lean -- no agent types, no strategy."""
        registry = PromptRegistry()
        content = registry.get_content("search_manager/system")
        assert content is not None
        assert "search session manager" in content
        assert "claude_search" not in content

    def test_response_schema_loaded(self) -> None:
        registry = PromptRegistry()
        schema = registry.get_schema("search_manager/response_schema")
        assert schema is not None
        assert "response" in schema["required"]
        assert schema["properties"]["response"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_plan_success(self, mock_executor: MockExecutor) -> None:
        """Manager.plan() creates task plan from keyed-dict output."""
        retry_executor = RetryExecutor(mock_executor, _retry_config())

        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=_L1_ALLOCATION,
            level=1,
        )

        keyed_plan = {
            "claude_search_1": "q1",
            "gemini_search_1": "q2",
            "tavily_search_1": "q3",
            "perplexity_search_1": "q4",
        }
        mock_executor.add_result(
            ExecutionResult.from_success(
                json.dumps({"type": "result", "structured_output": keyed_plan}),
                "sess-mgr",
            )
        )

        result = await manager.plan("test query")

        assert result.success is True
        assert isinstance(result.content, dict)
        assert len(result.content["tasks"]) == 4
        assert manager.session_id == "sess-mgr"

    @pytest.mark.asyncio
    async def test_plan_sets_session_id(self, mock_executor: MockExecutor) -> None:
        alloc = {"claude_search": 1}
        retry_executor = RetryExecutor(mock_executor, _retry_config())
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=alloc,
            level=1,
        )

        keyed_plan = {"claude_search_1": "q"}
        mock_executor.add_result(
            ExecutionResult.from_success(
                json.dumps({"type": "result", "structured_output": keyed_plan}),
                "new-sess",
            )
        )

        await manager.plan("test")
        assert manager.session_id == "new-sess"

    @pytest.mark.asyncio
    async def test_plan_failure(self, mock_executor: MockExecutor) -> None:
        retry_executor = RetryExecutor(mock_executor, RetryConfig(
            max_transient_retries=0, max_validation_retries=0,
            base_delay_seconds=0.01, max_delay_seconds=0.1, exponential_base=2.0,
        ))
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=_L1_ALLOCATION,
            level=1,
        )

        mock_executor.add_result(ExecutionResult.from_error("plan broke"))

        result = await manager.plan("test")
        assert result.success is False
        assert "plan broke" in (result.error or "")

    @pytest.mark.asyncio
    async def test_synthesize_success(self, mock_executor: MockExecutor) -> None:
        retry_executor = RetryExecutor(mock_executor, _retry_config())
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=_L1_ALLOCATION,
            level=1,
            session_id="mgr-sess",
        )

        mock_executor.add_result(
            ExecutionResult.from_success(
                json.dumps({
                    "type": "result",
                    "structured_output": {"response": "synthesized answer"},
                }),
                "synth-sess",
            )
        )

        from prism.workers.base import AgentResult
        results = [
            AgentResult.from_success(content="data1", agent_key="claude_search_1"),
            AgentResult.from_success(content="data2", agent_key="gemini_search_1"),
        ]

        result = await manager.synthesize(results)

        assert result.success is True
        assert result.content == "synthesized answer"
        # Verify --resume was used
        req = mock_executor.calls[0][0]
        assert req.resume_session == "mgr-sess"
        assert req.timeout_seconds is None

    @pytest.mark.asyncio
    async def test_synthesize_no_session_id(self, mock_executor: MockExecutor) -> None:
        retry_executor = RetryExecutor(mock_executor, _retry_config())
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=_L1_ALLOCATION,
            level=1,
        )

        from prism.workers.base import AgentResult
        result = await manager.synthesize([AgentResult.from_success(content="d")])
        assert result.success is False
        assert "No session_id" in (result.error or "")

    @pytest.mark.asyncio
    async def test_follow_up_chat_success(self, mock_executor: MockExecutor) -> None:
        retry_executor = RetryExecutor(mock_executor, _retry_config())
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=_L1_ALLOCATION,
            level=1,
            session_id="existing-sess",
        )

        mock_executor.add_result(
            ExecutionResult.from_success(
                json.dumps({
                    "type": "result",
                    "structured_output": {"response": "follow-up answer"},
                }),
                "new-sess",
            )
        )

        result = await manager.follow_up_chat("What about X?")

        assert result.success is True
        assert result.content == "follow-up answer"
        req = mock_executor.calls[0][0]
        assert req.resume_session == "existing-sess"

    @pytest.mark.asyncio
    async def test_follow_up_chat_no_session(self, mock_executor: MockExecutor) -> None:
        retry_executor = RetryExecutor(mock_executor, _retry_config())
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=_L1_ALLOCATION,
            level=1,
        )

        result = await manager.follow_up_chat("question")
        assert result.success is False
        assert "No session_id" in (result.error or "")

    @pytest.mark.asyncio
    async def test_follow_up_search_success(self, mock_executor: MockExecutor) -> None:
        alloc = {"claude_search": 1}
        retry_executor = RetryExecutor(mock_executor, _retry_config())
        manager = ManagerAgent(
            executor=retry_executor,
            model="sonnet",
            agent_allocation=alloc,
            level=1,
            session_id="existing-sess",
        )

        keyed_plan = {"claude_search_1": "fq1"}
        mock_executor.add_result(
            ExecutionResult.from_success(
                json.dumps({"type": "result", "structured_output": keyed_plan}),
                "fu-sess",
            )
        )

        result = await manager.follow_up_search("dig deeper")

        assert result.success is True
        assert isinstance(result.content, dict)
        assert len(result.content["tasks"]) == 1
        req = mock_executor.calls[0][0]
        assert req.resume_session == "existing-sess"

    def test_plan_request_has_system_prompt_and_model(self, mock_executor: MockExecutor) -> None:
        """Plan uses model and system_prompt; synthesis does not."""
        manager = ManagerAgent(
            executor=mock_executor,
            model="opus",
            agent_allocation=_L1_ALLOCATION,
            level=1,
        )
        assert manager._system_prompt is not None
        assert "search session manager" in manager._system_prompt


# ---------------------------------------------------------------------------
# Worker Pattern Tests (generic patterns)
# ---------------------------------------------------------------------------
class TestWorkerPatterns:
    """Test worker-executor interaction patterns."""

    @pytest.mark.asyncio
    async def test_worker_uses_executor(
        self,
        mock_executor: MockExecutor,
        sample_request: ExecutionRequest,
    ) -> None:
        mock_executor.add_result(
            ExecutionResult.from_success('{"result": "data"}')
        )

        result = await mock_executor.execute(sample_request)

        assert result.success is True
        assert len(mock_executor.calls) == 1
        assert mock_executor.calls[0][0] == sample_request

    @pytest.mark.asyncio
    async def test_worker_with_schema_validation(
        self,
        mock_executor: MockExecutor,
        sample_schema: dict,
    ) -> None:
        config = RetryConfig(
            max_transient_retries=1,
            max_validation_retries=1,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
            exponential_base=2.0,
        )
        retry_executor = RetryExecutor(mock_executor, config)

        request = ExecutionRequest(
            prompt="Get structured data",
            model="sonnet",
            json_schema=sample_schema,
        )
        mock_executor.add_result(
            ExecutionResult.from_success(
                '{"result": "answer", "confidence": 0.9}',
                "sess-1",
            )
        )

        result = await retry_executor.execute(request, schema=sample_schema)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_with_tools(
        self,
        mock_executor: MockExecutor,
    ) -> None:
        request = ExecutionRequest(
            prompt="Search the web",
            model="sonnet",
            tools="mcp",
            allowed_tools=("web_search", "web_fetch"),
        )
        mock_executor.add_result(
            ExecutionResult.from_success('{"sources": ["url1"]}', "sess-1")
        )

        result = await mock_executor.execute(request)

        assert result.success is True
        called_request = mock_executor.calls[0][0]
        assert called_request.tools == "mcp"
        assert "web_search" in called_request.allowed_tools


# ---------------------------------------------------------------------------
# Agent ABC / Subclass Verification
# ---------------------------------------------------------------------------
class TestAgentSubclasses:
    """Verify all workers are proper Agent subclasses."""

    def test_claude_search_is_agent(self, mock_executor: MockExecutor) -> None:
        agent = ClaudeSearchAgent(
            executor=mock_executor, model="s", timeout=60, visible_timeout=30,
        )
        assert isinstance(agent, Agent)

    def test_tavily_search_is_agent(self, mock_executor: MockExecutor) -> None:
        agent = TavilySearchAgent(
            executor=mock_executor, model="s", timeout=60, visible_timeout=30,
        )
        assert isinstance(agent, Agent)

    def test_perplexity_search_is_agent(
        self, mock_executor: MockExecutor
    ) -> None:
        agent = PerplexitySearchAgent(
            executor=mock_executor, model="s", timeout=60, visible_timeout=30,
        )
        assert isinstance(agent, Agent)

    def test_gemini_search_is_agent(self) -> None:
        executor = MagicMock()
        agent = GeminiSearchAgent(
            executor=executor, model="g", timeout=60, visible_timeout=30,
        )
        assert isinstance(agent, Agent)
