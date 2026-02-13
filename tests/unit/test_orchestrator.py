"""Tests for orchestrator: flow, dispatcher.

Validates multi-provider L0, timeout=None for L1-3,
factory-based dispatcher, manager-driven synthesis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prism.core.response import ExecutionResult
from prism.core.retry import RetryExecutor
from prism.orchestrator.dispatcher import WorkerDispatcher
from prism.orchestrator.flow import ALL_PROVIDERS, SearchFlow, SearchResult
from prism.workers.base import AgentResult
from prism.workers.manager import Task, TaskPlan

from .conftest import MockExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_registry() -> MagicMock:
    registry = MagicMock()
    registry.register = AsyncMock()
    registry.unregister = AsyncMock()
    return registry


def _make_session_repository() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    return repo


def _make_search_flow(
    mock_executor: MockExecutor,
    gemini_executor: MagicMock | None = None,
    dispatcher: MagicMock | None = None,
) -> SearchFlow:
    """Create SearchFlow with mocked dependencies."""
    return SearchFlow(
        retry_executor=mock_executor,
        gemini_executor=gemini_executor or MagicMock(),
        dispatcher=dispatcher or MagicMock(),
        session_registry=_make_session_registry(),
        session_repository=_make_session_repository(),
        user_id="test-user",
    )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------
class TestSearchResult:
    def test_to_dict(self) -> None:
        result = SearchResult(
            success=True,
            content="test",
            session_id="sess-1",
            level=0,
            query="q",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["content"] == "test"
        assert d["session_id"] == "sess-1"
        assert d["level"] == 0

    def test_defaults(self) -> None:
        result = SearchResult(success=False, content="")
        assert result.level == 0
        assert result.error is None
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# WorkerDispatcher
# ---------------------------------------------------------------------------
class TestWorkerDispatcher:
    """Test dispatcher delegates to factory."""

    def _make_dispatcher(self) -> WorkerDispatcher:
        return WorkerDispatcher(
            claude_executor=MagicMock(),
            gemini_executor=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_dispatch_empty_plan(self) -> None:
        dispatcher = self._make_dispatcher()
        plan = TaskPlan(tasks=[], )

        results = await dispatcher.dispatch(
            task_plan=plan,
            worker_timeout=60,
            visible_timeout=30,
            level=1,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_dispatch_delegates_to_factory(self) -> None:
        """Dispatcher creates workers via factory, not direct imports."""
        dispatcher = self._make_dispatcher()

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="result")
        )

        with patch(
            "prism.orchestrator.dispatcher.create_worker",
            return_value=mock_worker,
        ) as mock_factory:
            plan = TaskPlan(
                tasks=[Task(query="test", agent_type="claude_search")],
            )

            results = await dispatcher.dispatch(
                task_plan=plan,
                worker_timeout=60,
                visible_timeout=30,
                level=2,
            )

            mock_factory.assert_called_once_with(
                agent_type="claude_search",
                claude_executor=dispatcher._claude_executor,
                gemini_executor=dispatcher._gemini_executor,
                level=2,
                timeout=60,
                visible_timeout=30,
            )
            assert len(results) == 1
            assert results[0].success is True

    @pytest.mark.asyncio
    async def test_dispatch_passes_level(self) -> None:
        """Level is passed to factory for model selection."""
        dispatcher = self._make_dispatcher()

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="ok")
        )

        with patch(
            "prism.orchestrator.dispatcher.create_worker",
            return_value=mock_worker,
        ) as mock_factory:
            plan = TaskPlan(
                tasks=[Task(query="q", agent_type="gemini_search")],
            )

            await dispatcher.dispatch(
                task_plan=plan,
                worker_timeout=600,
                visible_timeout=480,
                level=3,
            )

            assert mock_factory.call_args.kwargs["level"] == 3

    @pytest.mark.asyncio
    async def test_dispatch_parallel_multiple_tasks(self) -> None:
        """Multiple tasks dispatched in parallel via asyncio.gather."""
        dispatcher = self._make_dispatcher()

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="ok")
        )

        with patch(
            "prism.orchestrator.dispatcher.create_worker",
            return_value=mock_worker,
        ):
            plan = TaskPlan(
                tasks=[
                    Task(query="q1", agent_type="claude_search"),
                    Task(query="q2", agent_type="tavily_search"),
                    Task(query="q3", agent_type="perplexity_search"),
                ],
            )

            results = await dispatcher.dispatch(
                task_plan=plan,
                worker_timeout=60,
                visible_timeout=30,
                level=1,
            )

            assert len(results) == 3
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dispatch_handles_task_exception(self) -> None:
        """Tasks that raise exceptions are caught and returned as errors."""
        dispatcher = self._make_dispatcher()

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "prism.orchestrator.dispatcher.create_worker",
            return_value=mock_worker,
        ):
            plan = TaskPlan(
                tasks=[Task(query="q", agent_type="claude_search")],
            )

            results = await dispatcher.dispatch(
                task_plan=plan,
                worker_timeout=60,
                visible_timeout=30,
                level=1,
            )

            assert len(results) == 1
            assert results[0].success is False
            assert "boom" in (results[0].error or "")

    @pytest.mark.asyncio
    async def test_dispatch_tracks_wall_time(self) -> None:
        """Dispatcher adds wall_time_s and agent_key to result metadata."""
        dispatcher = self._make_dispatcher()

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="ok")
        )

        with patch(
            "prism.orchestrator.dispatcher.create_worker",
            return_value=mock_worker,
        ):
            plan = TaskPlan(
                tasks=[
                    Task(query="q", agent_type="claude_search", key="claude_search_1"),
                ],
            )

            results = await dispatcher.dispatch(
                task_plan=plan,
                worker_timeout=60,
                visible_timeout=30,
                level=1,
            )

            assert len(results) == 1
            assert results[0].metadata.get("agent_key") == "claude_search_1"
            assert "wall_time_s" in results[0].metadata


# ---------------------------------------------------------------------------
# SearchFlow — validation
# ---------------------------------------------------------------------------
class TestSearchFlowValidation:
    """Test input validation (doesn't need real executors)."""

    @pytest.mark.asyncio
    async def test_invalid_level(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = await flow.execute_search("query", level=5)
        assert result.success is False
        assert "Invalid level" in (result.error or "")

    @pytest.mark.asyncio
    async def test_negative_level(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = await flow.execute_search("query", level=-1)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = await flow.execute_search("")
        assert result.success is False
        assert "empty" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_whitespace_query(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = await flow.execute_search("   ")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_default_level_is_zero(self, mock_executor: MockExecutor) -> None:
        """Level defaults to 0."""
        flow = _make_search_flow(mock_executor)
        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="ok")
        )
        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ):
            result = await flow.execute_search("test query")
            assert result.level == 0


# ---------------------------------------------------------------------------
# SearchFlow — Level 0 multi-provider
# ---------------------------------------------------------------------------
class TestSearchFlowLevel0:
    """Test L0 multi-provider via factory."""

    @pytest.mark.asyncio
    async def test_l0_default_provider(self, mock_executor: MockExecutor) -> None:
        """L0 with providers=None uses config default (claude_search)."""
        flow = _make_search_flow(mock_executor)

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="search result")
        )

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ) as mock_factory:
            result = await flow.execute_search("test query", level=0)

            assert result.success is True
            assert result.content == "search result"
            assert result.level == 0
            assert mock_factory.call_args.kwargs["agent_type"] == "claude_search"

    @pytest.mark.asyncio
    async def test_l0_single_provider_no_section_header(
        self, mock_executor: MockExecutor
    ) -> None:
        """Single provider returns content directly without section headers."""
        flow = _make_search_flow(mock_executor)

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="direct content")
        )

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ):
            result = await flow.execute_search(
                "test", level=0, providers=["tavily_search"]
            )

            assert result.success is True
            assert result.content == "direct content"
            assert "##" not in result.content

    @pytest.mark.asyncio
    async def test_l0_multiple_providers_sectioned(
        self, mock_executor: MockExecutor
    ) -> None:
        """Multiple providers return sectioned results."""
        flow = _make_search_flow(mock_executor)

        call_count = 0

        async def _execute_side_effect(prompt, timeout_seconds=None, parent_session_id=None):
            nonlocal call_count
            call_count += 1
            return AgentResult.from_success(content=f"result {call_count}")

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(side_effect=_execute_side_effect)

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ):
            result = await flow.execute_search(
                "test", level=0, providers=["claude_search", "tavily_search"]
            )

            assert result.success is True
            assert "## claude_search" in result.content
            assert "## tavily_search" in result.content
            assert "---" in result.content

    @pytest.mark.asyncio
    async def test_l0_mix_expands_to_all_providers(
        self, mock_executor: MockExecutor
    ) -> None:
        """providers=['mix'] expands to all 4 provider types."""
        flow = _make_search_flow(mock_executor)

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="ok")
        )

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ) as mock_factory:
            result = await flow.execute_search(
                "test", level=0, providers=["mix"]
            )

            assert result.success is True
            assert mock_factory.call_count == 4
            created_types = [
                call.kwargs["agent_type"] for call in mock_factory.call_args_list
            ]
            assert set(created_types) == set(ALL_PROVIDERS)

    @pytest.mark.asyncio
    async def test_l0_unknown_provider_error(
        self, mock_executor: MockExecutor
    ) -> None:
        flow = _make_search_flow(mock_executor)
        result = await flow.execute_search(
            "test", level=0, providers=["nonexistent"]
        )
        assert result.success is False
        assert "Unknown provider" in (result.error or "")

    @pytest.mark.asyncio
    async def test_l0_all_providers_fail(
        self, mock_executor: MockExecutor
    ) -> None:
        flow = _make_search_flow(mock_executor)

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_error(error="provider failed")
        )

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ):
            result = await flow.execute_search("test", level=0)
            assert result.success is False
            assert "All providers failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_l0_partial_failure(
        self, mock_executor: MockExecutor
    ) -> None:
        """If one provider fails but another succeeds, result is still successful."""
        flow = _make_search_flow(mock_executor)

        call_count = 0

        async def _execute_side_effect(prompt, timeout_seconds=None, parent_session_id=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AgentResult.from_error(error="timeout")
            return AgentResult.from_success(content="good data")

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(side_effect=_execute_side_effect)

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ):
            result = await flow.execute_search(
                "test", level=0, providers=["claude_search", "tavily_search"]
            )

            assert result.success is True
            assert result.metadata.get("errors") is not None

    @pytest.mark.asyncio
    async def test_l0_uses_factory_with_level_0(
        self, mock_executor: MockExecutor
    ) -> None:
        """Factory is called with level=0 for L0 searches."""
        flow = _make_search_flow(mock_executor)

        mock_worker = MagicMock()
        mock_worker.execute = AsyncMock(
            return_value=AgentResult.from_success(content="ok")
        )

        with patch(
            "prism.orchestrator.flow.create_worker",
            return_value=mock_worker,
        ) as mock_factory:
            await flow.execute_search("test", level=0)
            assert mock_factory.call_args.kwargs["level"] == 0


# ---------------------------------------------------------------------------
# SearchFlow — Level 1-3
# ---------------------------------------------------------------------------
class TestSearchFlowLevel1_3:
    """Test L1-3 orchestrated flow with ManagerAgent plan/synthesize."""

    @staticmethod
    def _keyed_plan_output(session_id: str = "mgr-sess") -> ExecutionResult:
        """Create a successful manager plan result in keyed-dict format."""
        import json
        keyed_plan = {
            "claude_search_1": "q1",
            "gemini_search_1": "q2",
            "tavily_search_1": "q3",
            "perplexity_search_1": "q4",
        }
        return ExecutionResult.from_success(
            json.dumps({"type": "result", "structured_output": keyed_plan}),
            session_id,
        )

    @staticmethod
    def _synthesis_output(session_id: str = "synth-sess") -> ExecutionResult:
        """Create a successful synthesis result."""
        import json
        return ExecutionResult.from_success(
            json.dumps({"type": "result", "structured_output": {"response": "synthesized"}}),
            session_id,
        )

    @pytest.mark.asyncio
    async def test_l1_manager_called_with_timeout_none(
        self, mock_executor: MockExecutor
    ) -> None:
        """Manager plan() is called with timeout_seconds=None."""
        retry_config = _retry_config()
        retry_executor = RetryExecutor(mock_executor, retry_config)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(
            return_value=[AgentResult.from_success(content="worker data")]
        )

        flow = SearchFlow(
            retry_executor=retry_executor,
            gemini_executor=MagicMock(),
            dispatcher=dispatcher,
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )

        # Queue plan + synthesis results
        mock_executor.add_result(self._keyed_plan_output())
        mock_executor.add_result(self._synthesis_output())

        result = await flow.execute_search("test query", level=1)

        assert result.success is True
        # First call is plan — check timeout_seconds=None
        req = mock_executor.calls[0][0]
        assert req.timeout_seconds is None

    @pytest.mark.asyncio
    async def test_l1_synthesize_uses_resume(
        self, mock_executor: MockExecutor
    ) -> None:
        """Synthesize call uses --resume with manager's session_id."""
        retry_config = _retry_config()
        retry_executor = RetryExecutor(mock_executor, retry_config)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(
            return_value=[AgentResult.from_success(content="data")]
        )

        flow = SearchFlow(
            retry_executor=retry_executor,
            gemini_executor=MagicMock(),
            dispatcher=dispatcher,
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )

        mock_executor.add_result(self._keyed_plan_output("mgr-sess-id"))
        mock_executor.add_result(self._synthesis_output())

        await flow.execute_search("test query", level=2)

        # Second call is synthesis — check resume_session
        synth_req = mock_executor.calls[1][0]
        assert synth_req.resume_session == "mgr-sess-id"
        assert synth_req.timeout_seconds is None

    @pytest.mark.asyncio
    async def test_l1_dispatcher_receives_level(
        self, mock_executor: MockExecutor
    ) -> None:
        """Dispatcher.dispatch is called with the correct level."""
        retry_config = _retry_config()
        retry_executor = RetryExecutor(mock_executor, retry_config)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(
            return_value=[AgentResult.from_success(content="data")]
        )

        flow = SearchFlow(
            retry_executor=retry_executor,
            gemini_executor=MagicMock(),
            dispatcher=dispatcher,
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )

        # L3 allocation: claude=3, gemini=2, tavily=2, perplexity=1
        import json
        l3_plan = {
            "claude_search_1": "q1",
            "claude_search_2": "q2",
            "claude_search_3": "q3",
            "gemini_search_1": "q4",
            "gemini_search_2": "q5",
            "tavily_search_1": "q6",
            "tavily_search_2": "q7",
            "perplexity_search_1": "q8",
        }
        mock_executor.add_result(
            ExecutionResult.from_success(
                json.dumps({"type": "result", "structured_output": l3_plan}),
                "mgr-sess",
            )
        )
        mock_executor.add_result(self._synthesis_output())

        await flow.execute_search("test", level=3)

        dispatch_call = dispatcher.dispatch.call_args
        assert dispatch_call.kwargs["level"] == 3

    @pytest.mark.asyncio
    async def test_l1_manager_failure(
        self, mock_executor: MockExecutor
    ) -> None:
        """Manager failure returns error result."""
        retry_config = _retry_config(retries=0)
        retry_executor = RetryExecutor(mock_executor, retry_config)

        flow = SearchFlow(
            retry_executor=retry_executor,
            gemini_executor=MagicMock(),
            dispatcher=MagicMock(),
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )

        mock_executor.add_result(ExecutionResult.from_error("manager broke"))

        result = await flow.execute_search("test", level=1)

        assert result.success is False
        assert "Manager planning failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_l1_all_workers_fail(
        self, mock_executor: MockExecutor
    ) -> None:
        """All workers failing returns error."""
        retry_config = _retry_config()
        retry_executor = RetryExecutor(mock_executor, retry_config)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(
            return_value=[AgentResult.from_error(error="all failed")]
        )

        flow = SearchFlow(
            retry_executor=retry_executor,
            gemini_executor=MagicMock(),
            dispatcher=dispatcher,
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )

        mock_executor.add_result(self._keyed_plan_output())

        result = await flow.execute_search("test", level=1)

        assert result.success is False
        assert "All workers failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_l1_synthesis_failure_uses_fallback(
        self, mock_executor: MockExecutor
    ) -> None:
        """When synthesis fails, fallback combines worker results."""
        retry_config = _retry_config()
        retry_executor = RetryExecutor(mock_executor, retry_config)

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(
            return_value=[
                AgentResult.from_success(content="worker 1 data"),
                AgentResult.from_success(content="worker 2 data"),
            ]
        )

        flow = SearchFlow(
            retry_executor=retry_executor,
            gemini_executor=MagicMock(),
            dispatcher=dispatcher,
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )

        mock_executor.add_result(self._keyed_plan_output())
        mock_executor.add_result(ExecutionResult.from_error("synthesis failed"))

        result = await flow.execute_search("test", level=1)

        assert result.success is True
        assert result.metadata.get("fallback") is True
        assert "worker 1 data" in result.content
        assert "worker 2 data" in result.content


# ---------------------------------------------------------------------------
# SearchFlow — provider resolution
# ---------------------------------------------------------------------------
class TestProviderResolution:
    def test_resolve_providers_none(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = flow._resolve_providers(None)
        assert result == ["claude_search"]  # config default

    def test_resolve_providers_mix(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = flow._resolve_providers(["mix"])
        assert set(result) == set(ALL_PROVIDERS)
        assert len(result) == 4

    def test_resolve_providers_explicit(self, mock_executor: MockExecutor) -> None:
        flow = _make_search_flow(mock_executor)
        result = flow._resolve_providers(["tavily_search", "gemini_search"])
        assert result == ["tavily_search", "gemini_search"]

    def test_resolve_providers_mix_overrides_other(
        self, mock_executor: MockExecutor
    ) -> None:
        flow = _make_search_flow(mock_executor)
        result = flow._resolve_providers(["claude_search", "mix"])
        assert set(result) == set(ALL_PROVIDERS)


# ---------------------------------------------------------------------------
# SearchFlow — DI signature
# ---------------------------------------------------------------------------
class TestSearchFlowDI:
    """Verify SearchFlow accepts gemini_executor (no synthesizer)."""

    def test_init_accepts_gemini_executor(self, mock_executor: MockExecutor) -> None:
        gemini = MagicMock()
        flow = SearchFlow(
            retry_executor=mock_executor,
            gemini_executor=gemini,
            dispatcher=MagicMock(),
            session_registry=_make_session_registry(),
            session_repository=_make_session_repository(),
            user_id="test",
        )
        assert flow._gemini_executor is gemini


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry_config(retries: int = 1):
    from prism.config import RetryConfig
    return RetryConfig(
        max_transient_retries=retries,
        max_validation_retries=retries,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        exponential_base=2.0,
    )
