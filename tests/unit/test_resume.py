"""Tests for session resume feature.

Covers:
- SearchFlow.resume_session() success and failure paths
- execute_resume() thin wrapper delegation
- Server-level resume() validation (L0 rejection, expiry, missing claude_session_id)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from prism.core.response import ExecutionRequest, ExecutionResult
from prism.database.models import SearchSession, SessionStatus
from prism.orchestrator.flow import SearchFlow, SearchResult
from prism.tools.search import execute_resume

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


def _make_search_flow(mock_executor: MockExecutor) -> SearchFlow:
    return SearchFlow(
        retry_executor=mock_executor,
        gemini_executor=MagicMock(),
        dispatcher=MagicMock(),
        session_registry=_make_session_registry(),
        session_repository=_make_session_repository(),
    )


def _response_output(text: str, session_id: str | None = None) -> ExecutionResult:
    """Create an ExecutionResult with the {"response": "..."} format."""
    return ExecutionResult.from_success(
        json.dumps({"type": "result", "structured_output": {"response": text}}),
        session_id,
    )


def _make_db_session(
    *,
    level: int = 2,
    claude_session_id: str | None = "claude-sess-123",
    created_at: datetime | None = None,
    status: SessionStatus = SessionStatus.COMPLETED,
) -> MagicMock:
    """Create a mock SearchSession DB object."""
    session = MagicMock(spec=SearchSession)
    session.id = uuid.uuid4()
    session.level = level
    session.claude_session_id = claude_session_id
    session.status = status
    session.created_at = created_at or datetime.now(timezone.utc)
    session.updated_at = datetime.now(timezone.utc)
    session.completed_at = datetime.now(timezone.utc)
    session.query = "original query"
    session.summary = "test summary"
    session.duration_ms = 5000
    session.error_message = None
    session.result = {"success": True, "content": "original result"}
    return session


# ---------------------------------------------------------------------------
# SearchFlow.resume_session -- success
# ---------------------------------------------------------------------------

class TestResumeSessionSuccess:
    """Test SearchFlow.resume_session() success path."""

    @pytest.mark.asyncio
    async def test_resume_returns_success_result(
        self, mock_executor: MockExecutor
    ) -> None:
        """Successful resume returns SearchResult with extracted content."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(_response_output("follow-up answer", "new-sess-id"))

        result = await flow.resume_session(
            claude_session_id="orig-claude-sess",
            follow_up="What about X?",
            session_id="db-session-id",
        )

        assert result.success is True
        assert result.content == "follow-up answer"
        assert result.session_id == "db-session-id"
        assert result.query == "What about X?"
        assert result.metadata["resumed_from"] == "orig-claude-sess"

    @pytest.mark.asyncio
    async def test_resume_sends_correct_execution_request(
        self, mock_executor: MockExecutor
    ) -> None:
        """ExecutionRequest has resume_session, no timeout, rendered prompt."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(_response_output("ok"))

        await flow.resume_session(
            claude_session_id="sess-to-resume",
            follow_up="Follow up question",
            session_id="db-id",
        )

        assert len(mock_executor.calls) == 1
        req, _ = mock_executor.calls[0]
        assert isinstance(req, ExecutionRequest)
        assert "Follow up question" in req.prompt
        assert req.resume_session == "sess-to-resume"
        assert req.timeout_seconds is None

    @pytest.mark.asyncio
    async def test_resume_uses_response_schema(
        self, mock_executor: MockExecutor
    ) -> None:
        """Resume request includes json_schema for structured output."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(_response_output("data"))

        await flow.resume_session(
            claude_session_id="sess-id",
            follow_up="query",
            session_id="db-id",
        )

        req, _ = mock_executor.calls[0]
        assert req.json_schema is not None
        assert "response" in req.json_schema.get("required", [])


# ---------------------------------------------------------------------------
# SearchFlow.resume_session -- failure
# ---------------------------------------------------------------------------

class TestResumeSessionFailure:
    """Test SearchFlow.resume_session() failure path."""

    @pytest.mark.asyncio
    async def test_resume_executor_error(
        self, mock_executor: MockExecutor
    ) -> None:
        """Executor returning error propagates as failed SearchResult."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(
            ExecutionResult.from_error("Connection refused")
        )

        result = await flow.resume_session(
            claude_session_id="sess-id",
            follow_up="query",
            session_id="db-id",
        )

        assert result.success is False
        assert result.content == ""
        assert "Connection refused" in (result.error or "")
        assert result.session_id == "db-id"

    @pytest.mark.asyncio
    async def test_resume_executor_error_fallback_message(
        self, mock_executor: MockExecutor
    ) -> None:
        """When error_message is None, fallback text is used."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(
            ExecutionResult(success=False, output="", error_message=None)
        )

        result = await flow.resume_session(
            claude_session_id="sess-id",
            follow_up="query",
            session_id="db-id",
        )

        assert result.success is False
        assert "failed" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# execute_resume -- thin wrapper
# ---------------------------------------------------------------------------

class TestExecuteResume:
    """Test execute_resume() delegates to flow and returns dict."""

    @pytest.mark.asyncio
    async def test_delegates_to_flow(self, mock_executor: MockExecutor) -> None:
        """execute_resume calls flow.resume_session and returns to_dict()."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(_response_output("data"))

        result = await execute_resume(
            flow=flow,
            claude_session_id="claude-sess",
            follow_up="follow up",
            session_id="db-sess",
        )

        assert isinstance(result, dict)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_error_returns_dict(self, mock_executor: MockExecutor) -> None:
        """Error path also returns dict via to_dict()."""
        flow = _make_search_flow(mock_executor)
        mock_executor.add_result(ExecutionResult.from_error("boom"))

        result = await execute_resume(
            flow=flow,
            claude_session_id="sess",
            follow_up="q",
            session_id="db-id",
        )

        assert isinstance(result, dict)
        assert result["success"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# Server-level resume() -- validation
# ---------------------------------------------------------------------------

class TestServerResume:
    """Test server.resume() MCP tool validation logic."""

    def _parse_yaml(self, yaml_str: str) -> dict:
        """Parse YAML response back to dict."""
        return yaml.safe_load(yaml_str)

    @pytest.mark.asyncio
    async def test_l0_session_not_resumable(self) -> None:
        """Level 0 sessions are rejected."""
        from prism import server

        session = _make_db_session(level=0)
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=session)

        with (
            patch.object(server, "_session_repository", mock_repo),
            patch.object(server, "_search_flow", MagicMock()),
            patch.object(server, "_resolve_user_id", return_value="test-user"),
        ):
            result_yaml = await server.resume.fn(
                session_id=str(session.id),
                follow_up="follow up",
            )

        data = self._parse_yaml(result_yaml)
        assert data["success"] is False
        assert "Level 0" in data["error"]

    @pytest.mark.asyncio
    async def test_expired_session_rejected(self) -> None:
        """Sessions older than retention TTL are rejected."""
        from prism import server

        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        session = _make_db_session(created_at=old_date)
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=session)

        with (
            patch.object(server, "_session_repository", mock_repo),
            patch.object(server, "_search_flow", MagicMock()),
            patch.object(server, "_resolve_user_id", return_value="test-user"),
        ):
            result_yaml = await server.resume.fn(
                session_id=str(session.id),
                follow_up="follow up",
            )

        data = self._parse_yaml(result_yaml)
        assert data["success"] is False
        assert "expired" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_claude_session_id_rejected(self) -> None:
        """Sessions without claude_session_id are rejected."""
        from prism import server

        session = _make_db_session(claude_session_id=None)
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=session)

        with (
            patch.object(server, "_session_repository", mock_repo),
            patch.object(server, "_search_flow", MagicMock()),
            patch.object(server, "_resolve_user_id", return_value="test-user"),
        ):
            result_yaml = await server.resume.fn(
                session_id=str(session.id),
                follow_up="follow up",
            )

        data = self._parse_yaml(result_yaml)
        assert data["success"] is False
        assert "Claude session ID" in data["error"]

    @pytest.mark.asyncio
    async def test_session_not_found(self) -> None:
        """Non-existent session returns error."""
        from prism import server

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=None)

        session_id = str(uuid.uuid4())
        with (
            patch.object(server, "_session_repository", mock_repo),
            patch.object(server, "_search_flow", MagicMock()),
            patch.object(server, "_resolve_user_id", return_value="test-user"),
        ):
            result_yaml = await server.resume.fn(
                session_id=session_id,
                follow_up="follow up",
            )

        data = self._parse_yaml(result_yaml)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_session_id_format(self) -> None:
        """Non-UUID session_id returns error."""
        from prism import server

        with (
            patch.object(server, "_session_repository", MagicMock()),
            patch.object(server, "_search_flow", MagicMock()),
            patch.object(server, "_resolve_user_id", return_value="test-user"),
        ):
            result_yaml = await server.resume.fn(
                session_id="not-a-uuid",
                follow_up="follow up",
            )

        data = self._parse_yaml(result_yaml)
        assert data["success"] is False
        assert "Invalid session ID" in data["error"]

    @pytest.mark.asyncio
    async def test_valid_session_delegates_to_flow(self) -> None:
        """Valid resumable session delegates to execute_resume."""
        from prism import server

        session = _make_db_session(level=2, claude_session_id="claude-123")
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=session)

        mock_flow = MagicMock()
        mock_resume_result = SearchResult(
            success=True,
            content="resumed content",
            session_id=str(session.id),
            query="follow up",
            metadata={"resumed_from": "claude-123"},
        )
        with patch(
            "prism.server.execute_resume",
            new_callable=AsyncMock,
            return_value=mock_resume_result.to_dict(),
        ) as mock_exec_resume:
            with (
                patch.object(server, "_session_repository", mock_repo),
                patch.object(server, "_search_flow", mock_flow),
                patch.object(server, "_resolve_user_id", return_value="test-user"),
            ):
                result_yaml = await server.resume.fn(
                    session_id=str(session.id),
                    follow_up="follow up",
                )

            mock_exec_resume.assert_called_once_with(
                flow=mock_flow,
                claude_session_id="claude-123",
                follow_up="follow up",
                session_id=str(session.id),
            )

        data = self._parse_yaml(result_yaml)
        assert data["success"] is True
