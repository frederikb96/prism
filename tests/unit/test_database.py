"""Tests for database module.

Uses async SQLite in-memory for fast, self-contained testing.
Real PostgreSQL coverage happens in E2E tests via docker-compose.dev.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from prism.config import DatabaseConfig
from prism.database import (
    Base,
    DatabaseConnection,
    SearchSession,
    SearchSessionRepository,
    SearchTaskRepository,
    SessionStatus,
    TaskStatus,
)

SQLITE_URL = "sqlite+aiosqlite://"
SQLITE_CONFIG = DatabaseConfig(url=SQLITE_URL, pool_size=1, max_overflow=0)


@pytest_asyncio.fixture
async def db() -> AsyncIterator[DatabaseConnection]:
    """In-memory SQLite database via DatabaseConnection-compatible wrapper."""
    engine = create_async_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    connection = DatabaseConnection.__new__(DatabaseConnection)
    connection._config = SQLITE_CONFIG
    connection._engine = engine
    connection._session_factory = session_factory

    yield connection
    await engine.dispose()


@pytest.fixture
def session_repo(db: DatabaseConnection) -> SearchSessionRepository:
    """Session repository backed by in-memory SQLite."""
    return SearchSessionRepository(db)


@pytest.fixture
def task_repo(db: DatabaseConnection) -> SearchTaskRepository:
    """Task repository backed by in-memory SQLite."""
    return SearchTaskRepository(db)


class TestDatabaseConnection:
    """Test DatabaseConnection lifecycle."""

    def test_engine_before_init_raises(self) -> None:
        """Accessing engine before init raises RuntimeError."""
        db = DatabaseConnection(SQLITE_CONFIG)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = db.engine

    def test_session_factory_before_init_raises(self) -> None:
        """Accessing session_factory before init raises RuntimeError."""
        db = DatabaseConnection(SQLITE_CONFIG)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = db.session_factory

    @pytest.mark.asyncio
    async def test_health_check_success(self, db: DatabaseConnection) -> None:
        """health_check returns True when database is accessible."""
        result = await db.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_session_context_commits_on_success(
        self, db: DatabaseConnection
    ) -> None:
        """Session context commits on successful exit."""
        async with db.session() as session:
            new_session = SearchSession(
                user_id="test-user",
                query="test query",
                prompt="test prompt",
                level=1,
            )
            session.add(new_session)
            await session.flush()
            session_id = new_session.id

        async with db.session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(SearchSession).where(SearchSession.id == session_id)
            )
            found = result.scalar_one_or_none()

        assert found is not None
        assert found.query == "test query"

    @pytest.mark.asyncio
    async def test_session_context_rollback_on_error(
        self, db: DatabaseConnection
    ) -> None:
        """Session context rolls back on exception."""
        session_id = uuid.uuid4()

        with pytest.raises(ValueError):
            async with db.session() as session:
                new_session = SearchSession(
                    id=session_id,
                    user_id="test-user",
                    query="test query",
                    prompt="test prompt",
                    level=1,
                )
                session.add(new_session)
                raise ValueError("Intentional error")

        async with db.session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(SearchSession).where(SearchSession.id == session_id)
            )
            found = result.scalar_one_or_none()

        assert found is None


class TestSearchSessionRepository:
    """Test SearchSessionRepository CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_session(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """create() returns a valid session with defaults."""
        session = await session_repo.create(
            user_id="user-1",
            query="What is Python?",
            prompt="Full prompt here",
            level=1,
        )

        assert session.id is not None
        assert session.user_id == "user-1"
        assert session.query == "What is Python?"
        assert session.prompt == "Full prompt here"
        assert session.level == 1
        assert session.status == SessionStatus.PENDING
        assert session.created_at is not None
        assert session.updated_at is not None

    @pytest.mark.asyncio
    async def test_create_with_custom_id(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """create() accepts a pre-generated session ID."""
        custom_id = uuid.uuid4()

        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=0,
            session_id=custom_id,
        )

        assert session.id == custom_id

    @pytest.mark.asyncio
    async def test_get_existing_session(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """get() returns session for correct user."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )

        found = await session_repo.get("user-1", created.id)

        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_get_wrong_user_returns_none(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """get() returns None if user doesn't match."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )

        found = await session_repo.get("user-2", created.id)

        assert found is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """get() returns None for unknown ID."""
        found = await session_repo.get("user-1", uuid.uuid4())

        assert found is None

    @pytest.mark.asyncio
    async def test_update_status(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """update() changes status field."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )

        result = await session_repo.update(
            created.id,
            "user-1",
            status=SessionStatus.RUNNING,
        )

        assert result is True

        found = await session_repo.get("user-1", created.id)
        assert found.status == SessionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_update_multiple_fields(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """update() can set multiple fields at once."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        completed_at = datetime.now(timezone.utc)

        result = await session_repo.update(
            created.id,
            "user-1",
            status=SessionStatus.COMPLETED,
            summary="Test summary",
            result={"content": "Answer", "success": True},
            completed_at=completed_at,
            duration_ms=1500,
        )

        assert result is True

        found = await session_repo.get("user-1", created.id)
        assert found.status == SessionStatus.COMPLETED
        assert found.summary == "Test summary"
        assert found.result == {"content": "Answer", "success": True}
        assert found.duration_ms == 1500

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_false(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """update() returns False for unknown session."""
        result = await session_repo.update(
            uuid.uuid4(),
            "user-1",
            status=SessionStatus.COMPLETED,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_empty_values_returns_true(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """update() with no values returns True (no-op)."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )

        result = await session_repo.update(created.id, "user-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_by_claude_session(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """get_by_claude_session() finds by Claude session ID."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        await session_repo.update(
            created.id,
            "user-1",
            claude_session_id="claude-sess-123",
        )

        found = await session_repo.get_by_claude_session("user-1", "claude-sess-123")

        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_get_by_claude_session_wrong_user(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """get_by_claude_session() returns None for wrong user."""
        created = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        await session_repo.update(
            created.id,
            "user-1",
            claude_session_id="claude-sess-123",
        )

        found = await session_repo.get_by_claude_session("user-2", "claude-sess-123")

        assert found is None

    @pytest.mark.asyncio
    async def test_list_sessions_returns_newest_first(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """list_sessions() orders by created_at descending."""
        await session_repo.create(
            user_id="user-1",
            query="first",
            prompt="p1",
            level=1,
        )
        await session_repo.create(
            user_id="user-1",
            query="second",
            prompt="p2",
            level=1,
        )
        await session_repo.create(
            user_id="user-1",
            query="third",
            prompt="p3",
            level=1,
        )

        sessions = await session_repo.list_sessions("user-1")

        assert len(sessions) == 3
        assert sessions[0].query == "third"
        assert sessions[2].query == "first"

    @pytest.mark.asyncio
    async def test_list_sessions_filters_by_user(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """list_sessions() only returns sessions for specified user."""
        await session_repo.create(
            user_id="user-1",
            query="user1 query",
            prompt="p1",
            level=1,
        )
        await session_repo.create(
            user_id="user-2",
            query="user2 query",
            prompt="p2",
            level=1,
        )

        sessions = await session_repo.list_sessions("user-1")

        assert len(sessions) == 1
        assert sessions[0].query == "user1 query"

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """list_sessions() respects limit parameter."""
        for i in range(5):
            await session_repo.create(
                user_id="user-1",
                query=f"query {i}",
                prompt=f"p{i}",
                level=1,
            )

        sessions = await session_repo.list_sessions("user-1", limit=3)

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_with_offset(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """list_sessions() respects offset parameter."""
        for i in range(5):
            await session_repo.create(
                user_id="user-1",
                query=f"query {i}",
                prompt=f"p{i}",
                level=1,
            )

        sessions = await session_repo.list_sessions("user-1", offset=2)

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_with_search(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """list_sessions() filters by search term."""
        await session_repo.create(
            user_id="user-1",
            query="Python programming",
            prompt="p1",
            level=1,
        )
        await session_repo.create(
            user_id="user-1",
            query="JavaScript development",
            prompt="p2",
            level=1,
        )

        sessions = await session_repo.list_sessions("user-1", search="Python")

        assert len(sessions) == 1
        assert "Python" in sessions[0].query

    @pytest.mark.asyncio
    async def test_list_sessions_search_matches_summary(
        self, session_repo: SearchSessionRepository
    ) -> None:
        """list_sessions() search also matches summary field."""
        created = await session_repo.create(
            user_id="user-1",
            query="generic query",
            prompt="p1",
            level=1,
        )
        await session_repo.update(
            created.id,
            "user-1",
            summary="Rust programming language overview",
        )

        sessions = await session_repo.list_sessions("user-1", search="Rust")

        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_delete_old_sessions(
        self, session_repo: SearchSessionRepository, db: DatabaseConnection
    ) -> None:
        """delete_old_sessions() removes sessions older than TTL."""
        old_time = datetime.now(timezone.utc) - timedelta(days=40)
        recent_time = datetime.now(timezone.utc) - timedelta(days=10)

        async with db.session() as session:
            old_session = SearchSession(
                user_id="user-1",
                query="old query",
                prompt="old",
                level=1,
            )
            old_session.created_at = old_time
            session.add(old_session)

            recent_session = SearchSession(
                user_id="user-1",
                query="recent query",
                prompt="recent",
                level=1,
            )
            recent_session.created_at = recent_time
            session.add(recent_session)

        deleted_count = await session_repo.delete_old_sessions(ttl_days=30)

        assert deleted_count == 1

        sessions = await session_repo.list_sessions("user-1")
        assert len(sessions) == 1
        assert sessions[0].query == "recent query"


class TestSearchTaskRepository:
    """Test SearchTaskRepository CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_task(
        self, task_repo: SearchTaskRepository, session_repo: SearchSessionRepository
    ) -> None:
        """create() returns a valid task."""
        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )

        task = await task_repo.create(
            session_id=session.id,
            worker_type="perplexity",
            worker_prompt="Search for X",
        )

        assert task.id is not None
        assert task.session_id == session.id
        assert task.worker_type == "perplexity"
        assert task.worker_prompt == "Search for X"
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_update_task_status(
        self, task_repo: SearchTaskRepository, session_repo: SearchSessionRepository
    ) -> None:
        """update() changes task status."""
        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        task = await task_repo.create(
            session_id=session.id,
            worker_type="tavily",
            worker_prompt="Search for Y",
        )

        result = await task_repo.update(
            task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        assert result is True

        tasks = await task_repo.get_by_session(session.id)
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.RUNNING
        assert tasks[0].started_at is not None

    @pytest.mark.asyncio
    async def test_update_task_with_result(
        self, task_repo: SearchTaskRepository, session_repo: SearchSessionRepository
    ) -> None:
        """update() can set JSON result."""
        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        task = await task_repo.create(
            session_id=session.id,
            worker_type="researcher",
            worker_prompt="Investigate Z",
        )

        result_data = {"content": "Found information", "sources": ["url1", "url2"]}
        result = await task_repo.update(
            task.id,
            status=TaskStatus.COMPLETED,
            result=result_data,
            completed_at=datetime.now(timezone.utc),
        )

        assert result is True

        tasks = await task_repo.get_by_session(session.id)
        assert tasks[0].result == result_data

    @pytest.mark.asyncio
    async def test_update_task_with_error(
        self, task_repo: SearchTaskRepository, session_repo: SearchSessionRepository
    ) -> None:
        """update() can set error message."""
        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        task = await task_repo.create(
            session_id=session.id,
            worker_type="perplexity",
            worker_prompt="Search",
        )

        result = await task_repo.update(
            task.id,
            status=TaskStatus.FAILED,
            error="API rate limited",
        )

        assert result is True

        tasks = await task_repo.get_by_session(session.id)
        assert tasks[0].status == TaskStatus.FAILED
        assert tasks[0].error == "API rate limited"

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_false(
        self, task_repo: SearchTaskRepository
    ) -> None:
        """update() returns False for unknown task."""
        result = await task_repo.update(
            uuid.uuid4(),
            status=TaskStatus.COMPLETED,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_by_session_returns_all_tasks(
        self, task_repo: SearchTaskRepository, session_repo: SearchSessionRepository
    ) -> None:
        """get_by_session() returns all tasks for session."""
        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )

        await task_repo.create(
            session_id=session.id,
            worker_type="perplexity",
            worker_prompt="Task 1",
        )
        await task_repo.create(
            session_id=session.id,
            worker_type="tavily",
            worker_prompt="Task 2",
        )
        await task_repo.create(
            session_id=session.id,
            worker_type="researcher",
            worker_prompt="Task 3",
        )

        tasks = await task_repo.get_by_session(session.id)

        assert len(tasks) == 3
        worker_types = {t.worker_type for t in tasks}
        assert worker_types == {"perplexity", "tavily", "researcher"}

    @pytest.mark.asyncio
    async def test_get_by_session_empty_for_nonexistent(
        self, task_repo: SearchTaskRepository
    ) -> None:
        """get_by_session() returns empty list for unknown session."""
        tasks = await task_repo.get_by_session(uuid.uuid4())

        assert tasks == []


class TestSessionTaskCascade:
    """Test cascade delete behavior."""

    @pytest.mark.asyncio
    async def test_delete_session_deletes_tasks(
        self,
        session_repo: SearchSessionRepository,
        task_repo: SearchTaskRepository,
        db: DatabaseConnection,
    ) -> None:
        """Deleting a session cascades to tasks."""
        session = await session_repo.create(
            user_id="user-1",
            query="test",
            prompt="test",
            level=1,
        )
        session_id = session.id

        await task_repo.create(
            session_id=session_id,
            worker_type="perplexity",
            worker_prompt="Task",
        )

        async with db.session() as sess:
            from sqlalchemy import delete

            await sess.execute(
                delete(SearchSession).where(SearchSession.id == session_id)
            )

        tasks = await task_repo.get_by_session(session_id)
        assert tasks == []
