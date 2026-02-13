"""
Repository layer for database operations.

All queries are user_id scoped for multi-tenancy.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, desc, or_, select, update

from prism.database.models import (
    SearchSession,
    SearchTask,
    SessionStatus,
    TaskStatus,
)

if TYPE_CHECKING:
    from prism.database.connection import DatabaseConnection


class SearchSessionRepository:
    """
    Repository for SearchSession CRUD operations.

    All operations are scoped by user_id for multi-tenancy.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        """
        Initialize repository with database connection.

        Args:
            db: Database connection instance
        """
        self._db = db

    async def create(
        self,
        user_id: str,
        query: str,
        prompt: str,
        level: int,
        session_id: uuid.UUID | None = None,
    ) -> SearchSession:
        """
        Create a new search session.

        Args:
            user_id: User identifier
            query: Search query
            prompt: Full prompt sent to agents
            level: Search level (0-3)
            session_id: Optional pre-generated session ID

        Returns:
            Created SearchSession
        """
        session = SearchSession(
            id=session_id or uuid.uuid4(),
            user_id=user_id,
            query=query,
            prompt=prompt,
            level=level,
            status=SessionStatus.PENDING,
        )

        async with self._db.session() as db_session:
            db_session.add(session)
            await db_session.flush()
            await db_session.refresh(session)
            return session

    async def get(self, user_id: str, session_id: uuid.UUID) -> SearchSession | None:
        """
        Get a session by ID.

        Args:
            user_id: User identifier (for access control)
            session_id: Session UUID

        Returns:
            SearchSession if found and owned by user, None otherwise
        """
        async with self._db.session() as db_session:
            result = await db_session.execute(
                select(SearchSession).where(
                    SearchSession.id == session_id,
                    SearchSession.user_id == user_id,
                )
            )
            return result.scalar_one_or_none()

    async def get_by_claude_session(
        self,
        user_id: str,
        claude_session_id: str,
    ) -> SearchSession | None:
        """
        Get a session by Claude session ID (for resume).

        Args:
            user_id: User identifier
            claude_session_id: Claude CLI session ID

        Returns:
            SearchSession if found, None otherwise
        """
        async with self._db.session() as db_session:
            result = await db_session.execute(
                select(SearchSession).where(
                    SearchSession.claude_session_id == claude_session_id,
                    SearchSession.user_id == user_id,
                )
            )
            return result.scalar_one_or_none()

    async def update(
        self,
        session_id: uuid.UUID,
        user_id: str | None = None,
        *,
        status: SessionStatus | None = None,
        claude_session_id: str | None = None,
        summary: str | None = None,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
        completed_at: datetime | None = None,
        duration_ms: int | None = None,
    ) -> bool:
        """
        Update a session's fields.

        Args:
            session_id: Session to update
            user_id: User identifier (for access control scoping)
            status: New status
            claude_session_id: Claude CLI session ID
            summary: Agent-generated summary
            result: Full result data
            error_message: Error message if failed
            completed_at: Completion timestamp
            duration_ms: Total duration in milliseconds

        Returns:
            True if session was updated, False if not found
        """
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = status
        if claude_session_id is not None:
            values["claude_session_id"] = claude_session_id
        if summary is not None:
            values["summary"] = summary
        if result is not None:
            values["result"] = result
        if error_message is not None:
            values["error_message"] = error_message
        if completed_at is not None:
            values["completed_at"] = completed_at
        if duration_ms is not None:
            values["duration_ms"] = duration_ms

        if not values:
            return True

        async with self._db.session() as db_session:
            where_clause = SearchSession.id == session_id
            if user_id is not None:
                stmt = (
                    update(SearchSession)
                    .where(where_clause, SearchSession.user_id == user_id)
                    .values(**values)
                )
            else:
                stmt = (
                    update(SearchSession)
                    .where(where_clause)
                    .values(**values)
                )
            result_proxy = await db_session.execute(stmt)
            return bool(result_proxy.rowcount > 0)  # type: ignore[attr-defined]

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        search: str | None = None,
    ) -> list[SearchSession]:
        """
        List sessions for a user with optional fuzzy search.

        Args:
            user_id: User identifier
            limit: Maximum number of results
            offset: Number of results to skip
            search: Optional search term for summary/query

        Returns:
            List of SearchSession objects, newest first
        """
        async with self._db.session() as db_session:
            stmt = (
                select(SearchSession)
                .where(SearchSession.user_id == user_id)
                .order_by(desc(SearchSession.created_at))
                .limit(limit)
                .offset(offset)
            )

            if search:
                search_pattern = f"%{search}%"
                stmt = stmt.where(
                    or_(
                        SearchSession.summary.ilike(search_pattern),
                        SearchSession.query.ilike(search_pattern),
                    )
                )

            result = await db_session.execute(stmt)
            return list(result.scalars().all())

    async def delete_old_sessions(self, ttl_days: int) -> int:
        """
        Delete sessions older than the retention period.

        Args:
            ttl_days: Number of days to retain sessions

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)

        async with self._db.session() as db_session:
            stmt = delete(SearchSession).where(SearchSession.created_at < cutoff)
            result = await db_session.execute(stmt)
            return int(result.rowcount)  # type: ignore[attr-defined]


class SearchTaskRepository:
    """Repository for SearchTask CRUD operations."""

    def __init__(self, db: DatabaseConnection) -> None:
        """
        Initialize repository with database connection.

        Args:
            db: Database connection instance
        """
        self._db = db

    async def create(
        self,
        session_id: uuid.UUID,
        worker_type: str,
        worker_prompt: str,
    ) -> SearchTask:
        """
        Create a new worker task.

        Args:
            session_id: Parent session ID
            worker_type: Type of worker (perplexity, tavily, researcher)
            worker_prompt: Prompt sent to the worker

        Returns:
            Created SearchTask
        """
        task = SearchTask(
            session_id=session_id,
            worker_type=worker_type,
            worker_prompt=worker_prompt,
            status=TaskStatus.PENDING,
        )

        async with self._db.session() as db_session:
            db_session.add(task)
            await db_session.flush()
            await db_session.refresh(task)
            return task

    async def update(
        self,
        task_id: uuid.UUID,
        *,
        status: TaskStatus | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> bool:
        """
        Update a task's fields.

        Args:
            task_id: Task to update
            status: New status
            result: Task result data
            error: Error message if failed
            started_at: When task started
            completed_at: When task completed

        Returns:
            True if task was updated, False if not found
        """
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = status
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error
        if started_at is not None:
            values["started_at"] = started_at
        if completed_at is not None:
            values["completed_at"] = completed_at

        if not values:
            return True

        async with self._db.session() as db_session:
            stmt = update(SearchTask).where(SearchTask.id == task_id).values(**values)
            result_proxy = await db_session.execute(stmt)
            return bool(result_proxy.rowcount > 0)  # type: ignore[attr-defined]

    async def get_by_session(self, session_id: uuid.UUID) -> list[SearchTask]:
        """
        Get all tasks for a session.

        Args:
            session_id: Parent session ID

        Returns:
            List of SearchTask objects
        """
        async with self._db.session() as db_session:
            result = await db_session.execute(
                select(SearchTask).where(SearchTask.session_id == session_id)
            )
            return list(result.scalars().all())
