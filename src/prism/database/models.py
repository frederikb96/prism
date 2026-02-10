"""
SQLAlchemy ORM models for Prism database.

Defines SearchSession and SearchTask tables for session persistence.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class SessionStatus(enum.Enum):
    """Status of a search session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(enum.Enum):
    """Status of a worker task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class SearchSession(Base):
    """
    Persisted search session.

    Stores query, results, and metadata for session resume and history.
    """

    __tablename__ = "search_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    claude_session_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, native_enum=False),
        nullable=False,
        default=SessionStatus.PENDING,
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    tasks: Mapped[list[SearchTask]] = relationship(
        "SearchTask",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_sessions_user_created", "user_id", "created_at"),
        Index("idx_sessions_claude_id", "claude_session_id"),
        Index("idx_sessions_status", "status"),
    )


class SearchTask(Base):
    """
    Worker task within a search session.

    Tracks individual worker executions for debugging and analytics.
    """

    __tablename__ = "search_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("search_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    worker_type: Mapped[str] = mapped_column(String(50), nullable=False)
    worker_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False),
        nullable=False,
        default=TaskStatus.PENDING,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    session: Mapped[SearchSession] = relationship(
        "SearchSession",
        back_populates="tasks",
    )

    __table_args__ = (Index("idx_tasks_session", "session_id"),)
