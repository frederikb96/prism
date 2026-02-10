"""
Session tracking for Claude CLI executions.

Thread-safe session registry using asyncio.Lock.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.core.process import CancellableProcess


@dataclass
class Session:
    """
    Tracks an active Claude CLI session.

    Sessions enable --resume for validation retries.
    """

    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    process: CancellableProcess | None = None
    is_active: bool = True

    def mark_complete(self) -> None:
        """Mark session as no longer active."""
        self.is_active = False
        self.process = None


class SessionRegistry:
    """
    Thread-safe registry of active sessions.

    Enables session lookup for cancellation and resumption.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        session_id: str,
        process: CancellableProcess | None = None,
    ) -> Session:
        """
        Register a new session.

        Args:
            session_id: Unique session identifier
            process: Optional process associated with session

        Returns:
            The registered Session object
        """
        async with self._lock:
            session = Session(session_id=session_id, process=process)
            self._sessions[session_id] = session
            return session

    async def get(self, session_id: str) -> Session | None:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier to look up

        Returns:
            Session if found, None otherwise
        """
        async with self._lock:
            return self._sessions.get(session_id)

    async def unregister(self, session_id: str) -> Session | None:
        """
        Remove a session from the registry.

        Args:
            session_id: Session to remove

        Returns:
            The removed Session if it existed, None otherwise
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.mark_complete()
            return session

    async def cancel(self, session_id: str) -> bool:
        """
        Cancel a session's process if running.

        Args:
            session_id: Session to cancel

        Returns:
            True if session was found and cancelled, False otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not (session and session.is_active and session.process):
                return False
            process = session.process

        # Cancel outside lock to avoid blocking other operations
        await process.cancel()

        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].mark_complete()
        return True

    async def cancel_all(self) -> int:
        """
        Cancel all active sessions.

        Returns:
            Number of sessions cancelled
        """
        # Collect processes under lock
        async with self._lock:
            to_cancel = [
                (sid, s.process)
                for sid, s in self._sessions.items()
                if s.is_active and s.process
            ]

        # Cancel outside lock
        for _, process in to_cancel:
            await process.cancel()

        # Mark complete under lock
        async with self._lock:
            for sid, _ in to_cancel:
                if sid in self._sessions:
                    self._sessions[sid].mark_complete()

        return len(to_cancel)

    async def list_active(self) -> list[Session]:
        """
        Get all active sessions.

        Returns:
            List of active Session objects
        """
        async with self._lock:
            return [s for s in self._sessions.values() if s.is_active]

    async def cleanup_completed(self) -> int:
        """
        Remove all completed sessions from registry.

        Returns:
            Number of sessions removed
        """
        async with self._lock:
            completed = [sid for sid, s in self._sessions.items() if not s.is_active]
            for sid in completed:
                del self._sessions[sid]
            return len(completed)
