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
    user_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    process: CancellableProcess | None = None
    is_active: bool = True
    children: set[str] = field(default_factory=set)
    parent_id: str | None = None

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
        parent_session_id: str | None = None,
        user_id: str = "",
    ) -> Session:
        """
        Register a new session.

        Args:
            session_id: Unique session identifier
            process: Optional process associated with session
            parent_session_id: Optional parent session to link to
            user_id: Owner of this session (inherited from parent if empty)

        Returns:
            The registered Session object
        """
        async with self._lock:
            resolved_user_id = user_id

            if parent_session_id is not None:
                parent = self._sessions.get(parent_session_id)
                if parent is not None:
                    parent.children.add(session_id)
                    if not resolved_user_id:
                        resolved_user_id = parent.user_id

            has_parent = parent_session_id and parent_session_id in self._sessions
            session = Session(
                session_id=session_id,
                user_id=resolved_user_id,
                process=process,
                parent_id=parent_session_id if has_parent else None,
            )

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

        Also removes the session from its parent's children set if linked.

        Args:
            session_id: Session to remove

        Returns:
            The removed Session if it existed, None otherwise
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.mark_complete()
                if session.parent_id and session.parent_id in self._sessions:
                    self._sessions[session.parent_id].children.discard(session_id)
            return session

    async def cancel(self, session_id: str) -> bool:
        """
        Cancel a session's process if running.

        If the session has no process but has children, cancels all child
        processes in parallel instead.

        Args:
            session_id: Session to cancel

        Returns:
            True if session was found and cancelled, False otherwise
        """
        # Collect process(es) to cancel under lock
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active:
                return False

            # Direct process on this session
            if session.process:
                to_cancel = [(session_id, session.process)]
            elif session.children:
                # Gather child processes
                to_cancel = []
                for cid in list(session.children):
                    child = self._sessions.get(cid)
                    if child and child.is_active and child.process:
                        to_cancel.append((cid, child.process))
                if not to_cancel:
                    return False
            else:
                return False

        # Cancel outside lock to avoid blocking other operations
        await asyncio.gather(*[proc.cancel() for _, proc in to_cancel])

        # Mark complete under lock
        async with self._lock:
            for sid, _ in to_cancel:
                if sid in self._sessions:
                    self._sessions[sid].mark_complete()
            if session_id in self._sessions:
                self._sessions[session_id].mark_complete()
        return True

    async def cancel_all(self, user_id: str | None = None) -> int:
        """
        Cancel active sessions, optionally scoped to a user.

        Args:
            user_id: If provided, only cancel sessions owned by this user.
                     If None, cancel all sessions (used for server shutdown).

        Returns:
            Number of sessions cancelled
        """
        # Collect processes under lock
        async with self._lock:
            to_cancel = [
                (sid, s.process)
                for sid, s in self._sessions.items()
                if s.is_active
                and s.process
                and (user_id is None or s.user_id == user_id)
            ]

        # Cancel all processes in parallel
        await asyncio.gather(*[process.cancel() for _, process in to_cancel])

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
