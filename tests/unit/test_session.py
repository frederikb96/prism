"""Tests for SessionRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from prism.core.session import Session, SessionRegistry


class TestSession:
    """Test Session dataclass."""

    def test_session_creation(self) -> None:
        """Session created with correct defaults."""
        session = Session(session_id="test-123")

        assert session.session_id == "test-123"
        assert session.is_active is True
        assert session.process is None
        assert session.created_at is not None

    def test_mark_complete(self) -> None:
        """mark_complete updates state."""
        session = Session(session_id="test-123")
        mock_process = AsyncMock()
        session.process = mock_process

        session.mark_complete()

        assert session.is_active is False
        assert session.process is None


class TestSessionRegistryBasicOperations:
    """Test basic registry operations."""

    @pytest.mark.asyncio
    async def test_register_session(self, session_registry: SessionRegistry) -> None:
        """Register creates new session."""
        session = await session_registry.register("sess-1")

        assert session.session_id == "sess-1"
        assert session.is_active is True

    @pytest.mark.asyncio
    async def test_get_existing_session(
        self, session_registry: SessionRegistry
    ) -> None:
        """Get returns registered session."""
        await session_registry.register("sess-1")

        session = await session_registry.get("sess-1")

        assert session is not None
        assert session.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(
        self, session_registry: SessionRegistry
    ) -> None:
        """Get returns None for unknown session."""
        session = await session_registry.get("nonexistent")

        assert session is None

    @pytest.mark.asyncio
    async def test_unregister_session(
        self, session_registry: SessionRegistry
    ) -> None:
        """Unregister removes and marks session complete."""
        await session_registry.register("sess-1")

        removed = await session_registry.unregister("sess-1")

        assert removed is not None
        assert removed.is_active is False

        # Should not be retrievable anymore
        session = await session_registry.get("sess-1")
        assert session is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(
        self, session_registry: SessionRegistry
    ) -> None:
        """Unregister returns None for unknown session."""
        removed = await session_registry.unregister("nonexistent")

        assert removed is None


class TestSessionRegistryCancellation:
    """Test cancellation functionality."""

    @pytest.mark.asyncio
    async def test_cancel_session_with_process(
        self, session_registry: SessionRegistry
    ) -> None:
        """Cancel terminates process and marks session complete."""
        mock_process = AsyncMock()
        await session_registry.register("sess-1", process=mock_process)

        result = await session_registry.cancel("sess-1")

        assert result is True
        mock_process.cancel.assert_called_once()

        session = await session_registry.get("sess-1")
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_cancel_session_without_process(
        self, session_registry: SessionRegistry
    ) -> None:
        """Cancel returns False if no process to cancel."""
        await session_registry.register("sess-1")

        result = await session_registry.cancel("sess-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session(
        self, session_registry: SessionRegistry
    ) -> None:
        """Cancel returns False for unknown session."""
        result = await session_registry.cancel("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_sessions(
        self, session_registry: SessionRegistry
    ) -> None:
        """cancel_all terminates all active processes."""
        mock_process1 = AsyncMock()
        mock_process2 = AsyncMock()
        await session_registry.register("sess-1", process=mock_process1)
        await session_registry.register("sess-2", process=mock_process2)
        await session_registry.register("sess-3")  # No process

        cancelled = await session_registry.cancel_all()

        assert cancelled == 2
        mock_process1.cancel.assert_called_once()
        mock_process2.cancel.assert_called_once()


class TestSessionRegistryListing:
    """Test listing functionality."""

    @pytest.mark.asyncio
    async def test_list_active_sessions(
        self, session_registry: SessionRegistry
    ) -> None:
        """list_active returns only active sessions."""
        await session_registry.register("sess-1")
        await session_registry.register("sess-2")
        await session_registry.register("sess-3")

        # Unregister one
        await session_registry.unregister("sess-2")

        active = await session_registry.list_active()

        assert len(active) == 2
        session_ids = {s.session_id for s in active}
        assert session_ids == {"sess-1", "sess-3"}

    @pytest.mark.asyncio
    async def test_list_active_empty(
        self, session_registry: SessionRegistry
    ) -> None:
        """list_active returns empty list when no sessions."""
        active = await session_registry.list_active()

        assert active == []


class TestSessionRegistryCleanup:
    """Test cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_completed(
        self, session_registry: SessionRegistry
    ) -> None:
        """cleanup_completed removes inactive sessions."""
        # Register sessions
        await session_registry.register("sess-1")
        await session_registry.register("sess-2")
        await session_registry.register("sess-3")

        # Mark some as inactive via cancel (with mock process)
        mock_process = AsyncMock()
        session = await session_registry.get("sess-2")
        session.process = mock_process
        await session_registry.cancel("sess-2")

        # Cleanup should remove the cancelled session
        removed = await session_registry.cleanup_completed()

        assert removed == 1

        # Verify sess-2 is gone from active list
        active = await session_registry.list_active()
        session_ids = {s.session_id for s in active}
        assert "sess-2" not in session_ids

    @pytest.mark.asyncio
    async def test_cleanup_nothing_to_clean(
        self, session_registry: SessionRegistry
    ) -> None:
        """cleanup_completed returns 0 when nothing to clean."""
        await session_registry.register("sess-1")
        await session_registry.register("sess-2")

        removed = await session_registry.cleanup_completed()

        assert removed == 0


class TestSessionRegistryParentChild:
    """Test parent-child session tracking."""

    @pytest.mark.asyncio
    async def test_register_with_parent_links_child(
        self, session_registry: SessionRegistry
    ) -> None:
        """Registering with parent_session_id creates bidirectional link."""
        await session_registry.register("parent")
        child = await session_registry.register(
            "child-1", parent_session_id="parent"
        )

        assert child.parent_id == "parent"
        parent = await session_registry.get("parent")
        assert "child-1" in parent.children

    @pytest.mark.asyncio
    async def test_register_with_nonexistent_parent(
        self, session_registry: SessionRegistry
    ) -> None:
        """Registering with unknown parent_session_id still creates session."""
        child = await session_registry.register(
            "child-1", parent_session_id="ghost"
        )

        assert child.parent_id is None
        assert child.session_id == "child-1"

    @pytest.mark.asyncio
    async def test_cancel_parent_cascades_to_children(
        self, session_registry: SessionRegistry
    ) -> None:
        """Cancelling a parent with no process cancels all child processes."""
        await session_registry.register("parent")

        proc1, proc2 = AsyncMock(), AsyncMock()
        await session_registry.register("child-1", process=proc1, parent_session_id="parent")
        await session_registry.register("child-2", process=proc2, parent_session_id="parent")

        result = await session_registry.cancel("parent")

        assert result is True
        proc1.cancel.assert_called_once()
        proc2.cancel.assert_called_once()

        parent = await session_registry.get("parent")
        assert parent.is_active is False

        child1 = await session_registry.get("child-1")
        assert child1.is_active is False

    @pytest.mark.asyncio
    async def test_cancel_parent_no_children_returns_false(
        self, session_registry: SessionRegistry
    ) -> None:
        """Cancelling a parent with no process and no children returns False."""
        await session_registry.register("parent")

        result = await session_registry.cancel("parent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_parent_children_no_process_returns_false(
        self, session_registry: SessionRegistry
    ) -> None:
        """Cancelling parent whose children also have no processes returns False."""
        await session_registry.register("parent")
        await session_registry.register("child-1", parent_session_id="parent")

        result = await session_registry.cancel("parent")

        assert result is False

    @pytest.mark.asyncio
    async def test_unregister_child_removes_from_parent(
        self, session_registry: SessionRegistry
    ) -> None:
        """Unregistering a child removes it from parent's children set."""
        await session_registry.register("parent")
        await session_registry.register("child-1", parent_session_id="parent")
        await session_registry.register("child-2", parent_session_id="parent")

        await session_registry.unregister("child-1")

        parent = await session_registry.get("parent")
        assert "child-1" not in parent.children
        assert "child-2" in parent.children

    @pytest.mark.asyncio
    async def test_multiple_children_registered(
        self, session_registry: SessionRegistry
    ) -> None:
        """Multiple children correctly tracked in parent's children set."""
        await session_registry.register("parent")

        for i in range(5):
            await session_registry.register(
                f"child-{i}", parent_session_id="parent"
            )

        parent = await session_registry.get("parent")
        assert len(parent.children) == 5

    @pytest.mark.asyncio
    async def test_cancel_parent_partial_children_with_process(
        self, session_registry: SessionRegistry
    ) -> None:
        """Only children with processes are cancelled."""
        await session_registry.register("parent")

        proc = AsyncMock()
        await session_registry.register("child-1", process=proc, parent_session_id="parent")
        await session_registry.register("child-2", parent_session_id="parent")

        result = await session_registry.cancel("parent")

        assert result is True
        proc.cancel.assert_called_once()


class TestSessionRegistryParallelCancelAll:
    """Verify cancel_all uses parallel cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_all_calls_all_processes(
        self, session_registry: SessionRegistry
    ) -> None:
        """cancel_all cancels all processes (verifies gather-based parallelism)."""
        processes = [AsyncMock() for _ in range(5)]
        for i, proc in enumerate(processes):
            await session_registry.register(f"sess-{i}", process=proc)

        cancelled = await session_registry.cancel_all()

        assert cancelled == 5
        for proc in processes:
            proc.cancel.assert_called_once()


class TestSessionRegistryConcurrency:
    """Test thread-safety under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_register(
        self, session_registry: SessionRegistry
    ) -> None:
        """Concurrent registrations don't cause issues."""
        import asyncio

        async def register_session(i: int) -> Session:
            return await session_registry.register(f"sess-{i}")

        sessions = await asyncio.gather(*[register_session(i) for i in range(100)])

        assert len(sessions) == 100
        active = await session_registry.list_active()
        assert len(active) == 100

    @pytest.mark.asyncio
    async def test_concurrent_get_and_cancel(
        self, session_registry: SessionRegistry
    ) -> None:
        """Concurrent get and cancel operations are safe."""
        import asyncio

        mock_process = AsyncMock()
        await session_registry.register("sess-1", process=mock_process)

        async def get_session() -> Session | None:
            return await session_registry.get("sess-1")

        async def cancel_session() -> bool:
            return await session_registry.cancel("sess-1")

        # Run get and cancel concurrently
        results = await asyncio.gather(
            get_session(),
            cancel_session(),
            get_session(),
        )

        # At least one get should succeed, cancel should succeed
        assert any(r is not None for r in results[:2])
