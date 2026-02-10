"""
Cancel MCP tool.

Thin wrapper calling SessionRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.core.session import SessionRegistry


async def execute_cancel(
    registry: SessionRegistry,
    session_id: str,
) -> dict[str, Any]:
    """
    Cancel a running search session.

    Args:
        registry: SessionRegistry instance (injected)
        session_id: Session ID to cancel

    Returns:
        Dictionary with cancellation status
    """
    cancelled = await registry.cancel(session_id)

    if cancelled:
        return {
            "success": True,
            "message": f"Session {session_id} cancelled",
            "session_id": session_id,
        }

    return {
        "success": True,
        "message": f"Session {session_id} not found or already completed",
        "session_id": session_id,
    }
