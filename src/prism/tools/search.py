"""
Search MCP tool.

Thin wrapper calling orchestrator.flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.orchestrator.flow import SearchFlow


async def execute_search(
    flow: SearchFlow,
    query: str,
    level: int = 0,
    providers: list[str] | None = None,
    *,
    user_id: str,
) -> dict[str, Any]:
    """
    Execute search and return MCP-compatible response.

    Args:
        flow: SearchFlow instance (injected)
        query: Search query
        level: Search depth (0-3)
        providers: L0 provider selection (None=config default)
        user_id: Requesting user's identifier

    Returns:
        Dictionary with search result
    """
    result = await flow.execute_search(
        query=query, level=level, providers=providers, user_id=user_id
    )
    return result.to_dict()


async def execute_resume(
    flow: SearchFlow,
    claude_session_id: str,
    follow_up: str,
    session_id: str,
    mode: str = "chat",
    level: int = 0,
) -> dict[str, Any]:
    """
    Resume a previous session and return MCP-compatible response.

    Args:
        flow: SearchFlow instance (injected)
        claude_session_id: Claude CLI session ID to resume
        follow_up: Follow-up question or search query
        session_id: Our DB session ID
        mode: "chat" for discussion, "search" for new search with workers
        level: Original session level (used for search mode agent allocation)

    Returns:
        Dictionary with resume result
    """
    if mode == "search":
        result = await flow.resume_with_search(
            claude_session_id=claude_session_id,
            follow_up=follow_up,
            session_id=session_id,
            level=level,
        )
    else:
        result = await flow.resume_session(
            claude_session_id=claude_session_id,
            follow_up=follow_up,
            session_id=session_id,
        )
    return result.to_dict()
