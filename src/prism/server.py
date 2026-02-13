"""
Prism MCP Server.

FastMCP server with DI wiring for all components.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, AsyncIterator

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from prism.config import get_config
from prism.core import ClaudeExecutor, GeminiExecutor, RetryExecutor, SessionRegistry
from prism.database import (
    SearchSessionRepository,
    close_database,
    init_database,
)
from prism.mcp_serializer import serialize_response
from prism.orchestrator import SearchFlow, WorkerDispatcher
from prism.tools import execute_fetch, execute_resume, execute_search

logger = logging.getLogger(__name__)

USER_ID_HEADER = "x-user-id"
DEFAULT_USER_ID = "default"

# Global singletons (initialized in lifespan)
_session_registry: SessionRegistry | None = None
_search_flow: SearchFlow | None = None
_session_repository: SearchSessionRepository | None = None


def _get_session_registry() -> SessionRegistry:
    """Get the session registry singleton."""
    if _session_registry is None:
        raise RuntimeError("Server not initialized")
    return _session_registry


def _get_search_flow() -> SearchFlow:
    """Get the search flow singleton."""
    if _search_flow is None:
        raise RuntimeError("Server not initialized")
    return _search_flow


def _get_session_repository() -> SearchSessionRepository:
    """Get the session repository singleton."""
    if _session_repository is None:
        raise RuntimeError("Server not initialized")
    return _session_repository


def _resolve_user_id() -> str:
    """Extract user ID from request headers, falling back to default."""
    headers = get_http_headers() or {}
    return headers.get(USER_ID_HEADER) or DEFAULT_USER_ID


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """
    Lifespan context manager for MCP server.

    Initializes all components via DI on startup.
    """
    global _session_registry, _search_flow, _session_repository

    logger.info("Prism MCP server starting")

    config = get_config()

    # Initialize database
    db = await init_database(config.database)
    _session_repository = SearchSessionRepository(db)
    logger.info("Database initialized")

    # Initialize components via DI
    _session_registry = SessionRegistry()

    executor = ClaudeExecutor(session_registry=_session_registry)
    retry_executor = RetryExecutor(executor=executor)
    gemini_executor = GeminiExecutor(session_registry=_session_registry)

    dispatcher = WorkerDispatcher(
        claude_executor=retry_executor,
        gemini_executor=gemini_executor,
    )

    _search_flow = SearchFlow(
        retry_executor=retry_executor,
        gemini_executor=gemini_executor,
        dispatcher=dispatcher,
        session_registry=_session_registry,
        session_repository=_session_repository,
    )

    logger.info("Components initialized")

    yield

    # Cleanup
    logger.info("Prism MCP server shutting down")
    if _session_registry:
        cancelled = await _session_registry.cancel_all()
        if cancelled:
            logger.info("Cancelled %d active sessions", cancelled)

    await close_database()

    _session_registry = None
    _search_flow = None
    _session_repository = None


# Create FastMCP server
mcp = FastMCP(
    name="prism",
    instructions=(
        "Prism is a unified web search interface with level-based search depth.\n\n"
        "Search Levels:\n"
        "- Level 0: Instant (default) - direct worker call, supports provider selection\n"
        "- Level 1: Quick search (2-3 workers, ~60s)\n"
        "- Level 2: Normal search (4-6 workers, ~150s)\n"
        "- Level 3: Deep search (8-12 workers, ~600s)\n\n"
        "L0 Providers:\n"
        "- claude_search, tavily_search, perplexity_search, gemini_search\n"
        "- 'mix' = all 4 in parallel\n"
        "- Default (None) = claude_search only\n\n"
        "Tools:\n"
        "- search(query, level?, providers?): Execute search at specified depth\n"
        "- fetch(url): Extract content from a URL via Tavily (advanced extraction)\n"
        "- resume(session_id, follow_up, mode?): Resume a previous L1-L3 search. "
        'mode="chat" (default) discusses results, mode="search" launches new workers\n'
        "- get_session(session_id): Get session details\n"
        "- list_sessions(limit?, offset?, search?): List past sessions\n"
        "- cancel_all(): Cancel all running searches for the current user"
    ),
    lifespan=lifespan,
)


@mcp.tool()
async def search(
    query: Annotated[str, "Search query"],
    level: Annotated[int, "Search depth: 0=instant, 1=quick, 2=normal, 3=deep"] = 0,
    providers: Annotated[
        list[str] | None,
        'L0 provider selection. Options: "claude_search", "tavily_search", '
        '"perplexity_search", "gemini_search". Special: "mix" = all 4 in parallel. '
        "Any combination allowed. Default (None): claude_search only. Ignored for L1-L3.",
    ] = None,
) -> str:
    """
    Execute a search with the specified level of depth.

    Level 0 uses direct worker call (fast, supports provider selection).
    Level 1-3 use manager + workers for comprehensive search.

    Returns human-readable YAML with content and metadata.
    """
    flow = _get_search_flow()
    user_id = _resolve_user_id()
    result = await execute_search(
        flow=flow, query=query, level=level, providers=providers, user_id=user_id
    )
    return serialize_response(result)


@mcp.tool()
async def fetch(
    url: Annotated[str, "URL to extract content from"],
) -> str:
    """
    Extract content from a single URL via Tavily.

    Uses advanced extraction depth for thorough content retrieval.
    Returns raw page content, images, and metadata.
    """
    result = await execute_fetch(url=url)
    return serialize_response(result)


@mcp.tool()
async def cancel_all() -> str:
    """
    Cancel all running search sessions for the current user.

    Gracefully stops all active searches owned by the requesting user.
    Returns count of cancelled sessions.
    Idempotent - returns success even if no sessions were active.
    """
    registry = _get_session_registry()
    user_id = _resolve_user_id()
    cancelled_count = await registry.cancel_all(user_id=user_id)
    result = {
        "success": True,
        "message": f"Cancelled {cancelled_count} active sessions",
        "cancelled_count": cancelled_count,
    }
    return serialize_response(result)


@mcp.tool()
async def get_session(
    session_id: Annotated[str, "Session UUID"],
) -> str:
    """
    Get details of a search session.

    Returns session metadata, status, and results if completed.
    """
    repo = _get_session_repository()
    user_id = _resolve_user_id()

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        return serialize_response({
            "success": False,
            "error": f"Invalid session ID format: {session_id}",
        })

    session = await repo.get(user_id, session_uuid)

    if session is None:
        return serialize_response({
            "success": False,
            "error": f"Session not found: {session_id}",
        })

    result: dict[str, Any] = {
        "success": True,
        "session": {
            "id": str(session.id),
            "query": session.query,
            "level": session.level,
            "status": session.status.value,
            "summary": session.summary,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "duration_ms": session.duration_ms,
            "error": session.error_message,
            "resumable": session.claude_session_id is not None and session.level > 0,
        },
    }

    if session.result:
        result["session"]["result"] = session.result

    return serialize_response(result)


@mcp.tool()
async def list_sessions(
    limit: Annotated[int, "Maximum number of sessions to return"] = 20,
    offset: Annotated[int, "Number of sessions to skip"] = 0,
    search: Annotated[str | None, "Search term for summary/query"] = None,
) -> str:
    """
    List search sessions with optional filtering.

    Returns newest sessions first. Use search parameter for fuzzy matching.
    """
    repo = _get_session_repository()
    user_id = _resolve_user_id()

    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    sessions = await repo.list_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset,
        search=search,
    )

    items = []
    for session in sessions:
        summary = session.summary
        if summary and len(summary) > 150:
            summary = summary[:150] + "..."

        metadata = None
        if session.result and isinstance(session.result, dict):
            metadata = session.result.get("metadata")

        items.append({
            "id": str(session.id),
            "level": session.level,
            "status": session.status.value,
            "summary": summary,
            "created_at": session.created_at.isoformat(),
            "duration_ms": session.duration_ms,
            "error": session.error_message,
            "resumable": session.claude_session_id is not None and session.level > 0,
            "metadata": metadata,
        })

    return serialize_response({
        "success": True,
        "sessions": items,
        "count": len(items),
        "offset": offset,
        "limit": limit,
    })


@mcp.tool()
async def resume(
    session_id: Annotated[str, "Session UUID to resume"],
    follow_up: Annotated[str, "Follow-up question or instruction"],
    mode: Annotated[
        str,
        'Resume mode: "chat" (default) to discuss existing results without new searches, '
        'or "search" to launch a full follow-up search with new workers using the '
        "original session context.",
    ] = "chat",
) -> str:
    """
    Resume a previous L1-L3 search session with a follow-up.

    Two modes:
    - chat: Discuss/ask about existing results (no new web searches)
    - search: Launch a new orchestrated search informed by previous session context

    Only works for completed L1-L3 sessions within the retention period.
    Level 0 sessions are not resumable.
    """
    valid_modes = ("chat", "search")
    if mode not in valid_modes:
        return serialize_response({
            "success": False,
            "error": f"Invalid mode: {mode!r}. Must be one of: {', '.join(valid_modes)}",
        })

    repo = _get_session_repository()
    user_id = _resolve_user_id()
    config = get_config()

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        return serialize_response({
            "success": False,
            "error": f"Invalid session ID format: {session_id}",
        })

    session = await repo.get(user_id, session_uuid)

    if session is None:
        return serialize_response({
            "success": False,
            "error": f"Session not found: {session_id}",
        })

    if session.level == 0:
        return serialize_response({
            "success": False,
            "error": "Level 0 sessions are not resumable",
        })

    if not session.claude_session_id:
        return serialize_response({
            "success": False,
            "error": "Session has no Claude session ID for resume",
        })

    ttl_days = config.retention.session_ttl_days
    expiry_cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    if session.created_at < expiry_cutoff:
        return serialize_response({
            "success": False,
            "error": f"Session expired (older than {ttl_days} days)",
            "session_created": session.created_at.isoformat(),
        })

    flow = _get_search_flow()
    result = await execute_resume(
        flow=flow,
        claude_session_id=session.claude_session_id,
        follow_up=follow_up,
        session_id=str(session.id),
        mode=mode,
        level=session.level,
    )
    return serialize_response(result)
