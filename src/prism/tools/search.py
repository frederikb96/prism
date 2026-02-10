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
    level: int = 1,
) -> dict[str, Any]:
    """
    Execute search and return MCP-compatible response.

    Args:
        flow: SearchFlow instance (injected)
        query: Search query
        level: Search depth (0-3, default 1)

    Returns:
        Dictionary with search result
    """
    result = await flow.execute_search(query=query, level=level)
    return result.to_dict()
