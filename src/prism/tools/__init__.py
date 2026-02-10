"""
MCP tool implementations.

Thin wrappers that delegate to orchestrator.
"""

from prism.tools.cancel import execute_cancel
from prism.tools.search import execute_search

__all__ = [
    "execute_search",
    "execute_cancel",
]
