"""
MCP tool implementations.

Thin wrappers that delegate to orchestrator.
"""

from prism.tools.fetch import execute_fetch
from prism.tools.search import execute_resume, execute_search

__all__ = [
    "execute_fetch",
    "execute_resume",
    "execute_search",
]
