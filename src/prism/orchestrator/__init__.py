"""
Orchestration layer for Prism search.

Coordinates workers, dispatches tasks, and synthesizes results.
"""

from prism.orchestrator.dispatcher import WorkerDispatcher
from prism.orchestrator.flow import SearchFlow, SearchResult

__all__ = [
    "WorkerDispatcher",
    "SearchFlow",
    "SearchResult",
]
