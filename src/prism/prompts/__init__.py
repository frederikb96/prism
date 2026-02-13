"""
Prompt management for Prism agents.

Provides lazy-loading prompt registry for system prompts and templates.
"""

from prism.prompts.registry import PromptRegistry, PromptTemplate

_shared_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Get the shared PromptRegistry singleton."""
    global _shared_registry
    if _shared_registry is None:
        _shared_registry = PromptRegistry()
    return _shared_registry


__all__ = [
    "PromptRegistry",
    "PromptTemplate",
    "get_registry",
]
