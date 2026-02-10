"""
Prompt management for Prism agents.

Provides lazy-loading prompt registry for system prompts and templates.
"""

from prism.prompts.registry import PromptRegistry, PromptTemplate

__all__ = [
    "PromptRegistry",
    "PromptTemplate",
]
