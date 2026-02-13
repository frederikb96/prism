"""
Prompt registry with lazy loading.

Loads prompts from files only when first accessed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base path for prompt files
PROMPTS_DIR = Path(__file__).parent


@dataclass
class PromptTemplate:
    """
    A loaded prompt template.

    Attributes:
        name: Template name/identifier
        content: The prompt content
        metadata: Additional metadata from the template
    """

    name: str
    content: str
    metadata: dict[str, Any] | None = None


class PromptRegistry:
    """
    Registry for prompt templates with lazy loading.

    Prompts are loaded from disk only when first accessed,
    then cached for subsequent requests.

    Usage:
        registry = PromptRegistry()
        system_prompt = registry.get("search_manager/system")
        level_prompt = registry.get("search_manager/level_2")
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        """
        Initialize registry.

        Args:
            prompts_dir: Base directory for prompts (defaults to module dir)
        """
        self._prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: dict[str, PromptTemplate] = {}

    def get(self, name: str) -> PromptTemplate | None:
        """
        Get a prompt template by name.

        Names are slash-separated paths, e.g.:
        - "search_manager/system" -> search_manager/system.md
        - "claude_researcher/system" -> claude_researcher/system.md
        - "perplexity_worker" -> perplexity_worker.md

        Args:
            name: Template name/path

        Returns:
            PromptTemplate if found, None otherwise
        """
        if name in self._cache:
            return self._cache[name]

        template = self._load_template(name)
        if template:
            self._cache[name] = template

        return template

    def get_content(self, name: str) -> str | None:
        """
        Get just the content of a prompt template.

        Convenience method when you only need the content string.

        Args:
            name: Template name/path

        Returns:
            Content string if found, None otherwise
        """
        template = self.get(name)
        return template.content if template else None

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """
        Get a JSON schema by name.

        Args:
            name: Schema name/path (e.g., "search_manager/task_schema")

        Returns:
            Parsed JSON schema if found, None otherwise
        """
        path = self._resolve_path(name, extension=".json")
        if not path or not path.exists():
            return None

        try:
            result: dict[str, Any] | None = json.loads(path.read_text())
            return result
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load schema %s: %s", name, e)
            return None

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()

    def _load_template(self, name: str) -> PromptTemplate | None:
        """
        Load a template from disk.

        Args:
            name: Template name/path

        Returns:
            PromptTemplate if found, None otherwise
        """
        path = self._resolve_path(name, extension=".md")
        if not path or not path.exists():
            logger.warning("Prompt template not found: %s", name)
            return None

        try:
            content = path.read_text()
            return PromptTemplate(name=name, content=content)
        except OSError as e:
            logger.warning("Failed to load prompt %s: %s", name, e)
            return None

    def _resolve_path(
        self,
        name: str,
        extension: str = ".md",
    ) -> Path | None:
        """
        Resolve a template name to a file path.

        Args:
            name: Template name (e.g., "search_manager/system")
            extension: File extension

        Returns:
            Resolved Path or None if invalid
        """
        # Convert name to path
        parts = name.split("/")
        if not parts:
            return None

        # Build path
        path = self._prompts_dir
        for part in parts[:-1]:
            path = path / part

        # Add extension to last part
        filename = parts[-1]
        if not filename.endswith(extension):
            filename = f"{filename}{extension}"

        path = path / filename

        # Security: ensure path is within prompts_dir
        try:
            path.resolve().relative_to(self._prompts_dir.resolve())
        except ValueError:
            logger.warning("Path traversal attempt: %s", name)
            return None

        return path

    def build_system_prompt(self, worker_type: str) -> str:
        """
        Compose system prompt with worker-specific tool section injected.

        Loads system.md template and replaces {worker_section} with
        the content from workers/<worker_type>.md.

        Args:
            worker_type: Worker type identifier (e.g., "websearch", "tavily")

        Returns:
            Composed system prompt string

        Raises:
            RuntimeError: If system.md or worker section file not found
        """
        template = self.get_content("system")
        if template is None:
            raise RuntimeError("System prompt template not found: system.md")

        worker_section = self.get_content(f"workers/{worker_type}")
        if worker_section is None:
            raise RuntimeError(f"Worker section not found: workers/{worker_type}.md")

        return template.replace("{worker_section}", worker_section.strip())

    def build_user_prompt(self, query: str, timeout_seconds: int) -> str:
        """
        Compose user prompt with query and timeout substituted.

        Loads user.md template and replaces {query} and {timeout_seconds}.

        Args:
            query: The search query
            timeout_seconds: Tool budget in seconds

        Returns:
            Composed user prompt string

        Raises:
            RuntimeError: If user.md template not found
        """
        template = self.get_content("user")
        if template is None:
            raise RuntimeError("User prompt template not found: user.md")

        return template.format(query=query, timeout_seconds=timeout_seconds)

    def list_templates(self, pattern: str = "**/*.md") -> list[str]:
        """
        List available templates.

        Args:
            pattern: Glob pattern for matching templates

        Returns:
            List of template names
        """
        templates = []
        for path in self._prompts_dir.glob(pattern):
            # Convert path to name
            relative = path.relative_to(self._prompts_dir)
            name = str(relative.with_suffix("")).replace("\\", "/")
            templates.append(name)

        return sorted(templates)
