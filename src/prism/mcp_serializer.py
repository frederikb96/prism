"""
MCP Response Serialization for Prism.

YAML serialization for human-readable MCP tool responses.
Matches Engram's format for consistency.
"""

from __future__ import annotations

from typing import Any

import yaml


class _BlockStyleDumper(yaml.SafeDumper):
    """Custom YAML dumper that uses block style for multiline strings."""

    pass


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Use literal block style (|) for multiline strings."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockStyleDumper.add_representer(str, _str_representer)


def serialize_response(data: dict[str, Any]) -> str:
    """
    Convert dict to YAML string for MCP response.

    Uses block style for multiline strings for human readability.
    """
    return yaml.dump(
        data,
        Dumper=_BlockStyleDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=1000,
    )
