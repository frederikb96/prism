"""
Prism MCP Server entry point.

Run with: python -m prism
"""

from prism.config import get_config

from .server import mcp

if __name__ == "__main__":
    config = get_config()
    mcp.run(transport=config.server.transport, host="0.0.0.0", port=config.server.port)
