"""
Prism MCP Server entry point.

Run with: python -m prism
"""

if __name__ == "__main__":
    from prism.config import get_config

    config = get_config()

    from prism.core.logging import setup_logging

    setup_logging(config.server.log_level)

    from prism.server import mcp

    mcp.run(transport=config.server.transport, host="0.0.0.0", port=config.server.port)
