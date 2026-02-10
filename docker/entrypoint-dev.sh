#!/bin/bash
set -e

# Install package in editable mode (source is mounted)
cd /app
uv pip install --system -e ".[dev]" 2>/dev/null || pip install -e ".[dev]"

# Run the server
exec python -m prism "$@"
