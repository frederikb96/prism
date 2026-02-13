#!/bin/bash
set -e

# Refresh editable link to mounted source (deps already in image)
cd /app
uv pip install --system --no-deps -e .

# Drop to appuser and run the server
exec gosu appuser python -m prism "$@"
