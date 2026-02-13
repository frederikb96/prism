#!/bin/bash
set -e

# Fix ownership of mounted volumes (podman creates as root)
chown -R appuser:appuser /home/appuser/.claude

# Refresh editable link to mounted source (deps already in image)
cd /app
uv pip install --system --no-deps -e .

# Run migrations as appuser (Alembic is the sole schema manager)
gosu appuser alembic upgrade head

# Drop to appuser and run the server
exec gosu appuser python -m prism "$@"
