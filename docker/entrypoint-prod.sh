#!/bin/bash
set -e

# Construct DB URL from postgres secret if not already set
if [ -z "$PRISM_DATABASE_URL" ] && [ -f /run/secrets/postgres_password ]; then
    PG_PASS=$(cat /run/secrets/postgres_password)
    export PRISM_DATABASE_URL="postgresql+asyncpg://prism:${PG_PASS}@postgres:5432/prism"
fi

cd /app

# Fix ownership of mounted volumes (podman creates as root)
chown -R appuser:appuser /home/appuser/.claude

# Run migrations (Alembic is the sole schema manager)
gosu appuser alembic upgrade head

exec gosu appuser python -m prism "$@"
