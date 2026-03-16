#!/bin/bash
set -e

# Accept DATABASE_URL (standard convention, e.g., CloudNativePG) as fallback
if [ -n "$DATABASE_URL" ] && [ -z "$PRISM_DATABASE_URL" ]; then
    export PRISM_DATABASE_URL="$DATABASE_URL"
fi

# Construct DB URL from Docker secrets if not already set (local podman compose)
if [ -z "$PRISM_DATABASE_URL" ] && [ -f /run/secrets/postgres_password ]; then
    PG_PASS=$(cat /run/secrets/postgres_password)
    export PRISM_DATABASE_URL="postgresql+asyncpg://prism:${PG_PASS}@postgres:5432/prism"
fi

cd /app

# Fix ownership of mounted volumes (podman rootless creates as root)
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser /home/appuser/.claude 2>/dev/null || true
fi

# Run migrations unless SKIP_MIGRATIONS is set (K8s uses init container)
if [ -z "$SKIP_MIGRATIONS" ]; then
    if [ "$(id -u)" = "0" ]; then
        gosu appuser alembic upgrade head
    else
        alembic upgrade head
    fi
fi

# Start server
if [ "$(id -u)" = "0" ]; then
    exec gosu appuser python -m prism "$@"
else
    exec python -m prism "$@"
fi
