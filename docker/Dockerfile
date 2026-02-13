# Prism MCP Server Dockerfile
# Multi-stage build with dev and prod targets

# =============================================================================
# Base stage: Common dependencies
# =============================================================================
FROM python:3.13-slim AS base

# Install system dependencies and Node.js for npx (Tavily MCP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install gosu for entrypoint user switching (both dev and prod)
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

# Install Gemini CLI for workers
RUN npm install -g @google/gemini-cli@0.28.1

# Create non-root user for podman rootless compatibility
RUN useradd -m -u 1000 appuser

# Install Claude CLI via official installer (as appuser, installs to ~/.claude/local/)
USER appuser
# renovate: datasource=npm depName=@anthropic-ai/claude-code
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.37
USER root

# Add Claude CLI to PATH (official installer puts it in ~/.local/bin and ~/.claude/local/bin)
ENV PATH="/home/appuser/.local/bin:/home/appuser/.claude/local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# =============================================================================
# Dev stage: Source mounted, editable install, hot-reload ready
# =============================================================================
FROM base AS dev

# Copy dependency spec + minimal source stub so editable install succeeds
COPY pyproject.toml README.md ./
RUN mkdir -p src/prism && touch src/prism/__init__.py

# Install all dependencies (cached in image layer)
RUN uv pip install --system --no-cache-dir -e ".[dev]"

# Create XDG directories (as root, will be owned by appuser via chown)
RUN mkdir -p /home/appuser/.config/prism \
    /home/appuser/.local/share/prism \
    /home/appuser/.cache/prism \
    && chown -R appuser:appuser /home/appuser/.config /home/appuser/.local /home/appuser/.cache

# gosu already installed in base stage

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf -o /dev/null --max-time 2 http://localhost:8765/ || test $? -eq 22

# Entrypoint runs as root for pip install, then drops to appuser
COPY docker/entrypoint-dev.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# =============================================================================
# Prod stage: Source copied, static install, minimal image
# =============================================================================
FROM base AS prod

# Copy source code and Alembic config
COPY pyproject.toml README.md alembic.ini ./
COPY src/ ./src/
COPY config/ ./config/
COPY hooks/ ./hooks/
COPY alembic/ ./alembic/

# Install package
RUN uv pip install --system --no-cache-dir . \
    && chmod +x /app/hooks/*.py

# Create XDG directories
RUN mkdir -p /home/appuser/.config/prism \
    /home/appuser/.local/share/prism \
    /home/appuser/.cache/prism \
    && chown -R appuser:appuser /home/appuser/.config /home/appuser/.local /home/appuser/.cache

# Entrypoint runs migrations then starts server
COPY docker/entrypoint-prod.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf -o /dev/null --max-time 2 http://localhost:8765/ || test $? -eq 22

ENTRYPOINT ["/entrypoint.sh"]
