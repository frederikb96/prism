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

# Install Claude CLI for Level 1-3 workers
RUN npm install -g @anthropic-ai/claude-code@2.1.37

# Create non-root user for podman rootless compatibility
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# =============================================================================
# Dev stage: Source mounted, editable install, hot-reload ready
# =============================================================================
FROM base AS dev

# Copy only dependency files first for better caching
COPY pyproject.toml ./

# Install dependencies only (not the package itself)
RUN uv pip install --system --no-cache-dir -e ".[dev]" || true

# Switch to non-root user
USER appuser

# Create XDG directories
RUN mkdir -p /home/appuser/.config/prism \
    /home/appuser/.local/share/prism \
    /home/appuser/.cache/prism

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD bash -c 'echo > /dev/tcp/localhost/8765' || exit 1

# Entrypoint installs package in editable mode and runs
COPY docker/entrypoint-dev.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# =============================================================================
# Prod stage: Source copied, static install, minimal image
# =============================================================================
FROM base AS prod

# Copy source code
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY hooks/ ./hooks/

# Install package
RUN uv pip install --system --no-cache-dir . \
    && chmod +x /app/hooks/*.py

# Switch to non-root user
USER appuser

# Create XDG directories
RUN mkdir -p /home/appuser/.config/prism \
    /home/appuser/.local/share/prism \
    /home/appuser/.cache/prism

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD bash -c 'echo > /dev/tcp/localhost/8765' || exit 1

# Default command
CMD ["python", "-m", "prism"]
