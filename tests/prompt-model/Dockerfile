# Prism prompt-model test container
# Claude CLI + Gemini CLI + Python for rapid search testing

FROM python:3.13-slim AS base

# System deps + Node.js (for Gemini CLI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Gemini CLI (npm, pinned)
RUN npm install -g @google/gemini-cli@0.27.3

# Python deps for runner
RUN pip install --no-cache-dir pyyaml

# Create non-root user
RUN useradd -m -u 1000 appuser

# Claude CLI (native installer, as appuser)
USER appuser
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.38
ENV PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app
