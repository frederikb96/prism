# Prism

[![CI](https://github.com/frederikb96/prism/actions/workflows/ci.yaml/badge.svg)](https://github.com/frederikb96/prism/actions/workflows/ci.yaml)
[![Release](https://img.shields.io/github/v/release/frederikb96/prism)](https://github.com/frederikb96/prism/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-level web search MCP server. Wraps Perplexity, Claude, and Tavily behind a unified interface.

## Search Levels

- **Level 0**: Instant - direct Perplexity API call
- **Level 1**: Quick - 1-2 sources, parallel workers
- **Level 2**: Standard - 3-5 sources, comprehensive
- **Level 3**: Deep - 6+ sources, exhaustive research

Levels 1-3 use a search manager that plans tasks, dispatches parallel workers, and synthesizes results.

## Quick Start

### Prerequisites

- Podman with `podman compose`
- API keys in `~/.config/prism/.env`:
  ```bash
  mkdir -p ~/.config/prism
  cat > ~/.config/prism/.env << 'EOF'
  PERPLEXITY_API_KEY=pplx-...
  TAVILY_API_KEY=tvly-...
  CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
  EOF
  ```

### Development

```bash
# Start dev environment (PostgreSQL + Prism with hot-reload)
podman compose -f docker-compose.dev.yaml up -d

# Data lives in /tmp for easy cleanup:
# /tmp/prism-postgres, /tmp/prism-data, /tmp/prism-cache

# View logs
podman logs prism-dev

# Stop
podman compose -f docker-compose.dev.yaml down
```

### Production

```bash
# Create postgres password
openssl rand -base64 32 > ~/.config/prism/postgres_password
chmod 600 ~/.config/prism/postgres_password

# Start (uses XDG paths: ~/.local/share/prism, ~/.cache/prism)
podman compose up -d
```

## Configuration

Config loads with priority: environment variables > `~/.config/prism/prism.yaml` > `config/config.yaml`

**Environment variable naming:** YAML paths joined with underscores, prefixed `PRISM_`:
- `server.port` → `PRISM_SERVER_PORT`
- `retry.max_transient_retries` → `PRISM_RETRY_MAX_TRANSIENT_RETRIES`

See `config/config.yaml` for all settings.

## MCP Registration

Add to Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "prism": {
      "url": "http://localhost:8765/sse",
      "transport": "sse"
    }
  }
}
```

## API

**Tools:**
- `search(query, level=0)` - Execute search at specified depth
- `cancel(session_id)` - Cancel running search
- `cancel_all()` - Cancel all running searches
- `get_session(session_id)` - Retrieve session details
- `list_sessions(limit=20)` - List recent sessions
- `resume(session_id, follow_up)` - Resume L1-L3 session with follow-up

## Development

### UV for Python

All Python commands through UV (never `python` or `pip` directly):

```bash
uv run pytest tests/unit/ -v
uv run python -m prism
uv run ruff check src/
```

### Testing

```bash
# Unit tests (fast, mocked, no API keys)
uv run pytest tests/unit/ -v

# E2E tests (requires running dev container + API keys)
uv run python tests/e2e/run_e2e.py
```

### Database Migrations

```bash
uv run alembic revision --autogenerate -m "Description"
uv run alembic upgrade head
```

## Architecture

```
                    MCP Client
                        |
                   [FastMCP Server]
                        |
        +---------------+---------------+
        |               |               |
    Level 0         Level 1-3       Session
   (direct)      (orchestrated)    Management
        |               |
   [Perplexity]  [Search Manager]
                        |
                 [Task Dispatcher]
                        |
            +-----------+-----------+
            |           |           |
       [Researcher] [Tavily]  [Perplexity]
            |           |           |
            +-----------+-----------+
                        |
                  [Synthesizer]
                        |
                    Response
```

**Key patterns:**
- Single Claude CLI invocation point (`core/executor.py`)
- Two-tier retry: transient errors + schema validation with `--resume`
- DI throughout, no global state
- Multi-tenancy via user_id scoping
