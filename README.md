# Prism

[![CI](https://github.com/frederikb96/prism/actions/workflows/ci.yaml/badge.svg)](https://github.com/frederikb96/prism/actions/workflows/ci.yaml)
[![Release](https://img.shields.io/github/v/release/frederikb96/prism)](https://github.com/frederikb96/prism/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-level web search MCP server. Wraps Claude, Gemini, Perplexity, and Tavily behind a unified interface.

## Search Levels

- **Level 0**: Instant - direct worker call (default: claude_search), supports multi-provider selection
- **Level 1**: Quick - 2-3 workers, parallel dispatch (~60s)
- **Level 2**: Standard - 4-6 workers, comprehensive (~150s)
- **Level 3**: Deep - 8-12 workers, exhaustive research (~600s)

All 4 worker types (claude_search, tavily_search, perplexity_search, gemini_search) are available at every level. L0 supports explicit provider selection via the `providers` parameter. Levels 1-3 use a search manager that plans tasks, dispatches parallel workers, and synthesizes results.

## Quick Start

### Prerequisites

- Podman with `podman compose`
- API keys: copy `.dev.env.example` to `.dev.env` and fill in values. Keys already in your OS environment (e.g., via `~/.bashrc`) flow through automatically -- only add entries for keys not in your shell.

### Development

```bash
cp .dev.env.example .dev.env    # fill in missing API keys
make dev                         # start PostgreSQL + Prism with hot-reload
make dev-logs                    # view logs
make dev-down                    # stop
```

Data lives in `/tmp` for easy cleanup: `/tmp/prism-postgres`, `/tmp/prism-claude`.

### Production

```bash
cp .prod.env.example .prod.env  # fill in secrets

# Create postgres password
openssl rand -base64 32 > ~/.config/prism/postgres_password
chmod 600 ~/.config/prism/postgres_password

make prod                        # start
```

Data persists at `~/.local/share/prism/postgres` and `~/.local/share/prism/claude`.

## Configuration

All settings with defaults and comments live in `config/config.yaml` -- this file IS the config documentation.

**Loading chain** (highest priority wins):
- `PRISM_SECTION_KEY` environment variables
- Custom override YAML (`config-custom/`, sparse -- only values that differ)
- `config/config.yaml` (defaults, always present in image)

**Env var naming:** YAML paths joined with underscores, prefixed `PRISM_`:
- `server.port` --> `PRISM_SERVER_PORT`
- `retry.max_transient_retries` --> `PRISM_RETRY_MAX_TRANSIENT_RETRIES`

**Secrets** use environment variables injected via `--env-file` into compose. See `.prod.env.example` for required variables. Custom config overrides go in `config-custom/` (gitignored).

## MCP Registration

Add to your MCP client config (e.g., `~/.claude.json` for Claude Code):

```json
{
  "mcpServers": {
    "prism": {
      "type": "http",
      "url": "http://localhost:8765/mcp",
      "headers": {
        "X-User-Id": "your-username"
      }
    }
  }
}
```

The `X-User-Id` header identifies the user for session scoping. Each user sees only their own sessions. Omitting the header defaults to `"default"`.

## API

**Tools:**
- `search(query, level=0, providers=None)` - Execute search at specified depth
  - `providers` (L0 only): `["claude_search"]`, `["tavily_search"]`, `["gemini_search"]`, `["perplexity_search"]`, any combination, or `["mix"]` for all 4 in parallel. Default (None): claude_search only.
- `cancel_all()` - Cancel all running searches for the current user
- `get_session(session_id)` - Retrieve session details
- `list_sessions(limit=20, offset=0, search=None)` - List recent sessions
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

# E2E tests (manages full container lifecycle + requires API keys)
uv run python tests/e2e/run_e2e.py                     # all tests
uv run python tests/e2e/run_e2e.py --only l0_default,l1 # specific tests
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
   (multi-provider)  (orchestrated)   Management
           |               |
    [Worker Factory]  [Search Manager]
           |               |
    +------+------+   [Worker Dispatcher]
    |      |      |        |
  [1-4 workers]    +-------+-------+-------+
    in parallel    |       |       |       |
           [Claude] [Gemini] [Tavily] [Perplexity]
                   |       |       |       |
                   +-------+-------+-------+
                           |
                     [Synthesizer]
                           |
                       Response
```

**Key patterns:**
- Dual CLI executors: `core/executor.py` (Claude), `core/gemini.py` (Gemini)
- Unified worker factory: same 4 worker types available at all levels
- Two-tier retry: transient errors + schema validation with `--resume` (Claude only)
- Time-aware hooks for both Claude and Gemini CLI agents
- JSON-lines structured logging
- DI throughout, no global state
- Multi-tenancy via user_id scoping
