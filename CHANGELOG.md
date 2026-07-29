# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-07-29

### Added

- Failure reasons are now parsed out of CLI stdout: both CLIs report *why* a run failed in their JSON payload while leaving stderr empty, so every API error previously collapsed to "Claude CLI exited with code 1". Rate limits, auth failures and overload now reach the session metadata and the logs
- A payload that flags `is_error` is treated as a failure even when the process exits 0
- `--fallback-model` (config `models.fallback`) so a saturated primary model degrades to a second model instead of failing the worker
- `manager_timeout_seconds` per level: planning, synthesis and resume ran unbounded, which let single L2 sessions reach 815s and blow past client timeouts
- `errors.max_message_chars` to cap surfaced failure reasons

### Changed

- Claude CLI 2.1.37 → 2.1.220. The `opus`/`haiku` aliases resolve against the installed CLI, so the old pin silently capped every "opus" worker at Opus 4.6; it now resolves to Opus 5
- Gemini CLI 0.28.1 → 0.52.0
- Gemini worker models pinned to rolling aliases `gemini-flash-latest` / `gemini-pro-latest` so a deprecated generation no longer strands a worker
- L1 worker timeout 140s → 200s, visible budget 50s → 80s. The old cap sat on top of the success-time distribution (p95 128s, max 140s), clipping healthy runs
- `fastmcp` 2.x → 3.4.5. The `<3.0.0` cap from 0.2.2 was verified against 3.4.5: tool calls, streamable HTTP and resume all work
- Base image python 3.13-slim → 3.14-slim
- Resume-chat manager model moved from a hardcoded `"sonnet"` to `models.session_manager.0`
- GitHub Actions bumped: checkout v7, setup-uv v8, buildx v4, login v4, build-push v7, gh-release v3

### Fixed

- Gemini workers aborted immediately under CLI ≥ 0.52, which refuses to run headless in an untrusted workspace; `--skip-trust` is now passed
- Gemini failures reported the Node `punycode` deprecation warning as the error because stderr was preferred over the payload
- Rate-limit errors were never retried: the reason never reached `error_message`, so it could not match the transient patterns

## [0.2.2] - 2026-05-29

### Fixed

- Cap `fastmcp` to `<3.0.0`: an unpinned resolve to fastmcp 3.x changed streamable-HTTP session handling and broke all tool calls with "Session not found"

## [0.2.1] - 2026-05-29

### Changed

- Gemini worker models bumped after upstream deprecation: L0/L1 `gemini-3-flash-preview` → `gemini-3.5-flash` (GA), L2/L3 `gemini-3-pro-preview` (shut down 2026-03-09) → `gemini-3.1-pro-preview`

## [0.2.0] - 2026-03-30

### Added

- `.dev.env.example` and `.prod.env.example` secret templates
- `config-custom/` directory for sparse YAML config overrides (gitignored)
- `Makefile` with dev/prod compose shortcuts and quality check targets
- Parent-child session tracking with cascading cancellation
- Parallel `cancel_all()` via `asyncio.gather` (was sequential, could exceed Docker stop timeout)
- `init: true` in both Docker Compose files for zombie process reaping
- Production entrypoint (`docker/entrypoint-prod.sh`) with DB URL construction from Docker secrets
- Alembic-only schema management: migrations run in entrypoints, `create_all` removed
- Gemini CLI as first-class search worker (gemini_search) via GeminiExecutor
- Multi-provider Level 0: `providers` parameter for explicit worker selection, `"mix"` for all 4 in parallel
- Worker factory (`workers/factory.py`) for unified worker creation across all levels
- JSON-lines structured logging (`core/logging.py`) for container log collection
- Time-aware hooks for both Claude and Gemini CLI agents (`core/hooks.py`)
- Hook log parsing and worker completion metrics
- YAML response serialization (`mcp_serializer.py`)
- Shared worker prompt templates (`prompts/system.md`, `prompts/user.md`)
- Per-worker prompt extensions under `prompts/workers/`
- Prompt-model test bench (`tests/prompt-model/`) for isolated prompt iteration
- Level-specific search manager prompts (`prompts/search_manager/levels/`, `prompts/search_manager/agents/`)
- E2E test suite with full container lifecycle management (5 tests: l0_default, l0_mix, l1, cancel, resume)
- Per-test result directories with structured YAML/JSON output

### Changed

- Compose secret injection migrated from `env_file:` to `--env-file` + `${VAR}` interpolation pattern
- Compose commands wrapped in Makefile targets (`make dev`, `make prod`, `make check`)
- `cancel_all()` now user-scoped via `X-User-Id` header (each user can only cancel their own sessions)
- In-memory session tracking includes `user_id` (inherited from parent session for child workers)
- Container volumes simplified: only persist PostgreSQL data and Claude CLI sessions (`~/.claude`)
- Removed unused `~/.local/share/prism` and `~/.cache/prism` container mounts
- MCP transport migrated from SSE to Streamable HTTP (`streamable-http`)
- Healthchecks standardized to TCP checks (transport-agnostic)
- E2E tests updated to use Streamable HTTP client endpoint
- Per-request user identification via `X-User-Id` HTTP header (replaces server-wide `PRISM_USER_ID` env var)
- `SearchFlow` accepts `user_id` per-call instead of at construction time
- L0 default from direct Perplexity API to claude_search (configurable via `level0.default_providers`)
- Workers unified: renamed to claude_search, tavily_search, perplexity_search, gemini_search
- All 4 worker types available at every level (L0-L3), not just L1-3
- Perplexity worker changed from direct API call to Claude + Perplexity MCP approach
- Config redesign: `models` section with per-level model config for session_manager, claude_workers, gemini_workers
- Config: added `level0`, `levels`, `models` sections; removed old per-worker config
- Timeouts no longer trigger retry for any worker (immediate return instead)
- Repository `update()` now requires `user_id` (enforced multi-tenancy)
- Prompt template rendering uses safe `.replace()` instead of `.format()`

### Fixed

- Hook log temp files (`/tmp/prism-hook-*.log`) now deleted after parsing
- Config loader resilient to `prism.yaml` being a directory (Podman mount edge case)
- Queries containing curly braces (`{`, `}`) no longer crash prompt rendering
- Post-hook now signals "time expired" to models when budget is exhausted

### Removed

- `cancel(session_id)` MCP tool (impractical: session_id unknown until search completes; use `cancel_all()` instead)
- Direct Perplexity API integration for L0 (replaced by unified worker approach)
- Old worker names: researcher, tavily, perplexity (replaced by *_search naming)

## [0.1.0] - 2026-02-08

### Added

- Multi-level search (Level 0-3) with parallel worker dispatch
- Level 0: Direct Perplexity API for instant answers
- Level 1-3: Orchestrated search with manager, dispatcher, synthesizer
- Worker agents: Claude Researcher, Tavily, Perplexity
- Two-tier retry: transient errors + schema validation with `--resume`
- PostgreSQL persistence with async SQLAlchemy (sessions, tasks)
- Session management: resume, cancel, list, history
- Multi-tenancy via user_id scoping
- Config system: YAML + env var overrides
- Dev environment: Podman Compose with hot-reload, /tmp data paths
- Production environment: XDG-compliant paths, secrets file
- Unit tests with async SQLite in-memory (112 tests)
- E2E tests via FastMCP Client over Streamable HTTP
