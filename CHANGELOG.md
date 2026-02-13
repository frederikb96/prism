# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Changed

- L0 default from direct Perplexity API to claude_search (configurable via `level0.default_providers`)
- Workers unified: renamed to claude_search, tavily_search, perplexity_search, gemini_search
- All 4 worker types available at every level (L0-L3), not just L1-3
- Perplexity worker changed from direct API call to Claude + Perplexity MCP approach
- Config redesign: `models` section with per-level model config for session_manager, claude_workers, gemini_workers
- Config: added `level0`, `levels`, `models` sections; removed old per-worker config

### Removed

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
- E2E tests via FastMCP Client over SSE
