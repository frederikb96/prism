# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
