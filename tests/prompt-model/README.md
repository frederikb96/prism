# Prompt-Model Test Infrastructure

Containerized test environment for evaluating web search prompts across CLI providers and models. Part of Prism's provider research phase.

Full research context: `.claude/provider-research.md`

## Quick Start

```bash
# Build (once, or after Dockerfile changes)
podman compose build

# Start container (stays alive for repeated exec)
podman compose up -d

# Run a test
podman exec prism-test python /app/scripts/run_test.py \
  --provider claude --model haiku --query simple --timeout 50

# Stop
podman compose down
```

All commands from `tests/prompt-model/`. Tests can run in parallel — each gets an isolated hook log.

## Providers & Models

**Claude** — `haiku`, `sonnet`, `opus` (short aliases)
- Tools: `WebSearch` + `WebFetch`
- Hooks: inline JSON via `--settings` (not file path — file path doesn't load hooks)

**Gemini** — `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3-flash-preview`, `gemini-3-pro-preview`
- Tools: `google_web_search` only
- Hooks: via `GEMINI_CLI_SYSTEM_SETTINGS_PATH` env var
- System prompt: via `GEMINI_SYSTEM_MD` temp file (no CLI flag)
- No `--sandbox` in container (needs nested runtime)
- Runner maps `GOOGLE_API_KEY` → `GEMINI_API_KEY`

## Container

**Dockerfile** installs Claude CLI (curl installer), Gemini CLI (npm), Python 3.13 + PyYAML. Runs as `appuser` (uid 1000).

**Volumes** — all directory mounts for hot-reload without restart:

| Mount | Container path | Mode |
|-------|---------------|------|
| `scripts/` | `/app/scripts/` | ro |
| `prompts/` | `/app/prompts/` | ro |
| `queries/` | `/app/queries/` | ro |
| `hooks/` | `/app/hooks/` | ro |
| `results/` | `/app/results/` | rw |

Compose uses `userns_mode: keep-id` for rootless podman UID mapping.

**Env file**: `../../.config-local/.env` (symlink to `~/.config/prism/.env`). Required: `CLAUDE_CODE_OAUTH_TOKEN`, `GOOGLE_API_KEY`.

## Hook System

Hooks enforce time budgets: inject remaining seconds before each tool call, block tools when time expires.

**Env vars** set by runner, read by hook:
- `PRISM_START_TIME` — epoch float
- `PRISM_TOOL_TIMEOUT` — seconds
- `PRISM_HOOK_FORMAT` — `"claude"` or `"gemini"`
- `PRISM_HOOK_LOG` — per-process JSON-lines log path (isolates parallel runs)

**Hook log format** (JSON lines):
```json
{"hook": "pre", "format": "claude", "decision": "allow", "remaining_s": 42, "time": 1770717698.76}
{"hook": "pre", "format": "claude", "decision": "block", "remaining_s": 0, "time": 1770717748.12}
```

## Results

Saved as `results/{query}-{provider}-{model}.json`. New runs overwrite previous results for the same combo.

Structure:
- `metrics` — wall time, tool calls, hook blocks, tokens, cost
- `hook_activity` — summary: total events, pre/post counts, blocks
- `hook_events` — raw per-event JSON log
- `response_text` — model's answer
- `result.output` — full provider JSON output

## Gotchas

- **`--settings` with file path does NOT load hooks** — must pass inline JSON string
- **Claude `--output-format json` can return a list** — runner extracts the `{"type":"result"}` item
- **Bun segfault during Claude install** in container is non-fatal — binary works fine
- **Individual file mounts break hot-reload** in rootless podman — always mount directories
- **Claude CLI does not forward hook stderr** — hook detection uses marker file, not stderr

## Test Status (2026-02-10)

| Provider | Model | Wall Time | Tool Calls | Blocks | Response |
|----------|-------|-----------|------------|--------|----------|
| Claude | haiku | 16.4s | 2 | 0 | 1,665 chars |
| Claude | sonnet | 51.0s | 2 | 0 | 2,144 chars |
| Claude | opus | 45.3s | 2 | 0 | 2,531 chars |
| Gemini | 2.5-flash-lite | 11.7s | 1 | 0 | 939 chars |
| Gemini | 2.5-flash | 16.3s | 1 | 0 | 2,630 chars |
| Gemini | 3-flash-preview | 21.0s | 2 | 0 | 1,938 chars |
| Gemini | 3-pro-preview | 27.8s | 1 | 0 | 1,187 chars |

All on `simple` query (Bitcoin price), 50s timeout. Hook blocking confirmed working with 5s timeout tests.
