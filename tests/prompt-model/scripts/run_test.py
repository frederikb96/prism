#!/usr/bin/env python3
"""Prompt-model test runner for Prism.

Runs a search query through a CLI provider + model with time-budgeted hooks.
Designed to run inside the prism-test container.

Usage (via podman exec):
    python /app/scripts/run_test.py --provider claude --model haiku --query simple
    python /app/scripts/run_test.py --provider gemini --model gemini-2.5-flash \
        --query simple --timeout 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
WORKERS_DIR = PROMPTS_DIR / "workers"
HOOKS_DIR = BASE_DIR / "hooks"
QUERIES_DIR = BASE_DIR / "queries"
RESULTS_DIR = BASE_DIR / "results"

HOOK_SCRIPT = str(HOOKS_DIR / "time_hook.py")

CLAUDE_HOOKS_CONFIG = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"python3 {HOOK_SCRIPT} pre",
                    }
                ],
            }
        ],
        "PostToolUse": [
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"python3 {HOOK_SCRIPT} post",
                    }
                ],
            }
        ],
    }
}


def make_hook_log() -> str:
    """Create a unique hook log path for this test run."""
    return f"/tmp/hook-{uuid.uuid4().hex[:8]}.log"


def read_hook_log(log_path: str) -> list[dict]:
    """Read structured hook events from the JSON-lines log file."""
    try:
        lines = Path(log_path).read_text().strip().splitlines()
    except (OSError, FileNotFoundError):
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def cleanup_hook_log(log_path: str) -> None:
    try:
        Path(log_path).unlink(missing_ok=True)
    except OSError:
        pass


def parse_hook_activity(events: list[dict]) -> dict:
    """Derive tool activity summary from hook events."""
    pre_events = [e for e in events if e.get("hook") == "pre"]
    post_events = [e for e in events if e.get("hook") == "post"]
    blocks = [e for e in events if e.get("decision") == "block"]
    return {
        "total_events": len(events),
        "pre_hooks": len(pre_events),
        "post_hooks": len(post_events),
        "tool_calls_observed": len(pre_events),
        "blocks": len(blocks),
    }


def load_query(name: str) -> dict:
    path = QUERIES_DIR / f"{name}.yaml"
    if not path.exists():
        print(f"ERROR: Query file not found: {path}")
        sys.exit(1)
    return yaml.safe_load(path.read_text())


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def build_system_prompt(worker_type: str) -> str:
    """Load system prompt and inject per-worker tool section."""
    template = load_prompt("system.md")
    worker_path = WORKERS_DIR / f"{worker_type}.md"
    if not worker_path.exists():
        print(f"ERROR: Worker section not found: {worker_path}")
        sys.exit(1)
    worker_section = worker_path.read_text().strip()
    return template.replace("{worker_section}", worker_section)


def build_user_prompt(query: dict, timeout_seconds: int) -> str:
    template = load_prompt("user.md")
    return template.format(
        query=query["query"],
        timeout_seconds=timeout_seconds,
    )


WORKER_TOOLS: dict[str, dict[str, Any]] = {
    "websearch": {
        "allowed_tools": ["WebSearch", "WebFetch"],
    },
    "tavily": {
        "tools_flag": "mcp",
        "allowed_tools": ["mcp__tavily__tavily_search", "mcp__tavily__tavily_extract"],
        "mcp_config": {
            "mcpServers": {
                "tavily": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "tavily-mcp"],
                    "env": {"TAVILY_API_KEY": "${TAVILY_API_KEY}"},
                },
            },
        },
    },
    "perplexity": {
        "tools_flag": "mcp",
        "allowed_tools": ["mcp__perplexity__search", "mcp__perplexity__reason"],
        "mcp_config": {
            "mcpServers": {
                "perplexity": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "perplexity-mcp"],
                    "env": {"PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"},
                },
            },
        },
    },
}


def run_claude(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    hard_timeout: int = 0,
    worker_type: str = "websearch",
    effort: str = "",
) -> dict:
    """Run a Claude Code search and return results."""
    hook_log = make_hook_log()
    env = os.environ.copy()
    env["PRISM_START_TIME"] = str(time.time())
    env["PRISM_TOOL_TIMEOUT"] = str(timeout_seconds)
    env["PRISM_HOOK_FORMAT"] = "claude"
    env["PRISM_HOOK_LOG"] = hook_log
    if effort:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = effort

    worker_config = WORKER_TOOLS[worker_type]
    settings_json = json.dumps(CLAUDE_HOOKS_CONFIG)

    cmd = [
        "claude",
        "-p", user_prompt,
        "--system-prompt", system_prompt,
        "--output-format", "json",
        "--model", model,
        "--no-session-persistence",
        "--settings", settings_json,
    ]

    if "tools_flag" in worker_config:
        cmd.extend(["--tools", worker_config["tools_flag"]])

    if "mcp_config" in worker_config:
        cmd.extend(["--mcp-config", json.dumps(worker_config["mcp_config"])])
        cmd.append("--strict-mcp-config")

    for tool in worker_config["allowed_tools"]:
        cmd.extend(["--allowedTools", tool])

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=hard_timeout or (timeout_seconds + 60),
    )
    elapsed = time.monotonic() - start

    try:
        output = json.loads(result.stdout)
        if isinstance(output, list):
            result_items = [
                item for item in output
                if isinstance(item, dict) and item.get("type") == "result"
            ]
            output = result_items[0] if result_items else {"raw_list": output, "parse_error": True}
    except json.JSONDecodeError:
        output = {"raw_stdout": result.stdout[:2000], "parse_error": True}

    hook_events = read_hook_log(hook_log)
    cleanup_hook_log(hook_log)

    return {
        "provider": "claude",
        "model": model,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": result.returncode,
        "output": output,
        "hook_events": hook_events,
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def run_gemini(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    hard_timeout: int = 0,
) -> dict:
    """Run a Gemini CLI search and return results."""
    hook_log = make_hook_log()
    env = os.environ.copy()
    env["PRISM_START_TIME"] = str(time.time())
    env["PRISM_TOOL_TIMEOUT"] = str(timeout_seconds)
    env["PRISM_HOOK_FORMAT"] = "gemini"
    env["PRISM_HOOK_LOG"] = hook_log
    env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(HOOKS_DIR / "gemini_settings.json")

    if "GEMINI_API_KEY" not in env and "GOOGLE_API_KEY" in env:
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]

    sys_md = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="prism-sys-", delete=False
    )
    sys_md.write(system_prompt)
    sys_md.close()
    env["GEMINI_SYSTEM_MD"] = sys_md.name

    cmd = [
        "gemini",
        "-p", user_prompt,
        "--model", model,
        "--allowed-tools", "google_web_search",
        "-o", "json",
    ]

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=hard_timeout or (timeout_seconds + 60),
    )
    elapsed = time.monotonic() - start

    os.unlink(sys_md.name)

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        output = {"raw_stdout": result.stdout[:2000], "parse_error": True}

    hook_events = read_hook_log(hook_log)
    cleanup_hook_log(hook_log)

    return {
        "provider": "gemini",
        "model": model,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": result.returncode,
        "output": output,
        "hook_events": hook_events,
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def extract_metrics(result: dict) -> dict:
    """Extract comparable metrics from a test result."""
    output = result.get("output", {})
    hook_activity = parse_hook_activity(result.get("hook_events", []))

    metrics = {
        "wall_time_s": result["elapsed_seconds"],
        "exit_code": result["exit_code"],
        "tool_calls": hook_activity["tool_calls_observed"],
        "hook_blocks": hook_activity["blocks"],
        "hook_events_total": hook_activity["total_events"],
    }

    if result["provider"] == "claude":
        metrics["response_length"] = len(output.get("result", ""))
        metrics["cost_usd"] = output.get("total_cost_usd", output.get("cost_usd", 0))
        metrics["duration_api_ms"] = output.get("duration_api_ms", output.get("duration_ms", 0))
        metrics["num_turns"] = output.get("num_turns", 0)
        usage = output.get("usage", {})
        metrics["input_tokens"] = usage.get("input_tokens", 0)
        metrics["output_tokens"] = usage.get("output_tokens", 0)
        metrics["cache_read_tokens"] = usage.get("cache_read_input_tokens", 0)

    elif result["provider"] == "gemini":
        metrics["response_length"] = len(output.get("response", ""))
        stats = output.get("stats", {})
        tools = stats.get("tools", {})
        by_name = tools.get("byName", {})
        metrics["gemini_tool_breakdown"] = {
            name: {
                "count": info.get("count", 0),
                "success": info.get("success", 0),
                "fail": info.get("fail", 0),
            }
            for name, info in by_name.items()
        }
        metrics["gemini_tool_duration_ms"] = tools.get("totalDurationMs", 0)
        for _model_name, model_stats in stats.get("models", {}).items():
            api = model_stats.get("api", {})
            metrics["api_latency_ms"] = api.get("totalLatencyMs", 0)
            tokens = model_stats.get("tokens", {})
            metrics["input_tokens"] = tokens.get("input", tokens.get("inputTokens", 0))
            metrics["output_tokens"] = tokens.get("candidates", tokens.get("outputTokens", 0))
            metrics["thinking_tokens"] = tokens.get("thoughts", tokens.get("thinkingTokens", 0))

    return metrics


def get_response_text(result: dict) -> str:
    """Extract the model's response text."""
    output = result.get("output", {})
    if result["provider"] == "claude":
        return output.get("result", output.get("raw_stdout", ""))
    elif result["provider"] == "gemini":
        return output.get("response", output.get("raw_stdout", ""))
    return ""


def print_result(result: dict, query: dict) -> None:
    """Print a formatted test result."""
    metrics = extract_metrics(result)
    response = get_response_text(result)

    print(f"\n{'=' * 70}")
    print(f"  {result['provider'].upper()} — {result['model']}")
    print(f"  Query: {query['name']} — {query['description']}")
    print(f"{'=' * 70}")

    print("\n  Metrics:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"     {k}:")
            for sk, sv in v.items():
                print(f"       {sk}: {sv}")
        else:
            print(f"     {k}: {v}")

    hook_activity = parse_hook_activity(result.get("hook_events", []))
    print("\n  Hook Activity:")
    print(f"     tool calls: {hook_activity['tool_calls_observed']}")
    print(f"     blocks: {hook_activity['blocks']}")
    print(f"     total events: {hook_activity['total_events']}")

    print(f"\n  Response ({len(response)} chars):")
    print(f"{'─' * 70}")
    if len(response) > 3000:
        print(response[:3000])
        print(f"\n  ... (truncated, {len(response)} total chars)")
    else:
        print(response)
    print(f"{'─' * 70}")


def save_result(result: dict, query: dict, timeout: int, label: str = "") -> None:
    """Save full result to results directory. Overwrites previous runs."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = f"-{label}" if label else ""
    filename = f"{query['name']}-{result['provider']}-{result['model']}{suffix}.json"
    filepath = RESULTS_DIR / filename

    metrics = extract_metrics(result)
    hook_activity = parse_hook_activity(result.get("hook_events", []))

    full_result = {
        "timestamp": timestamp,
        "label": label or None,
        "query": query,
        "timeout_seconds": timeout,
        "metrics": metrics,
        "hook_activity": hook_activity,
        "hook_events": result.get("hook_events", []),
        "response_text": get_response_text(result),
        "result": {
            "provider": result["provider"],
            "model": result["model"],
            "elapsed_seconds": result["elapsed_seconds"],
            "exit_code": result["exit_code"],
            "output": result["output"],
        },
    }
    filepath.write_text(json.dumps(full_result, indent=2, default=str))
    print(f"\n  Saved: {filepath.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prism prompt-model test runner")
    parser.add_argument(
        "--provider", required=True, choices=["claude", "gemini"],
        help="CLI provider to test",
    )
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument(
        "--query", required=True,
        help="Query name (simple, medium, hard)",
    )
    parser.add_argument(
        "--timeout", type=int, default=50,
        help="Tool call timeout in seconds (default: 50)",
    )
    parser.add_argument(
        "--hard-timeout", type=int, default=0,
        help="Hard process kill timeout in seconds (default: tool timeout + 60)",
    )
    parser.add_argument(
        "--worker-type", default="websearch",
        choices=["websearch", "tavily", "perplexity"],
        help="Claude worker tool set (default: websearch)",
    )
    parser.add_argument(
        "--effort", default=None,
        choices=["low", "medium", "high"],
        help="Claude Code effort level (sets CLAUDE_CODE_EFFORT_LEVEL)",
    )
    parser.add_argument(
        "--label", default="",
        help="Label suffix for result filename (e.g. 'effort-low', 'think-0')",
    )
    args = parser.parse_args()
    if not args.hard_timeout:
        args.hard_timeout = args.timeout + 60

    # Determine effective worker type (gemini auto-uses its own worker section)
    worker_type = "gemini" if args.provider == "gemini" else args.worker_type

    query = load_query(args.query)
    system_prompt = build_system_prompt(worker_type)
    user_prompt = build_user_prompt(query, args.timeout)

    # Auto-build label from configuration if not explicitly set
    if not args.label:
        label_parts = []
        if args.effort:
            label_parts.append(f"effort-{args.effort}")
        if worker_type not in ("websearch", "gemini"):
            label_parts.append(worker_type)
        args.label = "-".join(label_parts) if label_parts else ""
    label_info = f" [{args.label}]" if args.label else ""
    worker_info = f" worker={worker_type}" if worker_type != "websearch" else ""
    print(
        f"Starting test: {args.provider}/{args.model}{label_info}{worker_info}"
        f" -- query={query['name']} -- timeout={args.timeout}s"
    )

    try:
        if args.provider == "claude":
            result = run_claude(
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=args.timeout,
                hard_timeout=args.hard_timeout,
                worker_type=worker_type,
                effort=args.effort or "",
            )
        elif args.provider == "gemini":
            result = run_gemini(
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=args.timeout,
                hard_timeout=args.hard_timeout,
            )
        else:
            print(f"Unknown provider: {args.provider}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"\nHARD TIMEOUT: Process killed after {args.hard_timeout}s")
        sys.exit(1)

    print_result(result, query)
    save_result(result, query, args.timeout, label=args.label)


if __name__ == "__main__":
    main()
