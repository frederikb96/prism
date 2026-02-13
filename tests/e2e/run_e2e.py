#!/usr/bin/env python3
"""
End-to-end test runner for Prism MCP server.

Runs tests against the dev Podman container with real API calls.
Produces per-test YAML result files in tests/e2e/results/ with detailed
timing, tool call data, manager phases, and full prompt content.

Features:
- Fail-fast: stops on first test failure
- Graceful Ctrl+C: cancels all server searches, writes partial results
- Per-test detailed YAML result files
- Nicer stdout summary

Usage:
    uv run python tests/e2e/run_e2e.py                        # all tests
    uv run python tests/e2e/run_e2e.py --only l0_default,l1   # specific tests
    uv run python tests/e2e/run_e2e.py --skip cancel           # skip specific
    uv run python tests/e2e/run_e2e.py --no-docker             # reuse running container
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastmcp import Client
from fixtures import (
    ALL_TESTS,
    CANCEL_QUERY,
    FETCH_URL,
    L0_DEFAULT_QUERY,
    L0_MIX_QUERY,
    L1_QUERY,
    RESUME_FOLLOW_UP,
)
from log_checker import TestLogs, parse_test_logs

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.dev.yaml"
RESULTS_DIR = Path(__file__).parent / "results"
CONTAINER_NAME = "prism-dev"

# Streamable HTTP endpoint (8766 host -> 8765 container)
MCP_URL = "http://localhost:8766/mcp"

# Per-test wall-clock timeouts (seconds)
TIMEOUTS: dict[str, int] = {
    "l0_default": 90,
    "l0_gemini": 90,
    "l0_mix": 150,
    "l1": 300,
    "cancel": 30,
    "resume": 120,
    "fetch": 60,
}

CANCEL_DELAY_S = 5

# ---------------------------------------------------------------------------
# Ctrl+C handling
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _on_sigint(signum: int, frame: Any) -> None:
    """First Ctrl+C: request graceful shutdown. Second: force exit."""
    global _shutdown_requested
    if _shutdown_requested:
        sys.exit(130)
    _shutdown_requested = True
    print("\n\nInterrupted! Finishing current operation, then cleaning up...")
    print("Press Ctrl+C again to force exit.")


# ---------------------------------------------------------------------------
# YAML result file dumper (literal block for multiline strings)
# ---------------------------------------------------------------------------


class _BlockDumper(yaml.SafeDumper):
    """YAML dumper with literal block style for multiline strings."""

    pass


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Use | style for multiline, plain for single-line."""
    if "\n" in data:
        # Strip trailing whitespace per line — PyYAML falls back to quoted
        # style if any line has trailing spaces (block scalars can't preserve them)
        cleaned = "\n".join(line.rstrip() for line in data.split("\n"))
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _str_representer)


def _dump_yaml(data: dict[str, Any]) -> str:
    """Dump dict as human-readable YAML."""
    return yaml.dump(
        data,
        Dumper=_BlockDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=1000,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of a single E2E test."""

    name: str
    passed: bool
    duration: float
    error: str | None = None
    response: dict[str, Any] = field(default_factory=dict)
    logs: TestLogs | None = None


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------


class ContainerManager:
    """Manages Podman Compose lifecycle for E2E tests."""

    def __init__(self, compose_file: Path) -> None:
        self.compose_file = compose_file

    def down(self) -> None:
        """Stop and remove containers."""
        subprocess.run(
            ["podman", "compose", "-f", str(self.compose_file), "down", "-v"],
            capture_output=True,
            check=False,
        )

    def clean_volumes(self) -> None:
        """Remove temp data directories (podman recreates on start)."""
        for path in ("/tmp/prism-claude", "/tmp/prism-postgres"):
            subprocess.run(["rm", "-rf", path], capture_output=True, check=False)

    def up(self) -> bool:
        """Start containers."""
        result = subprocess.run(
            ["podman", "compose", "-f", str(self.compose_file), "up", "-d"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Failed: {result.stderr.strip()}")
            return False
        return True

    def wait_healthy(self, timeout: int = 90) -> bool:
        """Wait for container to report healthy status."""
        start = time.time()
        while time.time() - start < timeout:
            result = subprocess.run(
                [
                    "podman",
                    "inspect",
                    "--format",
                    "{{.State.Health.Status}}",
                    CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
            )
            status = result.stdout.strip()
            if status == "healthy":
                return True
            if status == "unhealthy":
                print(f"  Container unhealthy after {time.time() - start:.0f}s")
                return False
            time.sleep(2)
        print(f"  Timeout after {timeout}s")
        return False


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


class TestRunner:
    """Executes E2E tests against the Prism MCP server."""

    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url
        self._l1_session_id: str | None = None

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse YAML response from MCP tool call."""
        try:
            return yaml.safe_load(text) or {}
        except yaml.YAMLError:
            return {"content": text, "success": True}

    async def run_test(self, name: str) -> TestResult:
        """
        Run a single named test with structured log collection.

        Records wall time, collects container logs from the test's
        time window, and returns a TestResult with all data.
        """
        method = getattr(self, f"test_{name}", None)
        if method is None:
            return TestResult(
                name=name,
                passed=False,
                duration=0,
                error=f"Unknown test: {name}",
            )

        log_start = datetime.now(timezone.utc)
        wall_start = time.time()

        try:
            result = await method()
        except asyncio.TimeoutError:
            timeout = TIMEOUTS.get(name, 0)
            result = TestResult(
                name=name,
                passed=False,
                duration=time.time() - wall_start,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            result = TestResult(
                name=name,
                passed=False,
                duration=time.time() - wall_start,
                error=str(e),
            )

        log_end = datetime.now(timezone.utc)

        # Collect structured logs from this test's time window
        result.logs = parse_test_logs(CONTAINER_NAME, log_start, log_end)

        return result

    # -- Individual test methods -------------------------------------------

    async def test_l0_default(self) -> TestResult:
        """L0 search with default provider (claude_search)."""
        return await self._run_search(
            name="l0_default",
            query=L0_DEFAULT_QUERY,
            level=0,
            timeout=TIMEOUTS["l0_default"],
        )

    async def test_l0_gemini(self) -> TestResult:
        """L0 search with gemini_search provider."""
        return await self._run_search(
            name="l0_gemini",
            query=L0_DEFAULT_QUERY,
            level=0,
            timeout=TIMEOUTS["l0_gemini"],
            providers=["gemini_search"],
        )

    async def test_l0_mix(self) -> TestResult:
        """L0 search with all 4 providers in parallel."""
        return await self._run_search(
            name="l0_mix",
            query=L0_MIX_QUERY,
            level=0,
            timeout=TIMEOUTS["l0_mix"],
            providers=["mix"],
        )

    async def test_l1(self) -> TestResult:
        """L1 search with manager + workers. Saves session_id for resume."""
        result = await self._run_search(
            name="l1",
            query=L1_QUERY,
            level=1,
            timeout=TIMEOUTS["l1"],
        )

        if result.passed:
            session_id = result.response.get("session_id")
            if session_id:
                self._l1_session_id = session_id
            else:
                result.passed = False
                result.error = "No session_id in response (needed for resume test)"

        return result

    async def test_cancel(self) -> TestResult:
        """Start L1 search, wait 5s, call cancel_all, check cancelled_count >= 1."""
        name = "cancel"
        timeout = TIMEOUTS[name]
        start = time.time()

        try:
            async with Client(self.mcp_url) as client:
                # Start L1 search in background
                search_task = asyncio.create_task(
                    client.call_tool(
                        "search",
                        {"query": CANCEL_QUERY, "level": 1},
                    )
                )

                # Wait before cancelling
                await asyncio.sleep(CANCEL_DELAY_S)

                # Cancel all active searches
                cancel_raw = await asyncio.wait_for(
                    client.call_tool("cancel_all", {}),
                    timeout=10,
                )
                cancel_text = cancel_raw.content[0].text if cancel_raw.content else ""
                cancel_parsed = self._parse_response(cancel_text)

                # Wait for the search task to finish (cancelled or done)
                try:
                    await asyncio.wait_for(search_task, timeout=timeout)
                except (asyncio.TimeoutError, Exception):
                    pass

            duration = time.time() - start
            cancelled_count = cancel_parsed.get("cancelled_count", 0)

            if cancelled_count < 1:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=f"cancelled_count={cancelled_count} (expected >= 1)",
                    response=cancel_parsed,
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                response=cancel_parsed,
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_resume(self) -> TestResult:
        """Resume L1 session with follow-up chat question."""
        name = "resume"
        timeout = TIMEOUTS[name]
        start = time.time()

        if not self._l1_session_id:
            return TestResult(
                name=name,
                passed=False,
                duration=0,
                error="No L1 session_id available (l1 test must run and pass first)",
            )

        # Find the DB session UUID for the L1 search
        session_uuid = await self._find_l1_session_uuid()
        if not session_uuid:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error="Could not find completed L1 session in DB via list_sessions",
            )

        try:
            async with Client(self.mcp_url) as client:
                raw = await asyncio.wait_for(
                    client.call_tool(
                        "resume",
                        {
                            "session_id": session_uuid,
                            "follow_up": RESUME_FOLLOW_UP,
                        },
                    ),
                    timeout=timeout,
                )

            duration = time.time() - start
            text = raw.content[0].text if raw.content else ""
            parsed = self._parse_response(text)

            if not parsed.get("success"):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                    response=parsed,
                )

            content = parsed.get("content", "")
            if len(content) < 50:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=f"Content too short: {len(content)} chars (expected >= 50)",
                    response=parsed,
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                response=parsed,
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_fetch(self) -> TestResult:
        """Fetch content from example.com via Tavily extract wrapper."""
        name = "fetch"
        timeout = TIMEOUTS[name]
        start = time.time()

        try:
            async with Client(self.mcp_url) as client:
                raw = await asyncio.wait_for(
                    client.call_tool("fetch", {"url": FETCH_URL}),
                    timeout=timeout,
                )

            duration = time.time() - start
            text = raw.content[0].text if raw.content else ""
            parsed = self._parse_response(text)

            if not parsed.get("success"):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                    response=parsed,
                )

            raw_content = parsed.get("raw_content", "")
            if len(raw_content) < 10:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=f"raw_content too short: {len(raw_content)} chars",
                    response=parsed,
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                response=parsed,
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    # -- Helpers -----------------------------------------------------------

    async def _run_search(
        self,
        *,
        name: str,
        query: str,
        level: int,
        timeout: int,
        providers: list[str] | None = None,
        min_content_length: int = 100,
    ) -> TestResult:
        """
        Shared search test helper.

        Calls the search MCP tool, validates success and content length.
        """
        start = time.time()

        try:
            params: dict[str, Any] = {"query": query, "level": level}
            if providers is not None:
                params["providers"] = providers

            async with Client(self.mcp_url) as client:
                raw = await asyncio.wait_for(
                    client.call_tool("search", params),
                    timeout=timeout,
                )

            duration = time.time() - start
            text = raw.content[0].text if raw.content else ""
            parsed = self._parse_response(text)

            if not parsed.get("success"):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                    response=parsed,
                )

            content = parsed.get("content", "")
            if len(content) < min_content_length:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=(
                        f"Content too short: {len(content)} chars"
                        f" (expected >= {min_content_length})"
                    ),
                    response=parsed,
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                response=parsed,
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {timeout}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def _find_l1_session_uuid(self) -> str | None:
        """
        Find the completed L1 session DB UUID via list_sessions.

        Matches by level, completed status, resumable flag, and query prefix.
        Falls back to the stored session_id if list_sessions fails.
        """
        try:
            async with Client(self.mcp_url) as client:
                raw = await asyncio.wait_for(
                    client.call_tool("list_sessions", {"limit": 10}),
                    timeout=10,
                )
            text = raw.content[0].text if raw.content else ""
            parsed = self._parse_response(text)

            for session in parsed.get("sessions", []):
                if (
                    session.get("resumable")
                    and session.get("level") == 1
                    and session.get("status") == "completed"
                    and L1_QUERY[:40] in session.get("query", "")
                ):
                    return session.get("id")
        except Exception:
            pass

        # Fallback: try stored session_id directly (might work if it's a DB UUID)
        return self._l1_session_id


# ---------------------------------------------------------------------------
# Result file writing
# ---------------------------------------------------------------------------


def write_result_file(result: TestResult) -> Path:
    """
    Write a detailed YAML result file for one test.

    Includes response content, manager phases, worker details with
    per-tool-call timing, and full prompt content.

    Returns:
        Path to the written file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "test": result.name,
        "status": "PASS" if result.passed else "FAIL",
        "wall_time_s": round(result.duration, 2),
    }

    if result.error:
        data["error"] = result.error

    # -- Response section --
    if result.response:
        resp: dict[str, Any] = {
            "success": result.response.get("success", False),
        }

        content = result.response.get("content", "")
        if content:
            resp["content_length"] = len(content)
            resp["content"] = content

        if result.response.get("session_id"):
            resp["session_id"] = result.response["session_id"]

        if result.response.get("cancelled_count") is not None:
            resp["cancelled_count"] = result.response["cancelled_count"]

        if result.response.get("message"):
            resp["message"] = result.response["message"]

        metadata = result.response.get("metadata")
        if metadata:
            resp["metadata"] = metadata

        data["response"] = resp

    # -- Log-based sections --
    if result.logs:
        # Manager phases
        if result.logs.manager_phases:
            manager: dict[str, Any] = {}
            for phase in result.logs.manager_phases:
                entry: dict[str, Any] = {"wall_time_s": phase.wall_time_s}
                if phase.session_id:
                    entry["session_id"] = phase.session_id
                manager[phase.phase] = entry
            data["manager"] = manager

        # Workers
        if result.logs.workers:
            workers: list[dict[str, Any]] = []
            for w in result.logs.workers:
                w_entry: dict[str, Any] = {
                    "agent_key": w.agent_key,
                    "type": w.worker_type,
                    "status": "PASS" if w.success else "FAIL",
                    "wall_time_s": w.wall_time_s,
                    "model": w.model,
                    "content_length": w.response_length,
                    "tool_calls": w.tool_calls,
                    "hook_blocks": w.hook_blocks,
                }

                if w.tool_call_details:
                    calls: list[dict[str, Any]] = []
                    for tc in w.tool_call_details:
                        call: dict[str, Any] = {"tool": tc.get("tool", "unknown")}
                        if tc.get("blocked"):
                            call["BLOCKED"] = True
                            call["at_s"] = tc.get("start_s")
                        else:
                            call["start_s"] = tc.get("start_s")
                            call["end_s"] = tc.get("end_s")
                            call["duration_s"] = tc.get("duration_s")
                        if tc.get("remaining_s") is not None:
                            call["remaining_s"] = tc["remaining_s"]
                        calls.append(call)
                    w_entry["calls"] = calls

                workers.append(w_entry)
            data["workers"] = workers

        # Prompts
        if result.logs.prompts:
            prompts: dict[str, str] = {}
            for p in result.logs.prompts:
                prompts[p.prompt_type] = p.prompt
            data["prompts"] = prompts

        # Errors / warnings from logs
        if result.logs.errors:
            data["log_errors"] = result.logs.errors
        if result.logs.warnings:
            data["log_warnings"] = result.logs.warnings

    file_path = RESULTS_DIR / f"{result.name}.yaml"
    file_path.write_text(_dump_yaml(data))
    return file_path


# ---------------------------------------------------------------------------
# Stdout formatting
# ---------------------------------------------------------------------------


def _print_header(tests: list[str]) -> None:
    """Print test suite header."""
    print()
    print("E2E Test Suite")
    print("=" * 50)
    print(f"  Tests: {', '.join(tests)}")
    print()


def _print_test_line(result: TestResult) -> None:
    """Print single test result line."""
    marker = "  +" if result.passed else "  x"
    status = "PASS" if result.passed else "FAIL"
    name_padded = f"{result.name} ".ljust(24, ".")
    print(f"{marker} {name_padded} {status}  ({result.duration:.1f}s)")
    if result.error:
        # Indent error under the test line
        print(f"    Error: {result.error[:120]}")


def _print_summary(
    results: list[TestResult],
    interrupted: bool = False,
) -> None:
    """Print final summary with pass/fail counts."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration for r in results)

    print()
    print("=" * 50)

    if interrupted:
        print(f"  {passed}/{len(results)} passed  |  Interrupted")
    elif failed > 0:
        print(f"  {passed}/{len(results)} passed, {failed} failed  |  Stopped on first failure")
    else:
        print(f"  {passed}/{len(results)} passed  |  Total: {total_time:.1f}s")

    rel_results = RESULTS_DIR.relative_to(PROJECT_ROOT)
    print(f"  Results: {rel_results}/")
    print()


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Prism E2E tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tests: {', '.join(ALL_TESTS)}",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip",
        type=str,
        help="Comma-separated tests to skip",
    )
    group.add_argument(
        "--only",
        type=str,
        help="Comma-separated tests to run (only these)",
    )

    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Skip Podman setup (assumes container already running)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


async def _cancel_all_searches() -> None:
    """Send cancel_all to the server to stop in-flight searches."""
    try:
        async with Client(MCP_URL) as client:
            await asyncio.wait_for(
                client.call_tool("cancel_all", {}),
                timeout=5,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Main entry point."""
    global _shutdown_requested
    signal.signal(signal.SIGINT, _on_sigint)

    args = _parse_args()

    # Resolve test list
    if args.only:
        tests = [t.strip() for t in args.only.split(",")]
        invalid = [t for t in tests if t not in ALL_TESTS]
        if invalid:
            print(f"Unknown tests: {invalid}")
            print(f"Available: {ALL_TESTS}")
            return 1
    elif args.skip:
        skip = {t.strip() for t in args.skip.split(",")}
        invalid = skip - set(ALL_TESTS)
        if invalid:
            print(f"Unknown tests to skip: {invalid}")
            return 1
        tests = [t for t in ALL_TESTS if t not in skip]
    else:
        tests = list(ALL_TESTS)

    if not tests:
        print("No tests to run")
        return 0

    # Container setup
    if not args.no_docker:
        container = ContainerManager(COMPOSE_FILE)
        print("\nSetting up Podman environment...")
        print("  Stopping existing containers...")
        container.down()
        print("  Cleaning temp volumes...")
        container.clean_volumes()
        print("  Starting containers...")
        if not container.up():
            return 1
        print("  Waiting for health check...")
        if not container.wait_healthy():
            return 1
        print("  Container ready")

    # Clean old result files
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in RESULTS_DIR.glob("*.yaml"):
        f.unlink()

    _print_header(tests)

    # Run tests
    runner = TestRunner(MCP_URL)
    results: list[TestResult] = []

    for test_name in tests:
        if _shutdown_requested:
            break

        result = await runner.run_test(test_name)
        results.append(result)
        _print_test_line(result)
        write_result_file(result)

        # Fail-fast: stop on first failure
        if not result.passed and not _shutdown_requested:
            break

    # Cleanup on interrupt
    if _shutdown_requested:
        print("\nCancelling active searches...")
        await _cancel_all_searches()
        print("Cleanup complete.")

    _print_summary(results, interrupted=_shutdown_requested)

    # Leave container alive for inspection
    if not args.no_docker:
        print("Container left running for inspection.\n")

    if _shutdown_requested:
        return 130
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
