#!/usr/bin/env python3
"""
End-to-end test runner for Prism MCP server.

Runs tests against the Podman container to verify full functionality.
Uses FastMCP Client to connect to the running server via SSE.

Usage:
    python tests/e2e/run_e2e.py              # Run all tests
    python tests/e2e/run_e2e.py --only level_0,level_1  # Run specific tests
    python tests/e2e/run_e2e.py --skip cancel           # Skip specific tests
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastmcp import Client
from fixtures import (
    HOOK_BLOCK_QUERY,
    LEVEL_0_GEMINI_QUERY,
    LEVEL_0_MIX_QUERY,
    LEVEL_0_QUERY,
    LEVEL_1_QUERY,
    RESUME_QUERY,
)
from log_checker import check_container_logs, find_hook_blocks_in_logs

# Project root for docker-compose
PROJECT_ROOT = Path(__file__).parent.parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.dev.yaml"
CONTAINER_NAME = "prism-dev"

# SSE endpoint (port 8766 is exposed externally, maps to 8765 inside)
SSE_URL = "http://localhost:8766/sse"

# Test timeouts
LEVEL_0_TIMEOUT = 60
LEVEL_0_MIX_TIMEOUT = 120
LEVEL_0_GEMINI_TIMEOUT = 60
LEVEL_1_TIMEOUT = 180
HOOK_BLOCK_TIMEOUT = 90
RESUME_TIMEOUT = 180
CANCEL_SEARCH_DELAY = 10

# All available tests (run in this order)
ALL_TESTS = [
    "level_0",
    "level_0_mix",
    "level_0_gemini",
    "level_1",
    "hook_block",
    "resume",
    "cancel",
]


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    duration: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ContainerManager:
    """Manages Podman Compose operations."""

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
        """Clean temporary volumes."""
        paths = ["/tmp/prism-data", "/tmp/prism-cache", "/tmp/prism-postgres"]
        for path in paths:
            subprocess.run(["rm", "-rf", path], capture_output=True, check=False)
        for path in ["/tmp/prism-data", "/tmp/prism-cache"]:
            subprocess.run(["mkdir", "-p", path], capture_output=True, check=False)

    def up(self) -> bool:
        """Start containers and wait for healthy state."""
        result = subprocess.run(
            ["podman", "compose", "-f", str(self.compose_file), "up", "-d"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to start containers: {result.stderr}")
            return False
        return True

    def wait_healthy(self, timeout: int = 60) -> bool:
        """Wait for container to become healthy."""
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
                print(f"Container unhealthy after {time.time() - start:.1f}s")
                return False
            time.sleep(2)

        print(f"Timeout waiting for container health after {timeout}s")
        return False


class TestRunner:
    """Runs E2E tests using FastMCP Client."""

    def __init__(self, sse_url: str) -> None:
        self.sse_url = sse_url
        self.results: list[TestResult] = []
        self._level_1_session_id: str | None = None

    def _parse_yaml_response(self, text: str) -> dict[str, Any]:
        """Parse YAML response from MCP tool."""
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return {"content": text, "success": True}

    async def _run_l0_search(
        self,
        *,
        name: str,
        query: str,
        timeout: int,
        providers: list[str] | None = None,
        min_content_length: int = 100,
    ) -> TestResult:
        """
        Shared helper for Level 0 search tests.

        Args:
            name: Test name for reporting
            query: Search query
            timeout: Timeout in seconds
            providers: Optional provider list (None = config default)
            min_content_length: Minimum expected content length
        """
        start = time.time()

        try:
            params: dict[str, Any] = {"query": query, "level": 0}
            if providers is not None:
                params["providers"] = providers

            async with Client(self.sse_url) as client:
                result = await asyncio.wait_for(
                    client.call_tool("search", params),
                    timeout=timeout,
                )

            duration = time.time() - start

            response_text = result.content[0].text if result.content else ""
            parsed = self._parse_yaml_response(response_text)

            if not parsed.get("success", False):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                )

            content = parsed.get("content", "")
            if len(content) < min_content_length:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=(
                        f"Content too short: {len(content)} chars"
                        f" (expected >{min_content_length})"
                    ),
                    details={"content_length": len(content)},
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                details={
                    "content_length": len(content),
                    "session_id": parsed.get("session_id"),
                },
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

    async def test_level_0(self) -> TestResult:
        """
        Test Level 0 search with default provider (claude_search via config).

        Validates:
        - success=True
        - Content length > 100 characters
        """
        return await self._run_l0_search(
            name="level_0",
            query=LEVEL_0_QUERY,
            timeout=LEVEL_0_TIMEOUT,
        )

    async def test_level_0_mix(self) -> TestResult:
        """
        Test Level 0 search with all 4 providers in parallel ("mix").

        Validates:
        - success=True
        - Content present from multi-provider synthesis
        """
        return await self._run_l0_search(
            name="level_0_mix",
            query=LEVEL_0_MIX_QUERY,
            timeout=LEVEL_0_MIX_TIMEOUT,
            providers=["mix"],
        )

    async def test_level_0_gemini(self) -> TestResult:
        """
        Test Level 0 search with gemini_search provider only.

        Validates:
        - success=True
        - Content length > 100 characters
        """
        return await self._run_l0_search(
            name="level_0_gemini",
            query=LEVEL_0_GEMINI_QUERY,
            timeout=LEVEL_0_GEMINI_TIMEOUT,
            providers=["gemini_search"],
        )

    async def test_level_1(self) -> TestResult:
        """
        Test Level 1 search (parallel workers via manager).

        Validates:
        - success=True
        - Content length > 100 characters
        - session_id present (saved for resume test)
        """
        start = time.time()
        name = "level_1"

        try:
            async with Client(self.sse_url) as client:
                result = await asyncio.wait_for(
                    client.call_tool("search", {"query": LEVEL_1_QUERY, "level": 1}),
                    timeout=LEVEL_1_TIMEOUT,
                )

            duration = time.time() - start

            response_text = result.content[0].text if result.content else ""
            parsed = self._parse_yaml_response(response_text)

            if not parsed.get("success", False):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                )

            content = parsed.get("content", "")
            session_id = parsed.get("session_id")
            self._level_1_session_id = session_id

            if len(content) < 100:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=f"Content too short: {len(content)} chars (expected >100)",
                    details={"content_length": len(content), "session_id": session_id},
                )

            if not session_id:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error="No session_id in response",
                    details={"content_length": len(content)},
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                details={
                    "content_length": len(content),
                    "session_id": session_id,
                },
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {LEVEL_1_TIMEOUT}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_hook_block(self) -> TestResult:
        """
        Test that the time hook blocks tool calls when budget is exhausted.

        Sends a query designed to trigger tool usage. With a short tool budget
        (~1s), the hook should block on first tool call attempt.

        Validates:
        - success=True (model responds despite blocking)
        - Container JSON logs show hook_blocks > 0
        """
        start = time.time()
        name = "hook_block"

        try:
            async with Client(self.sse_url) as client:
                result = await asyncio.wait_for(
                    client.call_tool(
                        "search",
                        {"query": HOOK_BLOCK_QUERY, "level": 0},
                    ),
                    timeout=HOOK_BLOCK_TIMEOUT,
                )

            duration = time.time() - start

            response_text = result.content[0].text if result.content else ""
            parsed = self._parse_yaml_response(response_text)

            if not parsed.get("success", False):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                )

            # Check container logs for hook_blocks > 0
            hook_blocks = find_hook_blocks_in_logs(CONTAINER_NAME)

            details: dict[str, Any] = {
                "content_length": len(parsed.get("content", "")),
                "hook_blocks_in_logs": hook_blocks,
            }

            if hook_blocks == 0:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error="No hook_blocks found in container logs (expected > 0)",
                    details=details,
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                details=details,
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {HOOK_BLOCK_TIMEOUT}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_resume(self) -> TestResult:
        """
        Test sequential L1 query handling.

        Runs a second L1 search after the first (from test_level_1) to verify
        the server can handle sequential requests without issues.
        """
        start = time.time()
        name = "resume"

        try:
            async with Client(self.sse_url) as client:
                result = await asyncio.wait_for(
                    client.call_tool("search", {"query": RESUME_QUERY, "level": 1}),
                    timeout=RESUME_TIMEOUT,
                )

            duration = time.time() - start

            response_text = result.content[0].text if result.content else ""
            parsed = self._parse_yaml_response(response_text)

            if not parsed.get("success", False):
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=parsed.get("error", "Unknown error"),
                )

            content = parsed.get("content", "")
            if len(content) < 100:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=f"Content too short: {len(content)} chars",
                    details={
                        "content_length": len(content),
                        "prior_session": self._level_1_session_id,
                    },
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                details={
                    "content_length": len(content),
                    "prior_session": self._level_1_session_id,
                    "session_id": parsed.get("session_id"),
                },
            )

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=f"Timeout after {RESUME_TIMEOUT}s",
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def test_cancel(self) -> TestResult:
        """
        Test search cancellation.

        Starts a Level 1 search, waits, then calls cancel_all.
        Validates clean termination with no errors in logs.
        """
        start = time.time()
        name = "cancel"

        try:
            async with Client(self.sse_url) as client:
                # Start a Level 1 search in background
                search_task = asyncio.create_task(
                    client.call_tool("search", {"query": LEVEL_1_QUERY, "level": 1})
                )

                # Wait before cancelling
                await asyncio.sleep(CANCEL_SEARCH_DELAY)

                # Cancel all active searches
                cancel_result = await client.call_tool("cancel_all", {})
                cancel_text = (
                    cancel_result.content[0].text if cancel_result.content else ""
                )
                cancel_parsed = self._parse_yaml_response(cancel_text)

                # Wait for the search to complete (should be cancelled or finish)
                try:
                    await asyncio.wait_for(search_task, timeout=10)
                except asyncio.TimeoutError:
                    pass

            duration = time.time() - start

            # Check container logs for clean termination
            log_issues = check_container_logs(CONTAINER_NAME)
            errors = [i for i in log_issues if i.startswith("ERROR:")]

            if errors:
                return TestResult(
                    name=name,
                    passed=False,
                    duration=duration,
                    error=f"Errors in container logs: {errors[0][:100]}",
                    details={
                        "cancelled_count": cancel_parsed.get("cancelled_count", 0),
                        "log_errors": len(errors),
                    },
                )

            return TestResult(
                name=name,
                passed=True,
                duration=duration,
                details={
                    "cancelled_count": cancel_parsed.get("cancelled_count", 0),
                    "cancel_message": cancel_parsed.get("message", ""),
                },
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                error=str(e),
            )

    async def run_tests(self, tests: list[str]) -> list[TestResult]:
        """Run specified tests in order."""
        test_methods = {
            "level_0": self.test_level_0,
            "level_0_mix": self.test_level_0_mix,
            "level_0_gemini": self.test_level_0_gemini,
            "level_1": self.test_level_1,
            "hook_block": self.test_hook_block,
            "resume": self.test_resume,
            "cancel": self.test_cancel,
        }

        results = []
        for test_name in tests:
            if test_name not in test_methods:
                print(f"Unknown test: {test_name}")
                continue

            print(f"\nRunning {test_name}...")
            result = await test_methods[test_name]()
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"  {status} ({result.duration:.1f}s)")
            if result.error:
                print(f"  Error: {result.error}")

        return results


def check_logs_for_test(test_name: str, is_cancel: bool = False) -> list[str]:
    """
    Check container logs after a test.

    Args:
        test_name: Name of the test (for reporting)
        is_cancel: If True, warnings are acceptable (cancel is intentional)

    Returns:
        List of issues (empty if clean)
    """
    issues = check_container_logs(CONTAINER_NAME)

    if is_cancel:
        issues = [i for i in issues if i.startswith("ERROR:")]

    return issues


def print_results(results: list[TestResult], log_issues: dict[str, list[str]]) -> int:
    """Print test results summary and return exit code."""
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if result.passed:
            passed += 1
        else:
            failed += 1

        print(f"\n{result.name}: {status} ({result.duration:.1f}s)")

        if result.error:
            print(f"  Error: {result.error}")

        if result.details:
            for key, value in result.details.items():
                if key != "cancel_result":
                    print(f"  {key}: {value}")

        issues = log_issues.get(result.name, [])
        if issues:
            print(f"  Log issues ({len(issues)}):")
            for issue in issues[:5]:
                print(f"    - {issue[:100]}")

    print("\n" + "-" * 60)
    print(f"TOTAL: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Prism E2E tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tests: {', '.join(ALL_TESTS)}

Examples:
  python run_e2e.py                    # Run all tests
  python run_e2e.py --only level_0     # Run only level_0
  python run_e2e.py --skip cancel      # Skip cancel test
""",
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


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine which tests to run
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
            print(f"Available: {ALL_TESTS}")
            return 1
        tests = [t for t in ALL_TESTS if t not in skip]
    else:
        tests = ALL_TESTS.copy()

    if not tests:
        print("No tests to run")
        return 0

    print(f"Tests to run: {tests}")

    # Container setup
    container = ContainerManager(COMPOSE_FILE)

    if not args.no_docker:
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

    # Run tests
    runner = TestRunner(SSE_URL)
    results = await runner.run_tests(tests)

    # Check logs after each test
    log_issues: dict[str, list[str]] = {}
    for result in results:
        issues = check_logs_for_test(result.name, is_cancel=(result.name == "cancel"))
        if issues:
            log_issues[result.name] = issues

    # Print results
    exit_code = print_results(results, log_issues)

    # Container stays alive after tests (no docker down)
    print("\nContainer left running for inspection.")

    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
