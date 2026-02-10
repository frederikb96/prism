"""
Async subprocess wrapper with cancellation support.

Uses process groups for clean SIGTERM/SIGKILL cleanup.
"""

from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio.subprocess import Process


@dataclass
class ProcessResult:
    """Result from a completed process."""

    stdout: str
    stderr: str
    returncode: int

    @property
    def success(self) -> bool:
        """Check if process exited successfully."""
        return self.returncode == 0


class CancellableProcess:
    """
    Async subprocess wrapper with proper cleanup.

    Uses start_new_session=True to create a process group,
    enabling clean termination of the process and all children.
    """

    def __init__(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        sigterm_timeout: float = 2.0,
        sigkill_timeout: float = 5.0,
    ) -> None:
        """
        Initialize process configuration.

        Args:
            cmd: Command and arguments to execute
            env: Environment variables (merged with current env)
            timeout_seconds: Optional timeout for execution
            sigterm_timeout: Seconds to wait after SIGTERM before SIGKILL
            sigkill_timeout: Seconds to wait after SIGKILL before giving up
        """
        self._cmd = cmd
        self._env = env
        self._timeout_seconds = timeout_seconds
        self._sigterm_timeout = sigterm_timeout
        self._sigkill_timeout = sigkill_timeout
        self._process: Process | None = None
        self._cancelled = False

    async def run(self) -> ProcessResult:
        """
        Execute the process and wait for completion.

        Returns:
            ProcessResult with stdout, stderr, and return code

        Raises:
            asyncio.TimeoutError: If timeout_seconds exceeded
            asyncio.CancelledError: If cancel() was called
        """
        # Merge environment
        full_env = os.environ.copy()
        if self._env:
            full_env.update(self._env)

        self._process = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            start_new_session=True,  # Create process group for clean cleanup
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                self._process.communicate(),
                timeout=self._timeout_seconds,
            )

            return ProcessResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                returncode=self._process.returncode or 0,
            )

        except asyncio.TimeoutError:
            await self._terminate()
            raise

        except asyncio.CancelledError:
            self._cancelled = True
            await self._terminate()
            raise

    async def cancel(self) -> None:
        """
        Cancel the running process.

        Safe to call even if process hasn't started or already finished.
        """
        self._cancelled = True
        if self._process is not None:
            await self._terminate()

    async def _terminate(self) -> None:
        """
        Terminate the process group gracefully.

        Sends SIGTERM first, then SIGKILL after sigterm_timeout if needed.
        """
        if self._process is None or self._process.returncode is not None:
            return

        try:
            # Get process group ID (same as PID when start_new_session=True)
            pgid = os.getpgid(self._process.pid)

            # Send SIGTERM to process group
            os.killpg(pgid, signal.SIGTERM)

            # Wait briefly for graceful shutdown
            try:
                await asyncio.wait_for(self._process.wait(), timeout=self._sigterm_timeout)
            except asyncio.TimeoutError:
                # Force kill if still running
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    await asyncio.wait_for(self._process.wait(), timeout=self._sigkill_timeout)
                except (ProcessLookupError, asyncio.TimeoutError):
                    pass

        except ProcessLookupError:
            # Process already terminated
            pass
        except OSError:
            # Process group doesn't exist or other OS error
            pass

    @property
    def is_cancelled(self) -> bool:
        """Check if the process was cancelled."""
        return self._cancelled
