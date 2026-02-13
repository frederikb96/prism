"""
Execution request and result dataclasses.

Frozen dataclasses for immutability - use .with_changes() for modifications.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionRequest:
    """
    Immutable request for Claude CLI execution.

    Use .with_changes() to create modified copies for retries.
    """

    prompt: str
    model: str = "sonnet"
    timeout_seconds: int | None = 60
    tools: str | None = None
    allowed_tools: tuple[str, ...] = ()
    json_schema: dict[str, Any] | None = None
    system_prompt: str | None = None
    resume_session: str | None = None
    hooks_config: dict[str, Any] | None = None
    env_vars: tuple[tuple[str, str], ...] | None = None
    mcp_config: dict[str, Any] | None = None
    strict_mcp: bool = False
    no_session_persistence: bool = False
    effort: str | None = None

    def with_changes(self, **kwargs: Any) -> ExecutionRequest:
        """Create a new request with specified fields changed."""
        return dataclasses.replace(self, **kwargs)


@dataclass
class ExecutionResult:
    """
    Result of a Claude CLI execution.

    Use factory methods for common patterns:
    - ExecutionResult.success(output, session_id)
    - ExecutionResult.error(message, exit_code)
    - ExecutionResult.timeout(timeout_seconds)
    """

    success: bool
    output: str
    session_id: str | None = None
    exit_code: int | None = None
    error_message: str | None = None
    is_timeout: bool = False
    is_cancelled: bool = False

    @classmethod
    def from_success(
        cls,
        output: str,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """Create a successful result."""
        return cls(
            success=True,
            output=output,
            session_id=session_id,
            exit_code=0,
        )

    @classmethod
    def from_error(
        cls,
        message: str,
        exit_code: int = 1,
        output: str = "",
    ) -> ExecutionResult:
        """Create an error result."""
        return cls(
            success=False,
            output=output,
            error_message=message,
            exit_code=exit_code,
        )

    @classmethod
    def from_timeout(cls, timeout_seconds: int | None) -> ExecutionResult:
        """Create a timeout result."""
        return cls(
            success=False,
            output="",
            error_message=f"Execution timed out after {timeout_seconds}s",
            is_timeout=True,
        )

    @classmethod
    def from_cancelled(cls) -> ExecutionResult:
        """Create a cancelled result."""
        return cls(
            success=False,
            output="",
            error_message="Execution was cancelled",
            is_cancelled=True,
        )

    TRANSIENT_PATTERNS: tuple[str, ...] = (
        "connection",
        "timeout",
        "temporary",
        "unavailable",
        "rate limit",
        "429",
        "503",
        "502",
    )

    def is_transient_error(self) -> bool:
        """Determine if this error is transient and worth retrying."""
        if self.is_timeout:
            return True
        error_lower = (self.error_message or "").lower()
        return any(pattern in error_lower for pattern in self.TRANSIENT_PATTERNS)
