"""
Core execution infrastructure for Prism.

This module provides the foundational components for Claude CLI execution:

- ClaudeExecutor: THE single place where `claude -p` is invoked
- ExecutionRequest: Immutable request dataclass
- ExecutionResult: Result with factory methods
- RetryExecutor: Two-tier retry (transient + validation)
- SessionRegistry: Thread-safe session tracking
"""

from prism.core.executor import ClaudeExecutor
from prism.core.process import CancellableProcess, ProcessResult
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.retry import RetryExecutor, build_validation_retry_prompt
from prism.core.session import Session, SessionRegistry

__all__ = [
    # Executor
    "ClaudeExecutor",
    # Request/Response
    "ExecutionRequest",
    "ExecutionResult",
    # Retry
    "RetryExecutor",
    "build_validation_retry_prompt",
    # Process
    "CancellableProcess",
    "ProcessResult",
    # Session
    "Session",
    "SessionRegistry",
]
