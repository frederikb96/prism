"""
Core execution infrastructure for Prism.

This module provides the foundational components for CLI execution:

- ClaudeExecutor: THE single place where `claude -p` is invoked
- GeminiExecutor: THE single place where `gemini -p` is invoked
- ExecutionRequest: Immutable request dataclass
- ExecutionResult: Result with factory methods
- RetryExecutor: Two-tier retry (transient + validation) for Claude
- SessionRegistry: Thread-safe session tracking
"""

from prism.core.executor import ClaudeExecutor
from prism.core.gemini import GeminiExecutor
from prism.core.hooks import build_claude_hooks, build_gemini_settings_file, build_time_env_vars
from prism.core.logging import log_worker_completion, parse_hook_log, setup_logging
from prism.core.process import CancellableProcess, ProcessResult
from prism.core.response import ExecutionRequest, ExecutionResult
from prism.core.retry import RetryExecutor, build_validation_retry_prompt
from prism.core.session import Session, SessionRegistry

__all__ = [
    # Executors
    "ClaudeExecutor",
    "GeminiExecutor",
    # Hooks
    "build_claude_hooks",
    "build_gemini_settings_file",
    "build_time_env_vars",
    # Logging
    "log_worker_completion",
    "parse_hook_log",
    "setup_logging",
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
