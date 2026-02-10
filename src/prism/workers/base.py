"""
Base classes for worker agents.

Provides Agent ABC and ExecutorProtocol for dependency injection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from prism.core.response import ExecutionRequest, ExecutionResult


class ExecutorProtocol(Protocol):
    """
    Protocol for executors to enable DI and testing.

    Both ClaudeExecutor and RetryExecutor implement this interface.
    """

    async def execute(
        self,
        request: ExecutionRequest,
        schema: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a request and return result."""
        ...


@dataclass
class AgentResult:
    """
    Result from an agent execution.

    Attributes:
        success: Whether the agent completed successfully
        content: The agent's output (text or parsed JSON)
        raw_output: Original raw output from execution
        session_id: Claude session ID for potential resume
        error: Error message if failed
        metadata: Additional data from execution
    """

    success: bool
    content: str | dict[str, Any]
    raw_output: str = ""
    session_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_success(
        cls,
        content: str | dict[str, Any],
        raw_output: str = "",
        session_id: str | None = None,
        **metadata: Any,
    ) -> AgentResult:
        """Create a successful result."""
        return cls(
            success=True,
            content=content,
            raw_output=raw_output,
            session_id=session_id,
            metadata=metadata,
        )

    @classmethod
    def from_error(
        cls,
        error: str,
        raw_output: str = "",
        **metadata: Any,
    ) -> AgentResult:
        """Create an error result."""
        return cls(
            success=False,
            content="",
            raw_output=raw_output,
            error=error,
            metadata=metadata,
        )


class Agent(ABC):
    """
    Abstract base class for all worker agents.

    Agents are responsible for executing search tasks using their
    specific backend (Claude tools, Tavily MCP, Perplexity API).

    Subclasses must implement:
    - execute(): Run the agent with a prompt
    - agent_type: Property returning agent type identifier
    """

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        ...

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> AgentResult:
        """
        Execute the agent with the given prompt.

        Args:
            prompt: The search/task prompt
            timeout_seconds: Optional timeout override

        Returns:
            AgentResult with content or error
        """
        ...

    @property
    def is_cancellable(self) -> bool:
        """
        Whether this agent can be cancelled mid-execution.

        Most agents are cancellable. Perplexity is not (instant response).
        """
        return True
