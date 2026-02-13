"""
Worker agents for Prism search.

Each worker handles a specific search backend:
- ClaudeSearchAgent: Claude with WebSearch/WebFetch tools
- TavilySearchAgent: Claude with Tavily MCP tools
- PerplexitySearchAgent: Claude with Perplexity MCP tools
- GeminiSearchAgent: Gemini with Google Search
- ManagerAgent: Search planning with schema validation
"""

from prism.workers.base import Agent, AgentResult, ExecutorProtocol
from prism.workers.claude_search import ClaudeSearchAgent
from prism.workers.factory import create_worker
from prism.workers.gemini_search import GeminiSearchAgent
from prism.workers.manager import ManagerAgent
from prism.workers.perplexity_search import PerplexitySearchAgent
from prism.workers.tavily_search import TavilySearchAgent

__all__ = [
    # Base
    "Agent",
    "AgentResult",
    "ExecutorProtocol",
    # Factory
    "create_worker",
    # Workers
    "ClaudeSearchAgent",
    "TavilySearchAgent",
    "PerplexitySearchAgent",
    "GeminiSearchAgent",
    "ManagerAgent",
]
