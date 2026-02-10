"""
Worker agents for Prism search.

Each worker handles a specific search backend:
- ResearcherAgent: Claude with WebSearch/WebFetch tools
- TavilyAgent: Claude with Tavily MCP tools
- PerplexityAgent: Direct Perplexity API (no Claude)
- ManagerAgent: Search planning with schema validation
"""

from prism.workers.base import Agent, AgentResult, ExecutorProtocol
from prism.workers.manager import ManagerAgent
from prism.workers.perplexity import PerplexityAgent
from prism.workers.researcher import ResearcherAgent
from prism.workers.tavily import TavilyAgent

__all__ = [
    # Base
    "Agent",
    "AgentResult",
    "ExecutorProtocol",
    # Workers
    "ResearcherAgent",
    "TavilyAgent",
    "PerplexityAgent",
    "ManagerAgent",
]
