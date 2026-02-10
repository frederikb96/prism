"""Test fixtures and query constants for E2E tests."""

# Level 0: Direct Perplexity query (fast, no orchestration)
LEVEL_0_QUERY = "What are the latest Elastic agent features for fleet management?"

# Level 1: Quick search with parallel workers
LEVEL_1_QUERY = "How do Elastic agents connect to Fleet server and what are best practices?"

# Resume/follow-up query (tests session continuity concept)
RESUME_QUERY = (
    "tell me about Fleet and how we can set in Fleet "
    "the output of agents to tune the performance"
)
