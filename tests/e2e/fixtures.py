"""Test fixtures and query constants for E2E tests."""

# Level 0: Default provider (claude_search via config)
LEVEL_0_QUERY = "What are the latest Elastic agent features for fleet management?"

# Level 0 mix: All 4 providers in parallel
LEVEL_0_MIX_QUERY = "What is the current state of WebAssembly support in major browsers?"

# Level 0 gemini: Gemini search only
LEVEL_0_GEMINI_QUERY = "What are the key differences between ARM and x86 architectures?"

# Level 1: Quick search with parallel workers
LEVEL_1_QUERY = "How do Elastic agents connect to Fleet server and what are best practices?"

# Hook block: Query designed to trigger tool usage (which the time hook should block)
HOOK_BLOCK_QUERY = (
    "Search the web for the latest Python 3.13 release notes "
    "and summarize the new features you find"
)

# Resume/follow-up query (tests sequential L1 handling)
RESUME_QUERY = (
    "tell me about Fleet and how we can set in Fleet "
    "the output of agents to tune the performance"
)
