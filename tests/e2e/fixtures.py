"""Test fixtures and query constants for E2E tests."""

# Test 1: L0 default provider (claude_search via config)
L0_DEFAULT_QUERY = "What is the current price of Bitcoin in USD?"

# Test 2: L0 mix (all 4 providers in parallel)
L0_MIX_QUERY = "Compare WebAuthn/passkeys vs TOTP for enterprise SSO deployments"

# Test 3: L1 search (manager + parallel workers + synthesis)
L1_QUERY = (
    "Find the top 5 action movies released after 2020 that are under 2 hours long "
    "and have an IMDb rating above 7. For each movie provide: title, year, exact "
    "runtime, IMDb rating, and a one-line description."
)

# Test 4: Cancel (start L1 search, cancel after 5s)
CANCEL_QUERY = "What is the current price of Bitcoin in USD?"

# Test 5: Resume (follow-up chat on L1 session from test 3)
RESUME_FOLLOW_UP = (
    "Name only one movie from the previous search result which would be the underdog and why?"
)

# Test 6: Fetch (Tavily extract wrapper)
FETCH_URL = "https://www.iana.org/help/example-domains"

# Ordered test list
ALL_TESTS = ["l0_default", "l0_gemini", "l0_mix", "l1", "cancel", "resume", "fetch"]
