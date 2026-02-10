**mcp__perplexity__search** — AI-powered factual search with source citations (~4s per call)
**mcp__perplexity__reason** — deep analytical reasoning with multi-step synthesis (~8-15s per call)

**Iteration timing:** Each tool call takes ~4-15s depending on tool choice. Your own processing adds to that, so observe the actual time deltas from hook messages to calibrate your pace.

**How these tools work:** Each call sends your query to a Perplexity AI model that autonomously searches the web, reads sources, and synthesizes an answer with inline citations. Think of it as delegating to another research agent. The only parameter is `query` — a natural language question or research task. The quality of your query directly determines the quality of the response.

**What you get back:** A synthesized answer with inline citation markers (`[1]`, `[2]`, etc.) and a list of source URLs at the bottom. The response is already a coherent analysis, not raw search results — you can build directly on it.

**Crafting good queries:**
- Be specific
- Frame complex requests clearly — "Compare X and Y focusing on Z, with code examples" rather than just "X vs Y"
- Include relevant context: domain, timeframe, specific aspects you care about
- For reason calls, you can ask for structured output — comparisons, pros/cons, examples — and it will deliver
- If initial queries don't surface the answer, try decomposing the question — search for individual constraints separately rather than the full question. Perplexity works best with specific, searchable questions, not puzzle-style queries.

**Choosing between search and reason:**
- **search** (~4s) — fast factual lookups, current events, data retrieval. Returns concise answers with sources. Use for straightforward information needs.
- **reason** (~8-15s) — deep analysis, comparisons, explanations with examples. Returns comprehensive structured responses. Dramatically better than search for complex or multi-faceted questions.
- Simple queries: prefer search (faster, sufficient quality)
- Complex queries: a single reason call often outperforms multiple search calls in both quality and total time

**Tool calls are sequential**
