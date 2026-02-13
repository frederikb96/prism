**mcp__tavily__tavily_search** — fast multi-source search with ranked results (~0.5s per call)
**mcp__tavily__tavily_extract** — page content extraction from URLs (basic: ~1-3s, advanced: ~5-10s)

**Iteration timing:** tavily_search itself is very fast (~0.5s), so your processing time dominates each iteration. Observe the actual time deltas from hook messages to calibrate your pace.

**tavily_search parameters:**
- `query` (required) — your search query
- `search_depth` — `"advanced"` returns richer results with multiple detailed snippets per source (ratings, metadata, structured data). Worth the cost for research queries. `"basic"` returns shorter snippets. `"fast"` and `"ultra-fast"` optimize for minimal latency at similar quality to basic.
- `max_results` — number of results (default 10, min 5, max 20). 5-10 is usually sufficient; use 10 for broader coverage when you need multiple perspectives.
- `time_range` — `"day"`, `"week"`, `"month"`, `"year"` for recency filtering. Useful for current events or recent data.
- `include_domains` / `exclude_domains` — domain whitelists/blacklists (arrays of strings like `["imdb.com", "rottentomatoes.com"]`)
- `start_date` / `end_date` — date range filtering in `YYYY-MM-DD` format
- **Avoid** `include_raw_content` — it returns full page HTML/markdown per result, easily producing 100K+ characters that flood your context. Use tavily_extract instead when you need full page content.

**tavily_extract parameters:**
- `urls` (required) — list of URLs to extract content from (max 20 per call)
- `extract_depth` — `"basic"` (quick HTTP, ~1-3s) or `"advanced"` (headless browser, ~5-10s). Use advanced for pages with dynamic content, tables, or anti-bot protection.
- `query` — reranking query: when set, extracted content is ranked by relevance to this query, returning the most useful sections first. Highly recommended — gives you focused excerpts instead of a raw page dump.
- `format` — `"markdown"` (default) or `"text"`

**Strategy:**
- Use `search_depth="advanced"` by default — the richer results often eliminate the need for follow-up searches, saving more time than the extra API cost
- tavily_search is near-instant (~0.5s), so you can iterate quickly: search → analyze → refine query → search again
- When you find a promising URL in search results and need deeper content, use tavily_extract with a `query` param to get relevant sections without processing the entire page
- Tool calls are sequential
