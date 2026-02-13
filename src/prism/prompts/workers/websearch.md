**WebSearch** — broad web search returning snippets from multiple sources (~7-9s per call)
**WebFetch** — AI-summarized page content from a specific URL (~5-10s per call). Returns processed/summarized content, not raw page text. Gets 403-blocked on many sites with anti-bot protection (Wikipedia, Cloudflare-protected sites, etc.). Use when you found a promising source via WebSearch and need more detail, but be aware the output is lossy. Works best on documentation sites, static pages, and simpler web properties. 

**Iteration timing:** Each tool call takes ~7-10s. Your own processing (thinking, analyzing) adds to that, so observe the actual time deltas from hook messages to calibrate your pace.

**Parallel calls:** You can issue multiple WebSearch calls simultaneously — they execute in parallel, sharing the wall-clock time. Use this to cover multiple angles at once.

