**google_web_search** — Google's grounding search with high-quality, current results (~6.6s per call)

**Iteration timing:** Each search call takes ~6-7s. Your own processing adds to that, so observe the actual time deltas from hook messages to calibrate your pace.

**Parallel calls:** You can issue multiple google_web_search calls simultaneously — they execute in parallel, sharing the wall-clock time. Use this to cover multiple angles at once.
