Your search agents have completed their tasks. Synthesize the findings into a response.

## Agent Results

{worker_results}

## Synthesis Guidelines

- Cross-reference findings across agents -- information confirmed by multiple sources is more trustworthy
- Explicitly flag contradictions, inconsistencies, or gaps in coverage
- Distinguish clearly: high confidence (multiple sources) vs single-source claims
- Include sources and citations where agents provided them
- For each agent, note the wall time and whether it timed out -- include this transparency in your response
- Timed-out agents may have incomplete results -- flag this

## Recommended Response Structure

These are guidelines, not a strict template. Adapt to what fits the query best:

- **Direct answer** to the query upfront
- **Key findings** organized by confidence level or topic
- **Sources** with URLs where available
- **Limitations** -- what wasn't covered, timed-out agents, low-confidence areas
- **Agent timing** -- how long each took, any timeouts

{level_guidance}

## Required Output Format

```json
{"response": "Your complete response text here"}
```
