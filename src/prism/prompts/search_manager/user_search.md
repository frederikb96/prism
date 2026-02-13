{current_datetime}

## Search Query

{query}

## Available Agents

{agent_section}

## Search Strategy

- **Cross-referencing**: Create overlapping queries across agents -- reliability comes from independent verification, not trusting any single source
- **Doubt everything**: Multiple agent layers can produce false information -- build reliability from cross-referencing rather than assuming correctness
- **Specific queries**: Vague queries produce vague results -- each agent slot must have a clear, focused purpose
- **Self-contained prompts**: Each agent query must be fully self-contained with all necessary context -- agents cannot see each other or your planning

{level_guidance}

## Required Output Format

Output a JSON object with one key per agent slot. Each value is the complete search prompt for that agent.

```json
{schema_example}
```
