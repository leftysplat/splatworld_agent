---
name: learn
description: Synthesize feedback into updated preferences
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli learn
```

## FORBIDDEN ACTIONS

- Do NOT analyze feedback yourself
- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT describe what learning means

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles EVERYTHING:
   - Analyzes all feedback history
   - Extracts preference patterns
   - Updates the taste profile
   - Shows learning results

The CLI handles learning internally. You do not control this.
