---
name: cancel
description: Cancel the current SplatWorld action
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli cancel
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles cancellation

The CLI handles cancellation. You do not control this.
