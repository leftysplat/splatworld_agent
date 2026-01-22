---
name: splatworld-agent:review
description: Interactively review and rate generated images
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli review --unrated
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask your own confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT describe what ratings mean
- Do NOT offer to help with ratings

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles EVERYTHING interactively:
   - Opens each unrated image
   - Prompts for feedback (++/+/-/--/s/q)
   - Records ratings
3. After CLI completes, remind user about `/splatworld-agent:learn`

The CLI will prompt the user directly. You do not control this.
