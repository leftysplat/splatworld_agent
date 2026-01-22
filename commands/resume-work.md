---
name: splatworld-agent:resume-work
description: Resume work from previous session
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli resume-work
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT describe session history yourself

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles EVERYTHING:
   - Shows recent session history
   - Displays notes from last session
   - Shows current profile status
   - Highlights pending work
   - Starts new session

The CLI handles session restoration. You do not control this.
