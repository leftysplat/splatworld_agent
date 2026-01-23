---
name: splatworld:profile
description: View or edit your taste profile
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli profile
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT explain what the profile means

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI displays the profile information

The CLI handles profile display. You do not control this.
