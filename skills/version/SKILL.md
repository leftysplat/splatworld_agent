---
name: version
description: Show SplatWorld Agent version and install info
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli version
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT add extra commentary

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI outputs version info directly
