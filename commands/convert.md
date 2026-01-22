---
name: splatworld-agent:convert
description: Convert loved images to 3D splats
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli convert
```

## FORBIDDEN ACTIONS

- Do NOT run with --dry-run first
- Do NOT summarize or interpret the output
- Do NOT ask your own confirmation questions
- Do NOT intercept the CLI interaction

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles EVERYTHING interactively:
   - Shows available generations
   - Asks user to paste a generation ID or type "convert all"
   - Handles the conversion

The CLI will prompt the user directly. You do not control this.
