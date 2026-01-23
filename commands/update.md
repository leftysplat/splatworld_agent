---
name: splatworld:update
description: Update SplatWorld Agent to latest version
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli update
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT run git commands manually

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles EVERYTHING:
   - Fetches updates from remote
   - Shows new commits
   - Pulls latest changes
   - Displays update summary

The CLI handles updates internally. You do not control this.
