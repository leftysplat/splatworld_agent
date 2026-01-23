---
name: exit
description: Save session and exit SplatWorld Agent
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli exit
```

If user provides flags, pass them through:
- `--summary "text"` → add custom summary
- `--notes "text"` → add notes for next session

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT add your own session summary

## CORRECT BEHAVIOR

1. Run the single bash command above
2. The CLI handles EVERYTHING:
   - Calculates session activity
   - Generates summary
   - Saves session to history
   - Displays farewell

The CLI handles session management. You do not control this.
