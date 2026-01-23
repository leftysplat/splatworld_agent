---
name: learn
description: Synthesize feedback into updated preferences
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli learn
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
