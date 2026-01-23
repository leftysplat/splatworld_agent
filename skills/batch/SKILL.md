---
name: batch
description: Generate a batch of images for review
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli batch "USER_PROMPT" [USER_FLAGS]
```

Pass the user's arguments directly. Examples:
- User says `/batch "futuristic city"` → run `... batch "futuristic city"`
- User says `/batch "beach" -n 10` → run `... batch "beach" -n 10`

## FORBIDDEN ACTIONS

- Do NOT create prompt variations yourself
- Do NOT call generate command multiple times
- Do NOT summarize or interpret the output
- Do NOT ask your own confirmation questions
- Do NOT intercept the CLI interaction

## CORRECT BEHAVIOR

1. Parse user arguments
2. Run the single bash command above
3. The CLI handles EVERYTHING else:
   - Creates prompt variations
   - Generates images
   - Shows progress
   - Summarizes results

The CLI handles batch generation internally. You do not control this.
