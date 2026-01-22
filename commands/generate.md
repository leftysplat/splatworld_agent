---
name: splatworld-agent:generate
description: Generate a single image + splat from prompt
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli generate "USER_PROMPT" [USER_FLAGS]
```

Pass the user's arguments directly. Examples:
- User says `/generate "modern kitchen"` → run `... generate "modern kitchen"`
- User says `/generate "beach" --no-splat` → run `... generate "beach" --no-splat`

## FORBIDDEN ACTIONS

- Do NOT enhance or modify the user's prompt
- Do NOT summarize or interpret the output
- Do NOT ask your own confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT describe what the CLI is doing

## CORRECT BEHAVIOR

1. Parse user arguments
2. Run the single bash command above
3. The CLI handles EVERYTHING else

The CLI handles prompt enhancement, generation, and output. You do not control this.
