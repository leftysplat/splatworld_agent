---
name: splatworld-agent:train
description: Run adaptive training CLI (DO NOT implement yourself)
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*train*)
---

# CRITICAL: DO NOT IMPLEMENT TRAINING YOURSELF

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli train [USER_ARGS]
```

Pass the user's arguments directly to the CLI. Examples:
- User says `/train 2 "wild west"` → run `... train 2 "wild west"`
- User says `/train "alien beach"` → run `... train "alien beach"`
- User says `/train 5` → run `... train 5`

## FORBIDDEN ACTIONS

- Do NOT create prompt variations yourself
- Do NOT call `generate` command directly
- Do NOT plan multiple images upfront
- Do NOT describe what you're going to do
- Do NOT implement any training logic

## CORRECT BEHAVIOR

1. Parse user arguments
2. Run the single bash command above
3. The CLI handles EVERYTHING else interactively

The CLI will prompt the user for ratings between each image. You do not control this - the CLI does.
