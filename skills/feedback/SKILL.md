---
name: feedback
description: Provide feedback on a generation
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli feedback "USER_RATING"
```

Pass the user's rating directly. Examples:
- User says `/feedback ++` → run `... feedback "++"`
- User says `/feedback --` → run `... feedback "--"`
- User says `/feedback "too dark"` → run `... feedback "too dark"`

## FORBIDDEN ACTIONS

- Do NOT interpret or explain the rating
- Do NOT summarize the output
- Do NOT ask confirmation questions
- Do NOT intercept the CLI interaction

## CORRECT BEHAVIOR

1. Parse user's rating argument
2. Run the single bash command above
3. The CLI handles recording the feedback

The CLI handles feedback storage. You do not control this.
