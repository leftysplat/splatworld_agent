---
name: splatworld-agent:convert
description: Convert loved images to 3D splats
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Convert Command

This command requires user input. Follow these steps exactly.

## Step 1: Show available generations

Run this command to list what's available:

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli convert --list
```

## Step 2: Ask user what to convert

Use AskUserQuestion with:
- header: "Convert"
- question: "Which generation(s) do you want to convert to 3D splats?"
- options:
  - "Convert all loved (++)" — Convert all images rated ++
  - "Pick one" — I'll paste the generation ID

## Step 3: Run conversion

**If user chose "Convert all loved":**
```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli convert --all-loved
```

**If user chose "Pick one":**
Ask them to paste the generation ID, then run:
```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli convert -g "GENERATION_ID"
```

## FORBIDDEN ACTIONS

- Do NOT skip asking the user
- Do NOT guess which generation to convert
- Do NOT intercept or summarize CLI output beyond what's needed
