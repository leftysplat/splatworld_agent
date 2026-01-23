---
name: convert
description: Convert loved images to 3D splats
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Convert Command

This command requires user input. Follow these steps exactly.

## Step 1: Show available generations

Run this command to list what's available:

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli convert --list
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
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli convert --all-loved
```

**If user chose "Pick one":**
Ask them to paste the generation ID, then run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli convert -g "GENERATION_ID"
```

## Step 4: STOP

After CLI prints "Conversion complete!" — **STOP IMMEDIATELY**.

Tell user the conversion finished and show any viewer URLs from the output. Then END your response. Do not run any more tools. Do not wait for anything.

## FORBIDDEN ACTIONS

- Do NOT skip asking the user
- Do NOT guess which generation to convert
- Do NOT run any tools after conversion completes
- Do NOT wait or hang — just respond and stop
