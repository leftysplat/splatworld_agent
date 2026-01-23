---
name: splatworld:train
description: Guided training mode to calibrate your taste profile (20 images)
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Train Command

Training generates images one at a time and asks for ratings. Follow these steps.

## Step 1: Parse user arguments

User will provide a prompt and optional count:
- `/train "cozy cabin"` → prompt="cozy cabin"
- `/train 5 "cozy cabin"` → count=5, prompt="cozy cabin"

## Step 2: Generate ONE image

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli train "USER_PROMPT" --single
```

This generates one image and shows the file path and generation ID.

## Step 3: Ask for rating

Use AskUserQuestion with:
- header: "Rate"
- question: "Rate this image (view the file path shown above)?"
- options:
  - "++" — Love it!
  - "+" — Good
  - "-" — Not great
  - "--" — Hate it
  - "Skip" — Don't rate, continue
  - "Done" — Stop training

## Step 4: Record the rating

If user gave a rating (not Skip or Done):
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli review --rate "RATING" -g "GENERATION_ID"
```

## Step 5: Loop or finish

- If user chose "Done" → Stop and show summary
- If user specified a count and reached it → Stop and show summary
- Otherwise → Go back to Step 2 (generate next image)

## After training

Tell user:
- How many images were generated and rated
- If they have 3+ ratings, learning happens automatically
- Suggest `/splatworld:profile` to see their taste profile

## FORBIDDEN ACTIONS

- Do NOT run train without --single flag (causes interactive input issues)
- Do NOT skip asking for ratings
- Do NOT guess ratings
