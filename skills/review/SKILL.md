---
name: review
description: Interactively review and rate generated images
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*), AskUserQuestion
---

# Review Command

This command shows unrated images and collects ratings. Follow these steps.

## Step 1: List unrated images

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli review --list
```

If all images are rated, stop here.

## Step 2: For each unrated image, ask for rating

Use AskUserQuestion with:
- header: "Rate"
- question: "Rate this image: [GENERATION_ID] - [PROMPT]?"
- options:
  - "++" — Love it (will convert to 3D splat)
  - "+" — Good
  - "-" — Not great
  - "--" — Hate it
  - "Skip" — Don't rate this one
  - "Done" — Stop reviewing

## Step 3: Record the rating

For each rating (not Skip or Done):
```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli review --rate "RATING" -g "GENERATION_ID"
```

## Step 4: After review

Tell user:
- How many images were rated
- If they have 3+ ratings, suggest `/splatworld:learn`
- If they have loved images, suggest `/splatworld:convert`

## FORBIDDEN ACTIONS

- Do NOT skip asking the user for each rating
- Do NOT guess ratings
