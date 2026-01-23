---
name: train
description: Guided training mode to calibrate your taste profile (20 images)
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Train Command

Training generates images one at a time and asks for ratings. Follow these steps.

## Step 1: Parse user arguments

User will provide a prompt and optional count:
- `/train "cozy cabin"` -> prompt="cozy cabin"
- `/train 5 "cozy cabin"` -> count=5, prompt="cozy cabin"

## Step 2: Generate ONE image

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli train "USER_PROMPT" --single --json --no-rate
```

This outputs JSON with image details. Parse the JSON response.

### Handle Provider Failures (IGEN-02)

If the command fails with a provider error message containing "Provider nano failed" or similar:

1. Use AskUserQuestion:
   - header: "Provider Unavailable"
   - question: "Nano Banana Pro is unavailable. Switch to Gemini to continue?"
   - options:
     - "yes" - Yes, switch to Gemini
     - "no" - No, stop training

2. If user chooses "yes":
   ```bash
   export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli switch-provider gemini
   ```
   Then retry the generation.

3. If user chooses "no":
   Stop training and show summary.

### Handle Credit Warning (IGEN-03)

After successful generation, check JSON output for `usage_percentage >= 75`.

If at 75%+, use AskUserQuestion:
- header: "Credit Warning"
- question: "75% of Nano credits used. Switch to Gemini to conserve credits?"
- options:
  - "switch" - Yes, switch to Gemini
  - "continue" - No, keep using Nano

If user chooses "switch":
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli switch-provider gemini
```

## Step 3: Ask for rating

Use AskUserQuestion with:
- header: "Rate"
- question: "Rate this image (view the file path shown above)?"
- options:
  - "++" - Love it!
  - "+" - Good
  - "-" - Not great
  - "--" - Hate it
  - "Skip" - Don't rate, continue
  - "Done" - Stop training

## Step 4: Record the rating

If user gave a rating (not Skip or Done):
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli review --rate "RATING" -g "GENERATION_ID"
```

## Step 5: Loop or finish

- If user chose "Done" -> Stop and show summary
- If user specified a count and reached it -> Stop and show summary
- Otherwise -> Go back to Step 2 (generate next image)

## After training

Tell user:
- How many images were generated and rated
- Current provider used
- If they have 3+ ratings, learning happens automatically
- Suggest `/splatworld:profile` to see their taste profile

## Switching Providers Mid-Session (IGEN-04)

If user asks to switch providers during training:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli switch-provider PROVIDER
```

Then continue training with the new provider.

## FORBIDDEN ACTIONS

- Do NOT run train without --single flag (causes interactive input issues)
- Do NOT skip asking for ratings
- Do NOT guess ratings
- Do NOT automatically switch providers without asking user (IGEN-02 requires consent)
