---
name: splatworld-agent:train
description: Guided training mode to calibrate your taste profile (20 images)
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Run guided training to calibrate the taste profile.

This generates images in rounds, prompts for ratings, and learns preferences until the profile is calibrated (20+ rated images with good distribution).
</objective>

## Arguments

The user should provide a prompt describing what kind of images to generate for training.

Example: `/splatworld-agent:train "cozy cabin interior with warm lighting"`

## Your task

1. Extract the prompt from the user's command
2. Set up environment variables and run the train command:

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli train "USER_PROMPT_HERE"
```

The training is interactive - images will open for the user to review and rate.

Rating options:
- ++ = love it
- + = like it
- - = not great
- -- = hate it
- s = skip
- q = quit

Training continues until 20+ images are rated with at least 10% positive and 10% negative feedback.

Note: Requires ANTHROPIC_API_KEY, NANOBANANA_API_KEY (or GOOGLE_API_KEY), and optionally WORLDLABS_API_KEY environment variables.
