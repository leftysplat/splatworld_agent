---
name: splatworld-agent:batch
description: Generate a batch of images for review
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Generate multiple images at once for efficient review and iteration.
</objective>

## Arguments

Required: prompt describing the scene
Optional: -n COUNT (default 5), -c CYCLES (default 1)

Example: `/splatworld-agent:batch "futuristic city street" -n 5 -c 2`

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli batch "USER_PROMPT" -n 5
```

After batch completes, remind user:
1. Run `/splatworld-agent:review` to rate images
2. Run `/splatworld-agent:learn` to update preferences
3. Run `/splatworld-agent:convert` to turn favorites into 3D splats

Note: Requires NANOBANANA_API_KEY (or GOOGLE_API_KEY) environment variable.
