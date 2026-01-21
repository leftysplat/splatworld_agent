---
name: splatworld-agent:review
description: Interactively review and rate generated images
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Open an interactive review session to rate generated images.
</objective>

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli review --unrated
```

This opens each unrated image and prompts for feedback:
- ++ = love it (will be converted to splat)
- + = like it
- - = not great
- -- = hate it
- s = skip
- q = quit

After review, remind user to run `/splatworld-agent:learn` if they have 3+ feedback entries.
