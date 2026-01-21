---
name: splatworld-agent:learn
description: Synthesize feedback into updated preferences
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Analyze all feedback using Claude and update the taste profile with learned preferences.
</objective>

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli learn
```

This will:
1. Analyze all feedback history
2. Extract preference patterns (lighting, colors, composition, etc.)
3. Update the taste profile
4. Mark profile as calibrated if 20+ ratings with good distribution

After learning, use `/splatworld-agent:profile` to see updated preferences.

Note: Requires ANTHROPIC_API_KEY environment variable.
