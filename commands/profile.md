---
name: splatworld-agent:profile
description: View or edit your taste profile
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Display the current taste profile showing calibration status, learned preferences, and statistics.
</objective>

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli profile
```

This shows:
- Calibration status (trained vs needs more data)
- Training progress (X/20 ratings)
- Learned visual style preferences
- Learned composition preferences
- Quality criteria (must-have, never)
- Rating statistics
- Prompt enhancement preview
