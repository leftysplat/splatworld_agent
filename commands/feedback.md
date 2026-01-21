---
name: splatworld-agent:feedback
description: Provide feedback on a generation
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Record feedback on the most recent generation.
</objective>

## Arguments

Rating: ++ (love), + (like), - (meh), -- (hate)
Or text feedback

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli feedback "RATING"
```

Examples:
- `/splatworld-agent:feedback ++` - Love it
- `/splatworld-agent:feedback --` - Hate it
- `/splatworld-agent:feedback "too dark, prefer warmer tones"` - Text feedback
