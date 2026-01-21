---
name: splatworld-agent:exit
description: Save session and exit SplatWorld Agent
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Save the current session with activity summary and exit SplatWorld Agent.
</objective>

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli exit
```

This will:
1. Calculate session activity (generations, feedback, conversions, learns)
2. Auto-generate a summary of work accomplished
3. Append session to `sessions.jsonl` history
4. Remove `current_session.json`
5. Display farewell with session summary

### Optional flags

Add a custom summary:
```bash
python3 -m splatworld_agent.cli exit --summary "Explored alien landscape styles"
```

Add notes for next session:
```bash
python3 -m splatworld_agent.cli exit --notes "Try warmer color palettes next time"
```

To resume work later, use `/splatworld-agent:resume-work`.
