---
name: splatworld-agent:resume-work
description: Resume work from previous session
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Resume work from a previous session, showing history and current status.
</objective>

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli resume-work
```

This will:
1. Show recent session history (last 5 sessions)
2. Display notes from the last session if any
3. Show current profile status (calibration, stats)
4. Highlight unrated generations and images ready for conversion
5. Start a new session (creates `current_session.json`)
6. Display welcome panel with quick commands

### What gets tracked in a session

- Generations created
- Feedback/ratings given
- 3D conversions performed
- Learn cycles run
- Last prompt used

### Session workflow

1. Start with `/splatworld-agent:resume-work`
2. Do your work (generate, review, convert, etc.)
3. End with `/splatworld-agent:exit` to save your session

### If a session is already active

If you run `resume-work` while a session is already active, it will show the current session info and suggest using `exit` first.
