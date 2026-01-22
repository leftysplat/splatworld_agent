---
name: splatworld-agent:cancel
description: Cancel the current SplatWorld action
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# Cancel Current Action

Stop any ongoing SplatWorld Agent action immediately.

## What This Does

- Stops any running training session
- Cancels any pending conversions
- Halts any batch operations
- Saves current state so you can resume later

## Execution

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli cancel
```

## After Cancelling

- Training state is preserved - use `/splatworld-agent:resume-work` to continue
- Any completed work (generated images, ratings) is saved
- You can start fresh with `/splatworld-agent:train` anytime
