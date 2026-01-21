---
name: splatworld-agent:init
description: Initialize .splatworld/ in current project directory
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli init*)
---

<objective>
Initialize a new SplatWorld Agent project in the current directory.

This creates the .splatworld/ folder with an empty taste profile ready for training.
</objective>

## Your task

Run the splatworld-agent init command:

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli init
```

After initialization, inform the user:
1. The .splatworld/ folder has been created
2. Their next step is to run `/splatworld-agent:train "prompt"` to calibrate their taste profile
3. They need 20 rated images before the profile is calibrated
