---
name: splatworld-agent:convert
description: Convert loved images to 3D splats
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Convert images rated ++ (love) to 3D Gaussian splats using the Marble API.
</objective>

## Your task

First show what would be converted:

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli convert --dry-run
```

If user confirms, run the actual conversion:

```bash
python3 -m splatworld_agent.cli convert
```

Note: Each conversion costs $1.50 via the Marble API. Requires WORLDLABS_API_KEY environment variable.
