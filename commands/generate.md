---
name: splatworld-agent:generate
description: Generate a single image + splat from prompt
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Generate a single image from a prompt, optionally converting to 3D splat.
</objective>

## Arguments

Required: prompt
Optional: --no-splat (skip 3D conversion), --no-enhance (don't use taste profile)

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli generate "USER_PROMPT"
```

After generation, remind user to provide feedback with `/splatworld-agent:feedback ++` or `--`.

Note: Requires NANOBANANA_API_KEY (or GOOGLE_API_KEY). WORLDLABS_API_KEY needed for 3D conversion.
