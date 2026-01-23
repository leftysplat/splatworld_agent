---
name: download-splats
description: Download splat files that haven't been downloaded yet
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Download splat files from WorldLabs for conversions that don't have local splat files yet.
</objective>

## Your task

First show what splats are missing:

```bash
export PYTHONPATH=~/.claude/splatworld
python3 -m splatworld_agent.cli download-splats
```

If user confirms, the command will download the missing splats.

To download all without confirmation:

```bash
python3 -m splatworld_agent.cli download-splats --all
```

Note: Downloading splats may require a premium WorldLabs account. Requires WORLDLABS_API_KEY environment variable.
