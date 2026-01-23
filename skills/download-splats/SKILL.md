---
name: download-splats
description: Download splat files that haven't been downloaded yet
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*)
---

<objective>
Download splat files from WorldLabs for conversions that don't have local splat files yet.
</objective>

## Your task

First show what splats are missing:

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}"
python3 -m splatworld_agent.cli download-splats
```

If user confirms, the command will download the missing splats.

To download all without confirmation:

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}"
python3 -m splatworld_agent.cli download-splats --all
```

Note: Downloading splats may require a premium WorldLabs account. Requires WORLDLABS_API_KEY environment variable.
