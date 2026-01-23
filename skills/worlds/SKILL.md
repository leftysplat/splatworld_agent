---
name: worlds
description: List all worlds from your Marble/WorldLabs account
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*)
---

# Worlds Command

Lists all 3D worlds from your Marble/WorldLabs account via the API.

## Step 1: Run the command

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli worlds
```

## Step 2: Display results

Show the user the list of worlds with their viewer URLs.

If they want to open a specific world:
```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli worlds --open "WORLD_ID"
```

## FORBIDDEN ACTIONS

- Do NOT add commentary beyond what the CLI outputs
- Do NOT run any other tools after displaying results
