---
name: splatworld-agent:init
description: Initialize .splatworld/ in current project directory
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# Initialize SplatWorld Agent

This command requires API keys before initialization. Ask the user for each key, then run the CLI.

## Step 1: Collect API keys

Ask the user to paste their API keys one at a time:

1. "Please paste your **Anthropic API key** (for taste learning):"
2. "Please paste your **Google API key** (for Gemini image generation):"
3. "Please paste your **World Labs API key** (for 3D splat conversion):"

## Step 2: Save API keys

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli setup-keys --anthropic "ANTHROPIC_KEY" --google "GOOGLE_KEY" --worldlabs "WORLDLABS_KEY"
```

Replace placeholders with actual keys.

## Step 3: Initialize project

```bash
export PYTHONPATH=~/.claude/splatworld-agent && python3 -m splatworld_agent.cli init
```

## FORBIDDEN ACTIONS

- Do NOT add extra commentary beyond key collection
- Do NOT intercept CLI output
- Do NOT explain what the CLI is doing

## After Success

Tell the user: "Run `/splatworld-agent:train \"your prompt\"` to start training."
