---
name: splatworld:init
description: Initialize .splatworld/ in current project directory
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# Initialize SplatWorld Agent

This command requires API keys before initialization. Ask the user for each key, then run the CLI.

## Step 1: Collect API keys

Ask the user to paste their API keys one at a time. FLUX is the default image generator, but users can skip it:

1. "Please paste your **FLUX API key** (for FLUX.2 [pro] image generation), or type 'skip' to use Nano instead:"
2. "Please paste your **Nano Banana Pro API key** (fallback image generator), or type 'skip':"
3. "Please paste your **Anthropic API key** (for taste learning):"
4. "Please paste your **World Labs API key** (for 3D splat conversion):"

## Step 2: Save API keys

Build the setup-keys command with only the keys that were provided (not skipped):

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli setup-keys --bfl "BFL_KEY" --nano "NANO_KEY" --anthropic "ANTHROPIC_KEY" --worldlabs "WORLDLABS_KEY"
```

Omit any `--flag "value"` pair where the user typed "skip". Replace placeholders with actual keys.

## Step 3: Initialize project

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli init
```

Note: The init command may prompt for additional keys if not already configured. Let the CLI handle this.

## FORBIDDEN ACTIONS

- Do NOT add extra commentary beyond key collection
- Do NOT intercept CLI output
- Do NOT explain what the CLI is doing

## After Success

Tell the user: "Run `/splatworld:train \"your prompt\"` to start training."
