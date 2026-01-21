---
name: splatworld-agent:init
description: Initialize .splatworld/ in current project directory
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Initialize a new SplatWorld Agent project. Collect API keys first, then create the project.
</objective>

## Your task

### Step 1: Ask for API keys

Tell the user you need 3 API keys to set up SplatWorld Agent, then ask them to paste each one.

Say:
"I'll set up SplatWorld Agent for you. Please paste your API keys one at a time:

**Anthropic API key** (for taste learning):"

Wait for the user to paste their Anthropic key.

Then ask:
"**Google API key** (for Nano Banana Pro image generation):"

Wait for the user to paste their Google key.

Then ask:
"**World Labs API key** (for 3D splat conversion):"

Wait for the user to paste their World Labs key.

### Step 2: Save the API keys

Once you have all three keys, save them:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli setup-keys --anthropic "ANTHROPIC_KEY" --google "GOOGLE_KEY" --worldlabs "WORLDLABS_KEY"
```

Replace the placeholders with the actual keys the user provided.

### Step 3: Initialize the project

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli init
```

### Step 4: Confirm success

Tell the user:
- ✓ API keys configured
- ✓ .splatworld/ folder created
- Next: Run `/splatworld-agent:train "your prompt"` to start training (20 images needed)
