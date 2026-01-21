---
name: splatworld-agent:init
description: Initialize .splatworld/ in current project directory
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

<objective>
Initialize a new SplatWorld Agent project. Collect API keys first, then create the project.
</objective>

## Your task

### Step 1: Welcome and collect API keys

First, welcome the user and ask for their API keys. Use AskUserQuestion with text input fields.

Say something like:
"I'll help you set up SplatWorld Agent. First, I need your API keys to enable image generation, learning, and 3D conversion."

Then use AskUserQuestion to collect:

**Question 1:** "Please provide your API keys to get started:"
- Option for Anthropic API key (for learning/taste synthesis)
- Option for Google API key (for Nano Banana Pro image generation)
- Option for World Labs API key (for 3D splat conversion)

Note: The user can select "Other" to paste each key. Collect all three keys.

### Step 2: Save the API keys

Once you have the keys from the user, save them:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli setup-keys --anthropic "ANTHROPIC_KEY" --google "GOOGLE_KEY" --worldlabs "WORLDLABS_KEY"
```

Replace the placeholder values with the actual keys the user provided.

### Step 3: Verify keys were saved

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli check-keys
```

### Step 4: Initialize the project

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli init
```

### Step 5: Confirm success

Tell the user:
1. ✓ API keys configured
2. ✓ .splatworld/ folder created
3. Next step: Run `/splatworld-agent:train "your prompt"` to calibrate your taste profile
4. They need 20 rated images before the profile is calibrated
