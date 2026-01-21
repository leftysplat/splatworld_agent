---
name: splatworld-agent:init
description: Initialize .splatworld/ in current project directory
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

<objective>
Initialize a new SplatWorld Agent project in the current directory, ensuring API keys are configured first.
</objective>

## Your task

### Step 1: Check API keys

First, check if API keys are configured:

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli check-keys
```

### Step 2: If keys are missing, prompt the user

If any required keys are missing, use AskUserQuestion to collect them. Required keys:
- **ANTHROPIC_API_KEY** - For learning/taste synthesis (required)
- **WORLDLABS_API_KEY** - For 3D splat conversion (required for convert)
- **GOOGLE_API_KEY** or **NANO_API_KEY** - For image generation (at least one required)

Example prompt: "I need to configure your API keys before initializing. Please provide your API keys:"

### Step 3: Save the API keys

If the user provided keys, save them:

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli setup-keys --anthropic "KEY" --google "KEY" --worldlabs "KEY"
```

### Step 4: Initialize the project

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli init
```

### Step 5: Inform the user

After initialization, tell the user:
1. The .splatworld/ folder has been created
2. Their API keys are configured
3. Next step: Run `/splatworld-agent:train "prompt"` to calibrate their taste profile
4. They need 20 rated images before the profile is calibrated
