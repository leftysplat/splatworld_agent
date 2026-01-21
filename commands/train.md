---
name: splatworld-agent:train
description: Guided training mode to calibrate your taste profile (20 images)
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Run guided training to calibrate the taste profile by generating creative variations of the user's prompt.
</objective>

## Arguments

The user provides a base prompt/concept. You will expand it into varied, detailed prompts.

Example: `/splatworld-agent:train "beach on an alien world"`

## Your task

### Step 1: Create prompt variations

Take the user's base concept and create 5 unique, detailed prompt variations. Each should:
- Keep the core concept but explore different interpretations
- Add specific details (lighting, atmosphere, colors, time of day, weather, materials)
- Vary the mood and style (serene vs dramatic, minimal vs detailed)
- Be 1-2 sentences, vivid and specific

Example for "beach on an alien world":
1. "A serene beach with bioluminescent turquoise sand beside a deep purple ocean, twin moons reflecting on gentle waves, alien flora dotting the shoreline"
2. "Dramatic rocky alien coastline with towering crystalline formations, violent green waves crashing against obsidian cliffs under a stormy orange sky"
3. "Minimalist alien beach at dawn, pale pink sand stretching to a mercury-like silver ocean, a single massive ringed planet dominating the horizon"
4. "Lush tropical alien beach with oversized iridescent shells scattered on golden sand, phosphorescent tide pools, dense alien jungle meeting the shore"
5. "Frozen alien beach where a methane ocean meets ammonia ice sheets, strange geometric ice formations, dim distant sun casting long blue shadows"

### Step 2: Generate images for each variation

For each of your 5 prompt variations, run:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli generate "YOUR_DETAILED_PROMPT_HERE" --no-splat
```

Show the user each image as it's generated.

### Step 3: Collect ratings

After showing each image, ask the user to rate it:
- **++** = love it
- **+** = like it
- **-** = not great
- **--** = hate it

Record their feedback:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli feedback RATING
```

### Step 4: Continue or learn

After 5 images:
- If the user wants more, create 5 new variations and repeat
- Once they have 20+ ratings, run learn:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli learn
```

### Step 5: Check calibration

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli profile
```

Training is complete when the profile shows "CALIBRATED" (20+ ratings with good positive/negative mix).
