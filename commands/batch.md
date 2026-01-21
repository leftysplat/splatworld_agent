---
name: splatworld-agent:batch
description: Generate a batch of images for review
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Generate multiple creative variations of a prompt for efficient review and iteration.
</objective>

## Arguments

Required: base prompt/concept
Optional: -n COUNT (default 5)

Example: `/splatworld-agent:batch "futuristic city street" -n 5`

## Your task

### Step 1: Create prompt variations

Take the user's base concept and create N unique, detailed prompt variations (default 5). Each should:
- Keep the core concept but explore different interpretations
- Add specific details (lighting, atmosphere, colors, time of day, weather, materials, architectural style)
- Vary the mood and style
- Be 1-2 sentences, vivid and specific

Example for "futuristic city street":
1. "Neon-lit futuristic city street at night, rain-slicked pavement reflecting holographic advertisements, flying vehicles in the misty sky above"
2. "Clean minimalist futuristic city street at dawn, white geometric buildings, autonomous vehicles gliding silently, cherry blossom trees lining the walkway"
3. "Gritty cyberpunk street market, dense crowds under tangled power lines, street food vendors with glowing signs, steam rising from grates"
4. "Elevated futuristic city street on a mega-structure, glass walkways between towers, clouds below, golden sunset light streaming through"
5. "Post-rain futuristic city street, puddles reflecting a massive curved screen displaying news, people with transparent umbrellas, soft diffused lighting"

### Step 2: Generate images

For each prompt variation, run:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli generate "YOUR_DETAILED_PROMPT_HERE" --no-splat
```

### Step 3: Summarize

After generating all images, tell the user:
- Generated X images with variations on their concept
- Run `/splatworld-agent:review` to rate them
- Run `/splatworld-agent:convert` to turn favorites into 3D splats
