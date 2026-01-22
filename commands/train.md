---
name: splatworld-agent:train
description: Guided training mode to calibrate your taste profile (20 images)
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Run adaptive training that generates ONE image at a time, waits for your rating, then adapts the next image based on your feedback.
</objective>

## Arguments

- First argument: number of images to generate (optional)
- Remaining arguments: base prompt/concept (optional if resuming)

Examples:
- `/splatworld-agent:train 5 "beach on an alien world"` - Generate 5 images
- `/splatworld-agent:train "beach on an alien world"` - Train until stopped
- `/splatworld-agent:train 10` - Generate 10 images using saved prompt
- `/splatworld-agent:train` - Resume previous session

## Your task

**IMPORTANT: Just run the CLI command. The CLI handles everything interactively.**

Run the train command and let the user interact with it directly:

```bash
export PYTHONPATH=~/.claude/splatworld-agent
python3 -m splatworld_agent.cli train [ARGS]
```

The CLI will:
1. Generate ONE prompt variant at a time
2. Show the variant reasoning
3. Generate the image
4. Open it for viewing
5. Wait for user to rate it (++/+/-/--)
6. Use that rating to adapt the NEXT variant
7. Repeat until count reached or user cancels

**Do NOT generate multiple prompts upfront. Do NOT batch generate images.**

The adaptive loop is handled entirely by the CLI. Your job is just to invoke it with the user's arguments.
