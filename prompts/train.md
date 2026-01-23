---
name: splatworld:train
description: Guided training to calibrate your taste profile
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Adaptive Training

Train your taste profile by generating and rating images one at a time.

## Arguments

Parse from user input:
- `prompt` (required): Base prompt for image generation
- `count` (optional): Number of images to generate (default: 5)

Example: `/train "cozy cabin"` or `/train 10 "futuristic city"`

## Workflow

### Step 1: Initialize

Set variables:
- `base_prompt` = user's prompt
- `remaining` = count (or 5 if not specified)
- `images_generated` = 0

### Step 2: Generate One Image

Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli train "$base_prompt" --single --no-rate --json
```

**Parse the JSON output:**
- The output should be a single JSON object on one line
- Parse to get: `image_number`, `generation_id`, `file_path`, `variant_id`
- If output is not valid JSON (e.g., error message), display the error to user and stop

Display: "Generated Image {image_number}: {file_path}"

### Step 3: Ask for Rating

Use AskUserQuestion:
- header: "Rate Image {image_number}"
- question: "How do you rate this image? (View at: {file_path})"
- options:
  - "++ Love it!"
  - "+ Good"
  - "- Not great"
  - "-- Hate it"
  - "Skip - Don't rate"
  - "Done - Stop training"
  - "New prompt - Change base prompt"

### Step 4: Handle Response

**If rating (++, +, -, --):**
Extract the rating symbol and run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli review --rate "{rating}" -g "{generation_id}"
```
Confirm: "Rated Image {image_number}: {rating}"
Increment `images_generated`, decrement `remaining`

**If "Skip":**
Say "Skipped - you can rate this later with /splatworld:review"
Increment `images_generated`, decrement `remaining`

**If "Done":**
Show summary: "Training complete! Generated {images_generated} images."
Exit workflow.

**If "New prompt":**
Use AskUserQuestion:
- header: "Change Prompt"
- question: "Enter your new base prompt:"
- options: ["Other (type your prompt)"]

Update `base_prompt` with the new value.
Say "Base prompt changed to: {base_prompt}"

### Step 5: Loop or Exit

If `remaining > 0` and user didn't choose "Done":
- Go to Step 2
- If `images_generated % 5 == 0`, suggest: "You've generated 5 images. Consider selecting 'New prompt' if you want to explore a different direction."

If `remaining == 0`:
- Show summary: "Training complete! Generated {images_generated} images."
- Suggest: "Run /splatworld:learn to update your taste profile."

## Notes

- The CLI handles prompt enhancement and adaptation internally
- Ratings are saved immediately via review --rate
- Training state is persisted for resume functionality
- If user closes mid-training, they can resume with /splatworld:resume
