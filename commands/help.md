---
name: splatworld:help
description: Show available SplatWorld Agent commands and usage guide
---

<objective>
Display the complete SplatWorld Agent command reference.

Output ONLY the reference content below. Do NOT add commentary.
</objective>

<reference>
# SplatWorld Agent Command Reference

**SplatWorld Agent** is an iterative 3D Gaussian splat generation tool with taste learning. It learns your aesthetic preferences over time to generate images that match your style.

## Quick Start

1. `/splatworld:init` - Initialize project with .splatworld/ folder
2. `/splatworld:train "your prompt"` - Train your taste profile (20 images)
3. `/splatworld:batch "your prompt"` - Generate with learned preferences
4. `/splatworld:convert` - Convert favorites to 3D splats

## Training (Start Here)

**`/splatworld:train <prompt>`**
Guided training until profile is calibrated.

- Runs generate-review-learn cycles automatically
- Requires 20 rated images with good positive/negative mix
- Opens each image for quick rating (++/+/-/--)
- Learns your preferences between rounds

Usage: `/splatworld:train "cozy cabin interior"`

**`/splatworld:learn`**
Manually run learning on feedback.

- Analyzes all feedback with Claude
- Extracts preference patterns
- Updates taste profile
- Use after manual review sessions

Usage: `/splatworld:learn`

## Batch Workflow (After Training)

**`/splatworld:batch <prompt>`**
Generate multiple images for review.

- Default 5 images per cycle
- Can run multiple cycles with learning between
- Much faster iteration than single generation

Usage: `/splatworld:batch "futuristic city street" -n 5 -c 2`

**`/splatworld:review`**
Interactively rate generated images.

- Opens each image and prompts for rating
- Ratings: ++ (love), + (like), - (meh), -- (hate), s (skip), q (quit)
- Only loved images (++) will be converted to splats

Usage: `/splatworld:review`
Usage: `/splatworld:review --unrated`

**`/splatworld:convert`**
Convert loved images to 3D splats.

- Converts all ++ rated images by default
- Uses Marble API ($1.50 per conversion)
- Downloads .spz splat files and .glb meshes

Usage: `/splatworld:convert`
Usage: `/splatworld:convert --dry-run`

## Direct Mode

**`/splatworld:direct <prompt>`**
Generate complete 3D world from prompt.

- Executes full pipeline: enhance -> generate -> convert to 3D
- Returns viewer URL for immediate viewing
- Use `--provider gemini` to override default provider

Usage: `/splatworld:direct "cozy cabin in mountains"`

## Single Generation

**`/splatworld:generate <prompt>`**
Generate one image + optional splat.

- Enhances prompt with learned preferences
- Can skip splat conversion with --no-splat

Usage: `/splatworld:generate "modern kitchen"`

**`/splatworld:feedback <rating>`**
Rate the last generation.

- Ratings: ++ (love), + (like), - (meh), -- (hate)
- Or provide text feedback

Usage: `/splatworld:feedback ++`

## Profile Management

**`/splatworld:init`**
Initialize .splatworld/ in current project.

Creates:
- `.splatworld/profile.json` - Your taste profile
- `.splatworld/generations/` - Generated images
- `.splatworld/exemplars/` - Reference images you love
- `.splatworld/anti-exemplars/` - Reference images you hate

Usage: `/splatworld:init`

**`/splatworld:profile`**
View your taste profile.

Shows:
- Calibration status
- Learned preferences
- Rating statistics
- Prompt enhancement preview

Usage: `/splatworld:profile`

**`/splatworld:exemplar <image>`**
Add reference image you love.

- Copies image to .splatworld/exemplars/
- Influences future prompt enhancement

Usage: `/splatworld:exemplar reference.png -n "love the warm lighting"`

**`/splatworld:anti-exemplar <image>`**
Add reference image you hate.

- Copies image to .splatworld/anti-exemplars/
- Teaches what to avoid

Usage: `/splatworld:anti-exemplar bad_example.png -n "too flat"`

**`/splatworld:history`**
Browse past generations.

Usage: `/splatworld:history`
Usage: `/splatworld:history -n 20`

**`/splatworld:worlds`**
List all worlds from your Marble/WorldLabs account.

- Fetches directly from Marble API
- Shows all worlds you've created (not just local)
- Displays viewer URLs for each world

Usage: `/splatworld:worlds`

## Setup

**`/splatworld:config`**
View configuration status.

Shows API key status for:
- NANOBANANA_API_KEY / GOOGLE_API_KEY (image generation)
- ANTHROPIC_API_KEY (learning)
- WORLDLABS_API_KEY (3D conversion)

Usage: `/splatworld:config`

## Files & Structure

```
.splatworld/
├── profile.json          # Your taste profile
├── feedback.jsonl        # All feedback history
├── generations/          # Generated images by date
│   └── 2024-01-21/
│       └── gen-xxx/
│           ├── source.png
│           ├── scene.spz
│           └── metadata.json
├── exemplars/            # Reference images you love
└── anti-exemplars/       # Reference images you hate
```

## Environment Variables

```bash
# Image generation (required)
export NANOBANANA_API_KEY="..."  # or GOOGLE_API_KEY

# Learning (required for train/learn)
export ANTHROPIC_API_KEY="..."

# 3D conversion (required for convert)
export WORLDLABS_API_KEY="..."
```

## Common Workflows

**Initial training:**
```
/splatworld:init
/splatworld:train "your scene type"
# Rate 20+ images until calibrated
```

**Generate with trained profile:**
```
/splatworld:batch "your prompt" -n 5
/splatworld:review
/splatworld:convert
```

**Quick single generation:**
```
/splatworld:generate "your prompt"
/splatworld:feedback ++
```
</reference>
