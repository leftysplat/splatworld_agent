# SplatWorld Agent

A Claude Code plugin for iterative 3D Gaussian splat generation with taste learning. The agent learns your aesthetic preferences over time and applies them to future generations.

## Overview

**splatworld_agent** is different from [splatworld_framework](https://github.com/leftysplat/splatworld_framework):

| | splatworld_framework | splatworld_agent |
|---|---|---|
| Purpose | Batch production | Creative exploration |
| Interaction | "Generate 50 warehouse scenes" | "Help me find my style" |
| Learning | None | Learns your preferences |
| Output | Volume, consistency | Quality, personal taste |

## Installation

### Prerequisites

- Python 3.10+
- Claude Code CLI
- git
- API keys for:
  - World Labs Marble (3D conversion)
  - Nano Banana Pro or Gemini (image generation)
  - Anthropic (prompt enhancement)

### Quick Install

```bash
# One-line install (installs to ~/.claude/splatworld-agent)
curl -fsSL https://raw.githubusercontent.com/leftysplat/splatworld_agent/main/install.sh | bash
```

Or clone and run:
```bash
git clone https://github.com/leftysplat/splatworld_agent.git && (cd splatworld_agent && ./install.sh)
```

The installer will:
1. Install to `~/.claude/splatworld-agent/` (like GSD)
2. Set up commands at `~/.claude/commands/splatworld-agent/`
3. Available from **any project directory**

### Updating

Once installed, update from within Claude Code:
```
/splatworld-agent:update
```

This pulls the latest changes from the repository. Since commands are symlinked, updates are immediate.

### Configure API Keys

Create `~/.splatworld_agent/config.yaml`:

```yaml
api_keys:
  marble: "wlt-your-key-here"
  nano: "your-nano-key-here"      # Default image generator
  google: "your-google-key-here"  # Optional: Gemini alternative
  anthropic: "your-anthropic-key" # For taste-enhanced prompts

defaults:
  image_generator: nano  # or "gemini"
```

Or set environment variables:
```bash
export WORLDLABS_API_KEY="wlt-..."
export NANO_API_KEY="..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Quick Start

### 1. Initialize a Project

```bash
cd your-project
```

Then in Claude Code:
```
/splatworld-agent:init
```

This creates `.splatworld/` in your project with an empty taste profile.

### 2. Generate Your First Splat

```
/splatworld-agent:generate modern kitchen with marble countertops
```

The agent will:
1. Enhance your prompt based on your taste profile (empty at first)
2. Generate an image via Nano Banana Pro
3. Convert to 3D splat via Marble API
4. Save to `.splatworld/generations/YYYY-MM-DD/`
5. Show you the result and ask for feedback

### 3. Provide Feedback

After viewing a generation:
```
/splatworld-agent:feedback love the lighting, but too cluttered
```

Or use quick reactions:
```
/splatworld-agent:feedback ++      # Love it
/splatworld-agent:feedback --      # Hate it
/splatworld-agent:feedback +       # Good
/splatworld-agent:feedback -       # Not great
```

### 4. Add Exemplars

Have an image that captures exactly what you want?

```
/splatworld-agent:exemplar ./reference-images/perfect-kitchen.jpg
```

This adds it to your taste profile. The agent will reference it when generating.

### 5. View Your Taste Profile

```
/splatworld-agent:profile
```

Shows your learned preferences:
- Visual style preferences (lighting, color, mood)
- Composition preferences (density, framing, perspective)
- Domain preferences (environments you generate most)
- Quality criteria (what you consistently like/dislike)
- Exemplar gallery

### 6. Synthesize Learnings

Periodically (or manually):
```
/splatworld-agent:learn
```

This analyzes your feedback history and updates your preference vectors.

## Commands Reference

### Setup & Configuration

#### `init`
Initialize `.splatworld/` in a project directory.
```bash
splatworld-agent init [--path PATH]
```
| Option | Description |
|--------|-------------|
| `--path PATH` | Project path to initialize (default: current directory) |

#### `setup-keys`
Configure API keys for SplatWorld Agent.
```bash
splatworld-agent setup-keys [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--anthropic TEXT` | Anthropic API key |
| `--google TEXT` | Google API key |
| `--worldlabs TEXT` | World Labs (Marble) API key |
| `--nano TEXT` | Nano Banana API key |

#### `check-keys`
Check API key configuration status.
```bash
splatworld-agent check-keys
```

#### `config`
View or edit configuration.
```bash
splatworld-agent config
```

#### `install-prompts`
Install Claude Code slash command prompts.
```bash
splatworld-agent install-prompts
```

#### `update`
Update SplatWorld Agent to the latest version (pulls from git).
```bash
splatworld-agent update
```

---

### Image Generation

#### `generate`
Generate a single image and optionally convert to 3D splat.
```bash
splatworld-agent generate [OPTIONS] PROMPT...
```
| Option | Description |
|--------|-------------|
| `--seed INTEGER` | Random seed for reproducibility |
| `--no-enhance` | Don't enhance prompt with taste profile |
| `--no-splat` | Skip 3D splat generation |
| `--generator [nano\|gemini]` | Image generator to use |

**Example:**
```bash
splatworld-agent generate "cozy cabin with fireplace" --seed 42
```

#### `batch`
Generate multiple images for review in cycles.
```bash
splatworld-agent batch [OPTIONS] PROMPT...
```
| Option | Description |
|--------|-------------|
| `-n, --count INTEGER` | Number of images per cycle (default: 5) |
| `-c, --cycles INTEGER` | Number of generate-review-learn cycles |
| `--generator [nano\|gemini]` | Image generator to use |
| `--inline` | Show inline image previews (iTerm2/Kitty/WezTerm) |

**Example:**
```bash
splatworld-agent batch "modern kitchen" -n 5 -c 2
# Generates 5 images, review them, learns, generates 5 more
```

#### `train`
Guided training mode to calibrate your taste profile.
```bash
splatworld-agent train [OPTIONS] PROMPT...
```
| Option | Description |
|--------|-------------|
| `-n, --images-per-round INTEGER` | Images to generate per round |
| `--generator [nano\|gemini]` | Image generator to use |

Runs generate-review-learn cycles until calibrated (minimum 20 rated images).

**Example:**
```bash
splatworld-agent train "industrial warehouse"
```

---

### Rating & Feedback

#### `rate`
Rate images by number from the current batch.
```bash
splatworld-agent rate IMAGE_NUMS... {++|+|-|--}
```

**Rating scale:**
| Rating | Meaning |
|--------|---------|
| `++` | Love it (will be converted to splat) |
| `+` | Like it |
| `-` | Not great |
| `--` | Hate it |

**Examples:**
```bash
splatworld-agent rate 1 ++       # Rate image 1 as love
splatworld-agent rate 3 -        # Rate image 3 as not great
splatworld-agent rate 2 5 +      # Rate images 2 and 5 as good
```

#### `brate`
Rate multiple images with different ratings in one command.
```bash
splatworld-agent brate RATINGS_INPUT...
```

**Examples:**
```bash
splatworld-agent brate 1 ++ 2 - 3 -- 4 +
splatworld-agent brate 1++ 2- 3-- 4+    # Spaces optional
```

#### `review`
Interactively review and rate generated images.
```bash
splatworld-agent review [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `-b, --batch TEXT` | Review specific batch ID |
| `--all` | Review ALL unrated images across all batches |
| `-c, --current` | Review current batch (default) |
| `-n, --limit INTEGER` | Number of images to review |
| `--unrated` | Only show unrated images |
| `--inline` | Show inline image previews (iTerm2/Kitty/WezTerm) |

**Interactive ratings:**
- `++` = love it
- `+` = like it
- `-` = not great
- `--` = hate it
- `s` = skip
- `q` = quit review

#### `feedback`
Provide text feedback on a generation.
```bash
splatworld-agent feedback [OPTIONS] [FEEDBACK_TEXT]...
```
| Option | Description |
|--------|-------------|
| `-g, --generation TEXT` | Generation ID (defaults to last) |

**Example:**
```bash
splatworld-agent feedback "love the lighting, but too cluttered"
```

---

### 3D Splat Conversion

#### `convert`
Convert loved images to 3D splats via World Labs Marble API.
```bash
splatworld-agent convert [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--all-positive` | Convert all positively rated (+ and ++) |
| `-g, --generation TEXT` | Specific generation IDs to convert |
| `--dry-run` | Show what would be converted without doing it |

By default, converts all images rated `++` that don't already have splats.

#### `splats`
List all converted splats with their World Labs viewer URLs.
```bash
splatworld-agent splats [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--open TEXT` | Open viewer URL for a specific generation ID |

**Examples:**
```bash
splatworld-agent splats                    # List all splats with URLs
splatworld-agent splats --open abc123      # Open specific splat in browser
```

---

### Taste Profile

#### `profile`
View or edit your taste profile.
```bash
splatworld-agent profile [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--edit` | Open profile for editing |
| `--json` | Output as JSON |

#### `learn`
Synthesize feedback into updated preferences.
```bash
splatworld-agent learn [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be learned without saving |

#### `exemplar`
Add an exemplar image (things you love) to your taste profile.
```bash
splatworld-agent exemplar [OPTIONS] IMAGE_PATH
```
| Option | Description |
|--------|-------------|
| `-n, --notes TEXT` | Notes about why you like this |

**Example:**
```bash
splatworld-agent exemplar ./reference/perfect-kitchen.jpg -n "Love the warm lighting"
```

#### `anti-exemplar`
Add an anti-exemplar image (things you never want).
```bash
splatworld-agent anti-exemplar [OPTIONS] IMAGE_PATH
```
| Option | Description |
|--------|-------------|
| `-n, --notes TEXT` | Notes about why you dislike this |

---

### History & Session

#### `history`
Browse past generations.
```bash
splatworld-agent history [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `-n, --limit INTEGER` | Number of generations to show |

#### `exit`
Save session and exit SplatWorld Agent.
```bash
splatworld-agent exit [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `-s, --summary TEXT` | Summary of what was accomplished |
| `-n, --notes TEXT` | Notes for next session |

Records session activity (generations, feedback, conversions, learns) for later resumption.

#### `resume-work`
Resume work from previous session.
```bash
splatworld-agent resume-work
```

Shows recent session history, current status, and starts a new session.

---

### Slash Commands (Claude Code)

All CLI commands are available as slash commands in Claude Code:

| Slash Command | CLI Equivalent |
|---------------|----------------|
| `/splatworld-agent:init` | `splatworld-agent init` |
| `/splatworld-agent:generate <prompt>` | `splatworld-agent generate <prompt>` |
| `/splatworld-agent:batch <prompt>` | `splatworld-agent batch <prompt>` |
| `/splatworld-agent:train <prompt>` | `splatworld-agent train <prompt>` |
| `/splatworld-agent:review` | `splatworld-agent review` |
| `/splatworld-agent:feedback <text>` | `splatworld-agent feedback <text>` |
| `/splatworld-agent:learn` | `splatworld-agent learn` |
| `/splatworld-agent:convert` | `splatworld-agent convert` |
| `/splatworld-agent:splats` | `splatworld-agent splats` |
| `/splatworld-agent:display-links` | `splatworld-agent splats` |
| `/splatworld-agent:profile` | `splatworld-agent profile` |
| `/splatworld-agent:exemplar <path>` | `splatworld-agent exemplar <path>` |
| `/splatworld-agent:history` | `splatworld-agent history` |
| `/splatworld-agent:exit` | `splatworld-agent exit` |
| `/splatworld-agent:resume-work` | `splatworld-agent resume-work` |
| `/splatworld-agent:help` | `splatworld-agent help` |
| `/splatworld-agent:update` | `splatworld-agent update` |

## Project Structure

After initialization, your project will have:

```
your-project/
├── .splatworld/
│   ├── profile.json           # Your taste profile
│   ├── feedback.jsonl         # Feedback history
│   ├── exemplars/             # Reference images you love
│   ├── anti-exemplars/        # Reference images you hate
│   └── generations/           # Generated content
│       └── 2026-01-21/
│           ├── kitchen-001/
│           │   ├── source.png
│           │   ├── splat.spz
│           │   ├── mesh.obj
│           │   └── metadata.json
│           └── kitchen-002/
│               └── ...
```

## Taste Profile Structure

`profile.json` contains your learned preferences:

```json
{
  "version": "1.0",
  "created": "2026-01-21T12:00:00Z",
  "updated": "2026-01-21T18:30:00Z",

  "visual_style": {
    "lighting": {
      "preference": "moody, low-key",
      "avoid": "flat, overexposed",
      "confidence": 0.8
    },
    "color_palette": {
      "preference": "warm earth tones, desaturated",
      "avoid": "neon, oversaturated",
      "confidence": 0.7
    },
    "mood": {
      "preference": "atmospheric, cinematic",
      "avoid": "sterile, clinical",
      "confidence": 0.6
    }
  },

  "composition": {
    "density": {
      "preference": "moderate detail, not cluttered",
      "confidence": 0.75
    },
    "perspective": {
      "preference": "slightly elevated, wide angle",
      "confidence": 0.5
    },
    "foreground": {
      "preference": "include foreground elements for depth",
      "confidence": 0.6
    }
  },

  "domain": {
    "environments": ["industrial", "kitchen", "workshop"],
    "avoid_environments": ["fantasy", "sci-fi"],
    "confidence": 0.8
  },

  "quality": {
    "must_have": [
      "realistic lighting",
      "clear focal point",
      "navigable floor space"
    ],
    "never": [
      "floating objects",
      "obvious AI artifacts",
      "empty void backgrounds"
    ]
  },

  "exemplars": [
    {
      "path": "exemplars/perfect-kitchen.jpg",
      "added": "2026-01-21T14:00:00Z",
      "notes": "Love the warm lighting and lived-in feel"
    }
  ],

  "anti_exemplars": [
    {
      "path": "anti-exemplars/too-clean.jpg",
      "added": "2026-01-21T15:00:00Z",
      "notes": "Too sterile, no character"
    }
  ],

  "stats": {
    "total_generations": 47,
    "feedback_count": 32,
    "love_count": 12,
    "hate_count": 5
  }
}
```

## How Taste Learning Works

1. **Feedback collection**: Every rating and critique is logged to `feedback.jsonl`

2. **Pattern extraction**: The `/splatworld-agent:learn` command (or automatic periodic learning) analyzes feedback to find patterns:
   - "User consistently dislikes flat lighting" → update `visual_style.lighting.avoid`
   - "User always rates industrial environments highly" → update `domain.environments`

3. **Confidence scoring**: Preferences have confidence scores (0-1) based on consistency and sample size

4. **Prompt injection**: When you generate, the agent reads your profile and enhances your prompt:
   ```
   Your prompt: "modern kitchen"
   Enhanced:    "modern kitchen with warm, moody lighting, earth tone
                color palette, moderate detail density, include foreground
                elements for depth, realistic lighting, clear focal point"
   ```

5. **Exemplar reference**: If you have exemplars, the agent may describe them to inform generation

## Image Generators

### Nano Banana Pro (Default)

Best quality results. Requires Nano API key.

```yaml
# In config.yaml
defaults:
  image_generator: nano
```

### Google Gemini

Alternative option. Free tier available.

```yaml
defaults:
  image_generator: gemini
```

### Custom Generator

Implement the `ImageGenerator` interface to add your own:

```python
from splatworld_agent.generators import ImageGenerator

class MyGenerator(ImageGenerator):
    def generate(self, prompt: str, seed: int = None) -> bytes:
        # Return image bytes
        pass
```

## CLI Usage

The slash commands call the CLI under the hood. You can also use it directly:

```bash
# Initialize
splatworld-agent init

# Generate
splatworld-agent generate "modern kitchen" --seed 42

# Feedback
splatworld-agent feedback "love it" --generation kitchen-001

# View profile
splatworld-agent profile show

# Learn from feedback
splatworld-agent learn
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black splatworld_agent/
ruff check splatworld_agent/
```

## Image Rating Workflows

For detailed documentation on viewing and rating images, see [docs/USAGE.md](splatworld_agent/docs/USAGE.md).

**Quick summary:**
- **Default mode:** Images saved to `.splatworld/generations/`. Open in Finder/viewer, rate by number.
- **Inline mode:** Add `--inline` flag for terminal image previews (iTerm2/Kitty/WezTerm only).

## Troubleshooting

### "Profile not found"

Run `/splatworld-agent:init` first to create `.splatworld/` in your project.

### "API key not configured"

Set up your keys in `~/.splatworld_agent/config.yaml` or as environment variables.

### "Generation failed"

Check the error message. Common issues:
- Marble API rate limit (wait and retry)
- Invalid image format (ensure PNG/JPEG)
- Network timeout (retry)

### "Taste profile not updating"

Feedback is collected but preferences only update when you run `/splatworld-agent:learn` or after enough feedback accumulates (default: 10 new entries).

## License

MIT

## Related Projects

- [splatworld_framework](https://github.com/leftysplat/splatworld_framework) - Batch splat generation
- [splatworld_site](https://github.com/leftysplat/splatworld_site) - Autonomous research agent
- [splatworld_explore](https://github.com/leftysplat/splatworld_explore) - 3D splat viewer
- [splatworld_labeler](https://github.com/leftysplat/splatworld_labeler) - Scene annotation schemas
