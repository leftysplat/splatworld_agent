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

### Option 1: Plugin System (Recommended)

Install as a Claude Code plugin:

```
/plugin install splatworld --from https://github.com/leftysplat/splatworld
```

This installs the plugin and makes all `/splatworld:*` commands available immediately.

### Option 2: Manual Install

Clone to the Claude plugins directory:

```bash
git clone https://github.com/leftysplat/splatworld.git ~/.claude/splatworld
```

Restart Claude Code. Commands will be available as `/splatworld:*`.

### Updating

Once installed, update from within Claude Code:
```
/splatworld:update
```

This pulls the latest changes from the repository.

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
/splatworld:init
```

This creates `.splatworld/` in your project with an empty taste profile.

### 2. Generate Your First Splat

```
/splatworld:generate modern kitchen with marble countertops
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
/splatworld:feedback love the lighting, but too cluttered
```

Or use quick reactions:
```
/splatworld:feedback ++      # Love it
/splatworld:feedback --      # Hate it
/splatworld:feedback +       # Good
/splatworld:feedback -       # Not great
```

### 4. Add Exemplars

Have an image that captures exactly what you want?

```
/splatworld:exemplar ./reference-images/perfect-kitchen.jpg
```

This adds it to your taste profile. The agent will reference it when generating.

### 5. View Your Taste Profile

```
/splatworld:profile
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
/splatworld:learn
```

This analyzes your feedback history and updates your preference vectors.

## Commands Reference

### Setup & Configuration

#### `init`
Initialize `.splatworld/` in a project directory.
```bash
splatworld init [--path PATH]
```
| Option | Description |
|--------|-------------|
| `--path PATH` | Project path to initialize (default: current directory) |

#### `setup-keys`
Configure API keys for SplatWorld Agent.
```bash
splatworld setup-keys [OPTIONS]
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
splatworld check-keys
```

#### `config`
View or edit configuration.
```bash
splatworld config
```

#### `install-prompts`
Install Claude Code slash command prompts.
```bash
splatworld install-prompts
```

#### `update`
Update SplatWorld Agent to the latest version (pulls from git).
```bash
splatworld update
```

---

### Image Generation

#### `generate`
Generate a single image and optionally convert to 3D splat.
```bash
splatworld generate [OPTIONS] PROMPT...
```
| Option | Description |
|--------|-------------|
| `--seed INTEGER` | Random seed for reproducibility |
| `--no-enhance` | Don't enhance prompt with taste profile |
| `--no-splat` | Skip 3D splat generation |
| `--generator [nano\|gemini]` | Image generator to use |

**Example:**
```bash
splatworld generate "cozy cabin with fireplace" --seed 42
```

#### `batch`
Generate multiple images for review in cycles.
```bash
splatworld batch [OPTIONS] PROMPT...
```
| Option | Description |
|--------|-------------|
| `-n, --count INTEGER` | Number of images per cycle (default: 5) |
| `-c, --cycles INTEGER` | Number of generate-review-learn cycles |
| `--generator [nano\|gemini]` | Image generator to use |
| `--inline` | Show inline image previews (iTerm2/Kitty/WezTerm) |

**Example:**
```bash
splatworld batch "modern kitchen" -n 5 -c 2
# Generates 5 images, review them, learns, generates 5 more
```

#### `train`
Guided training mode to calibrate your taste profile.
```bash
splatworld train [OPTIONS] PROMPT...
```
| Option | Description |
|--------|-------------|
| `-n, --images-per-round INTEGER` | Images to generate per round |
| `--generator [nano\|gemini]` | Image generator to use |

Runs generate-review-learn cycles until calibrated (minimum 20 rated images).

**Example:**
```bash
splatworld train "industrial warehouse"
```

---

### Rating & Feedback

#### `rate`
Rate images by number from the current batch.
```bash
splatworld rate IMAGE_NUMS... {++|+|-|--}
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
splatworld rate 1 ++       # Rate image 1 as love
splatworld rate 3 -        # Rate image 3 as not great
splatworld rate 2 5 +      # Rate images 2 and 5 as good
```

#### `brate`
Rate multiple images with different ratings in one command.
```bash
splatworld brate RATINGS_INPUT...
```

**Examples:**
```bash
splatworld brate 1 ++ 2 - 3 -- 4 +
splatworld brate 1++ 2- 3-- 4+    # Spaces optional
```

#### `review`
Interactively review and rate generated images.
```bash
splatworld review [OPTIONS]
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
splatworld feedback [OPTIONS] [FEEDBACK_TEXT]...
```
| Option | Description |
|--------|-------------|
| `-g, --generation TEXT` | Generation ID (defaults to last) |

**Example:**
```bash
splatworld feedback "love the lighting, but too cluttered"
```

---

### 3D Splat Conversion

#### `convert`
Convert loved images to 3D splats via World Labs Marble API.
```bash
splatworld convert [OPTIONS]
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
splatworld splats [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--open TEXT` | Open viewer URL for a specific generation ID |

**Examples:**
```bash
splatworld splats                    # List all splats with URLs
splatworld splats --open abc123      # Open specific splat in browser
```

#### `download-splats`
Download splat files that haven't been downloaded yet.
```bash
splatworld download-splats [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--all` | Download all missing splats without confirmation |

Finds all converted generations (with viewer_url) that don't have local splat files
and downloads them from WorldLabs. Useful when `download_splats` is disabled in config
or for retrying failed downloads.

**Note:** Downloading splats may require a premium WorldLabs account.

**Examples:**
```bash
splatworld download-splats           # List missing splats, prompt to download
splatworld download-splats --all     # Download all without confirmation
```

---

### Taste Profile

#### `profile`
View or edit your taste profile.
```bash
splatworld profile [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--edit` | Open profile for editing |
| `--json` | Output as JSON |

#### `learn`
Synthesize feedback into updated preferences.
```bash
splatworld learn [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be learned without saving |

#### `exemplar`
Add an exemplar image (things you love) to your taste profile.
```bash
splatworld exemplar [OPTIONS] IMAGE_PATH
```
| Option | Description |
|--------|-------------|
| `-n, --notes TEXT` | Notes about why you like this |

**Example:**
```bash
splatworld exemplar ./reference/perfect-kitchen.jpg -n "Love the warm lighting"
```

#### `anti-exemplar`
Add an anti-exemplar image (things you never want).
```bash
splatworld anti-exemplar [OPTIONS] IMAGE_PATH
```
| Option | Description |
|--------|-------------|
| `-n, --notes TEXT` | Notes about why you dislike this |

---

### History & Session

#### `history`
Browse past generations.
```bash
splatworld history [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `-n, --limit INTEGER` | Number of generations to show |

#### `exit`
Save session and exit SplatWorld Agent.
```bash
splatworld exit [OPTIONS]
```
| Option | Description |
|--------|-------------|
| `-s, --summary TEXT` | Summary of what was accomplished |
| `-n, --notes TEXT` | Notes for next session |

Records session activity (generations, feedback, conversions, learns) for later resumption.

#### `resume-work`
Resume work from previous session.
```bash
splatworld resume-work
```

Shows recent session history, current status, and starts a new session.

---

### Slash Commands (Claude Code)

All CLI commands are available as slash commands in Claude Code:

| Slash Command | CLI Equivalent |
|---------------|----------------|
| `/splatworld:init` | `splatworld init` |
| `/splatworld:generate <prompt>` | `splatworld generate <prompt>` |
| `/splatworld:batch <prompt>` | `splatworld batch <prompt>` |
| `/splatworld:train <prompt>` | `splatworld train <prompt>` |
| `/splatworld:review` | `splatworld review` |
| `/splatworld:feedback <text>` | `splatworld feedback <text>` |
| `/splatworld:learn` | `splatworld learn` |
| `/splatworld:convert` | `splatworld convert` |
| `/splatworld:splats` | `splatworld splats` |
| `/splatworld:download-splats` | `splatworld download-splats` |
| `/splatworld:display-links` | `splatworld splats` |
| `/splatworld:profile` | `splatworld profile` |
| `/splatworld:exemplar <path>` | `splatworld exemplar <path>` |
| `/splatworld:history` | `splatworld history` |
| `/splatworld:exit` | `splatworld exit` |
| `/splatworld:resume-work` | `splatworld resume-work` |
| `/splatworld:help` | `splatworld help` |
| `/splatworld:update` | `splatworld update` |

## Project Structure

After initialization, your project will have:

```
your-project/
├── generated_images/          # Visible: Your generated images
│   └── 2026-01-21/
│       ├── gen-20260121-123456-abc123/
│       │   └── source.png
│       └── gen-20260121-123457-def456/
│           └── source.png
├── downloaded_splats/         # Visible: Your 3D splat files
│   ├── gen-20260121-123456-abc123.spz
│   └── gen-20260121-123457-def456.spz
└── .splatworld/               # Hidden: Config and metadata
    ├── profile.json           # Your taste profile
    ├── feedback.jsonl         # Feedback history
    ├── sessions.jsonl         # Session history
    ├── exemplars/             # Reference images you love
    ├── anti-exemplars/        # Reference images you hate
    └── generations/           # Generation metadata
        └── 2026-01-21/
            └── gen-20260121-123456-abc123/
                └── metadata.json
```

**Note:** Images and splats are stored in visible directories (`generated_images/`, `downloaded_splats/`)
so you can easily browse them in Finder or your file manager. Metadata is stored in the hidden
`.splatworld/` directory.

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

2. **Pattern extraction**: The `/splatworld:learn` command (or automatic periodic learning) analyzes feedback to find patterns:
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
splatworld init

# Generate
splatworld generate "modern kitchen" --seed 42

# Feedback
splatworld feedback "love it" --generation kitchen-001

# View profile
splatworld profile show

# Learn from feedback
splatworld learn
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

Run `/splatworld:init` first to create `.splatworld/` in your project.

### "API key not configured"

Set up your keys in `~/.splatworld_agent/config.yaml` or as environment variables.

### "Generation failed"

Check the error message. Common issues:
- Marble API rate limit (wait and retry)
- Invalid image format (ensure PNG/JPEG)
- Network timeout (retry)

### "Taste profile not updating"

Feedback is collected but preferences only update when you run `/splatworld:learn` or after enough feedback accumulates (default: 10 new entries).

## License

MIT

## Related Projects

- [splatworld_framework](https://github.com/leftysplat/splatworld_framework) - Batch splat generation
- [splatworld_site](https://github.com/leftysplat/splatworld_site) - Autonomous research agent
- [splatworld_explore](https://github.com/leftysplat/splatworld_explore) - 3D splat viewer
- [splatworld_labeler](https://github.com/leftysplat/splatworld_labeler) - Scene annotation schemas
