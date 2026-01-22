# Image Rating Workflows

This guide explains how to view and rate generated images in splatworld-agent. There are two workflows: the default workflow (works everywhere) and the inline preview workflow (for supported terminals).

## Image Rating Workflows Overview

splatworld-agent provides two ways to rate images:

| Workflow | Description | Best For |
|----------|-------------|----------|
| **Default** | Images saved to disk; view externally, rate by number | All terminals, SSH, VS Code |
| **Inline Preview** | Images displayed directly in terminal | iTerm2, Kitty, WezTerm |

---

## Default Workflow (Recommended)

The default workflow works in **all terminals** including VS Code integrated terminal, standard macOS Terminal, and SSH sessions.

### How It Works

1. **Generate images:**
   ```bash
   splatworld-agent batch "modern kitchen with warm lighting"
   ```

2. **Output shows summary:**
   ```
   Generated 4 images (1-4). View in .splatworld/generations/ then rate here.
   ```

3. **View images externally:**
   ```bash
   # Open in Finder (macOS)
   open .splatworld/generations/

   # Or use your preferred image viewer
   # Preview, Photos, VS Code, etc.
   ```

4. **Rate by number:**
   ```bash
   # Rate a single image
   splatworld-agent rate 1 ++

   # Rate multiple images with same rating
   splatworld-agent rate 1 2 3 +

   # Batch rate with different ratings
   splatworld-agent brate 1 ++ 2 - 3 +
   ```

### Why This Is the Default

- **Universal compatibility**: Works in all terminals, including VS Code integrated terminal, standard macOS Terminal, and SSH sessions
- **No dependencies**: Doesn't require terminal image protocol support
- **Better image quality**: Native image viewers provide higher fidelity than terminal rendering
- **Multi-image comparison**: Easily view multiple images side-by-side in Finder or an image viewer
- **Non-intrusive**: Terminal stays clean for command output

### Supported Terminals

The default workflow works in:
- VS Code integrated terminal
- macOS Terminal.app
- iTerm2, Kitty, WezTerm
- SSH sessions
- Any other terminal emulator

---

## Inline Terminal Preview (Optional)

For users with supported terminals who prefer to see images directly in the terminal, add the `--inline` flag.

### Supported Terminals

| Terminal | Protocol | Platform | Setup Required |
|----------|----------|----------|----------------|
| **iTerm2** | iTerm2 Inline Images | macOS | None |
| **Kitty** | Kitty Graphics Protocol | macOS/Linux | None |
| **WezTerm** | iTerm2-compatible | macOS/Linux/Windows | None |

### How to Use

```bash
# Generate with inline preview
splatworld-agent batch "modern kitchen" --inline

# Interactive review with inline preview
splatworld-agent review --inline
```

### How It Works

- Uses the [term-image](https://github.com/AnonymouX47/term-image) library for terminal image display
- Auto-detects terminal capabilities via TERM and TERM_PROGRAM environment variables
- Renders images using terminal-specific protocols for best quality
- Falls back gracefully if terminal doesn't support images

### Limitations

The `--inline` flag is **not supported** in:
- VS Code integrated terminal (no image protocol support)
- Standard macOS Terminal.app (no image protocol support)
- Most SSH sessions (unless terminal forwarding is configured)
- tmux/screen (unless passthrough is configured)

When used in unsupported terminals, you'll see the file path instead of the image.

### Image Quality Considerations

- Image quality depends on terminal font size and cell dimensions
- Large images are automatically scaled to fit terminal width
- For best quality comparison, use the default workflow with a native image viewer

---

## Interactive Review Command

The `review` command lets you interactively rate unrated images one at a time.

### Basic Usage

```bash
# Default: shows file paths, rate by command
splatworld-agent review

# With inline preview (supported terminals only)
splatworld-agent review --inline
```

### Rating Options

During interactive review, use these ratings:

| Input | Meaning | Description |
|-------|---------|-------------|
| `++` | Love it | This is exactly what I want |
| `+` | Like it | Good, but could be better |
| `-` | Meh | Not great, has issues |
| `--` | Hate it | Definitely not what I want |
| `s` | Skip | Skip this image for now |
| `q` | Quit | Exit review mode |

### Example Session

```
$ splatworld-agent review

Reviewing 4 unrated images...

Image 1/4: .splatworld/generations/2026-01-21/kitchen-001/source.png
Rate (++/+/-/--/s/q): ++

Image 2/4: .splatworld/generations/2026-01-21/kitchen-002/source.png
Rate (++/+/-/--/s/q): -

Image 3/4: .splatworld/generations/2026-01-21/kitchen-003/source.png
Rate (++/+/-/--/s/q): q

Review paused. 2 images rated, 2 remaining.
```

---

## Quick Rating Commands

For fast rating without interactive mode, use the `rate` and `brate` commands.

### Single Image Rating

```bash
splatworld-agent rate <number> <rating>
```

Examples:
```bash
splatworld-agent rate 1 ++    # Love image 1
splatworld-agent rate 3 -     # Meh for image 3
splatworld-agent rate 2 --    # Hate image 2
```

### Multiple Images, Same Rating

```bash
splatworld-agent rate <numbers...> <rating>
```

Examples:
```bash
splatworld-agent rate 1 2 3 +      # Like images 1, 2, and 3
splatworld-agent rate 4 5 6 7 --   # Hate images 4, 5, 6, and 7
```

### Batch Rating (Different Ratings)

```bash
splatworld-agent brate <number> <rating> [<number> <rating>...]
```

Examples:
```bash
splatworld-agent brate 1 ++ 2 - 3 +     # Rate 3 images differently
splatworld-agent brate 1 ++ 2 ++ 3 - 4 --   # Rate 4 images
```

### Rating Scale Reference

| Rating | Meaning | When to Use |
|--------|---------|-------------|
| `++` | Love | Perfect or near-perfect. This is exactly what you want. |
| `+` | Like | Good result. On the right track but could improve. |
| `-` | Meh | Mediocre. Some things work, others don't. |
| `--` | Hate | Bad result. Not at all what you want. |

---

## After Rating: Update Your Taste Profile

After rating images, run the `learn` command to update your taste profile:

```bash
splatworld-agent learn
```

This analyzes your feedback history and updates your preference vectors. Future generations will incorporate what you've learned from your ratings.

**Tip:** Rate at least 10-20 images before running `learn` to give the system enough data to identify patterns in your preferences.

---

## Summary

| Task | Command |
|------|---------|
| Generate images | `splatworld-agent batch "prompt"` |
| Generate with inline preview | `splatworld-agent batch "prompt" --inline` |
| View images (default) | `open .splatworld/generations/` |
| Rate single image | `splatworld-agent rate 1 ++` |
| Rate multiple same | `splatworld-agent rate 1 2 3 +` |
| Rate multiple different | `splatworld-agent brate 1 ++ 2 - 3 +` |
| Interactive review | `splatworld-agent review` |
| Interactive with preview | `splatworld-agent review --inline` |
| Update taste profile | `splatworld-agent learn` |
