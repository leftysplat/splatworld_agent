---
name: splatworld:download-splats
description: Download 3D splat files for converted images
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Download Splats

Download 3D Gaussian splat files (.spz) for images that have been converted to 3D.

## Arguments

Parse from user input (optional):
- `image_numbers` (optional): Specific image numbers to download (e.g., "1 3 5")

Example: `/download-splats` or `/download-splats 1 2 3`

## Workflow

### Step 1: Check for Missing Splats

Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli download-splats --list --json
```

**Parse the JSON output:**
- Expect a JSON array of objects
- Each object has: `generation_id`, `image_number`, `prompt`, `viewer_url`
- If output is empty array `[]`: "All splats are downloaded! Nothing to download."
- If output is not valid JSON: display error and stop

### Step 2: Show Missing Splats

Display summary: "Found {N} missing splat file(s):"

For each missing splat:
- "Image {image_number}: {prompt} (truncated to 50 chars)"

### Step 3: Confirm Download

Use AskUserQuestion:
- header: "Download Splats"
- question: "Download {N} splat file(s)? (Requires World Labs premium account)"
- options:
  - "Yes - Download all"
  - "No - Cancel"

**If "No":**
Exit with "Download cancelled."

### Step 4: Download

Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli download-splats --all
```

The CLI will show progress for each download.

### Step 5: Summary

After download completes, show:
- "Downloaded {N} splat files to downloaded_splats/"
- "View splats: Open .spz files in a compatible viewer"

If any downloads failed (check CLI output for "Premium account required"):
- "Note: Some downloads require a World Labs premium account."

## Specific Images Mode

If user provides image numbers (e.g., `/download-splats 1 3 5`):

Skip Step 3 (confirmation) and run directly:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli download-splats 1 3 5
```

Specific image downloads don't require confirmation.

## Notes

- Uses download-splats --list --json for pre-flight check
- Uses download-splats --all for batch download with confirmation
- Specific image numbers bypass confirmation (existing CLI behavior)
- Downloads may fail if user doesn't have World Labs premium account
