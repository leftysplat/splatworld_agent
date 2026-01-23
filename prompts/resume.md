---
name: splatworld:resume
description: Resume an interrupted training session
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Resume Training

Continue an interrupted training session.

## Workflow

### Step 1: Check Session State

Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli resume --list-unrated --json
```

**Parse the JSON output:**
- Expect object with `session` and `unrated_images` fields
- `session` contains: `session_id`, `base_prompt`, `images_generated`, `status`
- `unrated_images` is array of unrated image objects
- If `session` is null: "No training session to resume. Start one with /splatworld:train"
- If output is not valid JSON: display error and stop

### Step 2: Show Session Info

Display session summary:
- "Session: {session_id}"
- "Base prompt: {base_prompt}"
- "Images generated: {images_generated}"
- "Status: {status}"

### Step 3: Check for Unrated Images

If `unrated_images` array is not empty:

Use AskUserQuestion:
- header: "Unrated Images Found"
- question: "Found {N} unrated images from this session. What would you like to do?"
- options:
  - "Rate them first - Review unrated images before continuing"
  - "Skip to new - Start generating new images"
  - "Cancel - Don't resume"

**If "Rate them first":**
For each unrated image:
- Display: "Image {image_number}: {variant_prompt}"
- Display: "File: {file_path}"
- Use AskUserQuestion with rating options (same as review.md Step 3)
- Call: `review --rate "{rating}" -g "{generation_id}"`

After rating all, proceed to reactivate.

**If "Skip to new":**
Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli resume --skip-unrated
```
Proceed to Step 4.

**If "Cancel":**
Exit with "Resume cancelled."

### Step 4: Reactivate Session

If session wasn't already reactivated by --skip-unrated:
Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli resume --skip-unrated
```

Display: "Session reactivated!"
Display: "Continue training with: /splatworld:train \"{base_prompt}\""

Use AskUserQuestion:
- header: "Continue Training"
- question: "Would you like to continue generating images now?"
- options:
  - "Yes - Continue training"
  - "No - I'll do it later"

**If "Yes":**
Execute the train workflow with the base_prompt.

**If "No":**
Exit with "Run /splatworld:train when ready."

## Notes

- Uses resume --list-unrated --json for session state
- Uses resume --skip-unrated to reactivate without interactive prompts
- Uses review --rate for rating unrated images
- Seamlessly transitions to train workflow if user wants to continue
