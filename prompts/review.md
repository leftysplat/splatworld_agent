---
name: splatworld:review
description: Review and rate unrated images
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*), AskUserQuestion
---

# Review Unrated Images

Review images that haven't been rated yet.

## Workflow

### Step 1: Get Unrated Images

Run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli review --list --json
```

**Parse the JSON output:**
- Expect a JSON array of objects
- Each object has: `generation_id`, `image_number`, `file_path`, `prompt`, `created_at`
- If output is empty array `[]`: "All images are rated! Nothing to review."
- If output is not valid JSON: display error and stop

### Step 2: Show Summary

Count: "{N} unrated images found."

Use AskUserQuestion:
- header: "Review Images"
- question: "How would you like to proceed?"
- options:
  - "Review all - Rate images one by one"
  - "Skip - Do this later"

If "Skip": Exit with "You can review anytime with /splatworld:review"

### Step 3: Rating Loop

For each unrated image from JSON:

Display: "Image {image_number}: {prompt}"
Display: "File: {file_path}"

Use AskUserQuestion:
- header: "Rate Image {image_number}"
- question: "How do you rate this image?"
- options:
  - "++ Love it!"
  - "+ Good"
  - "- Not great"
  - "-- Hate it"
  - "Skip - Don't rate this one"
  - "Done - Stop reviewing"

**If rating (++, +, -, --):**
Extract rating symbol and run:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli review --rate "{rating}" -g "{generation_id}"
```
Confirm: "Rated Image {image_number}: {rating}"

**If "Skip":**
Continue to next image.

**If "Done":**
Exit loop.

### Step 4: Summary

Show: "Review complete! Rated {count} images."

If any ++ ratings: "Run /splatworld:convert to turn your favorites into 3D splats!"
If 3+ ratings: "Run /splatworld:learn to update your taste profile."

## Notes

- Uses review --list --json for machine-readable output
- Uses review --rate -g for non-interactive rating
- Ratings saved immediately
- Can stop anytime and continue later
