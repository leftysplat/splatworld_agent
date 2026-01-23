---
name: direct
description: Generate complete 3D world from single prompt (enhance -> generate -> convert)
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*), AskUserQuestion
---

# Direct Command

Generate a complete 3D world from a single prompt. Executes the full pipeline:
1. Enhance prompt with taste profile preferences
2. Generate panoramic image with Nano/Gemini
3. Convert to 3D splat with World Labs Marble

## Step 1: Run direct command

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli direct "USER_PROMPT" --json
```

Parse the JSON output to determine result. The TUI will display real-time progress during generation.

### Optional flags

- `--provider nano|gemini` - Override provider (default respects training state)
- `--no-download` - Skip downloading splat file locally

## Step 2: Handle Provider Failures (IGEN-02)

If JSON output contains `"status": "provider_failure"`:

1. Use AskUserQuestion:
   - header: "Provider Unavailable"
   - question: "Nano Banana Pro is unavailable. Retry with Gemini?"
   - options:
     - "yes" - Yes, retry with Gemini
     - "no" - No, cancel

2. If user chooses "yes":
   ```bash
   PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
   export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli direct "USER_PROMPT" --provider gemini --json
   ```

3. If user chooses "no":
   Report that direct generation was cancelled.

## Step 3: Handle Partial Success

If JSON output contains `"status": "partial_success"`:

The image was generated but 3D conversion failed. Tell user:
- Image was saved (show path)
- 3D conversion failed (show reason)
- They can retry conversion later with `/splatworld:convert IMAGE_NUMBER`

## Step 4: Report Success

On `"status": "success"`, tell user:

1. **Image created:** Show image number and path
2. **Enhanced prompt:** Show what modifications were made
3. **3D World:** Show viewer URL prominently (this is the main output!)
4. **Splat file:** Show local path if downloaded

Example response:
```
Created Image 42 from your prompt!

**View your 3D world:** https://marble.worldlabs.ai/world/abc123

I enhanced your prompt by adding warm lighting and atmospheric fog based on your taste profile.

Files saved:
- Image: generated_images/42.png
- Splat: splats/42.spz
```

## FORBIDDEN ACTIONS

- Do NOT modify the user's prompt before passing to CLI
- Do NOT automatically switch providers without asking (IGEN-02 requires consent)
- Do NOT skip showing the viewer URL
- Do NOT suppress the enhanced prompt reasoning
- Do NOT intercept or modify the CLI's progress output

## CORRECT BEHAVIOR

1. Parse user's prompt exactly as given
2. Run the direct CLI command with --json (TUI shows progress)
3. If provider fails, ask user with AskUserQuestion before trying alternate
4. Report results clearly with viewer URL prominent

## TASK COMPLETION

**IMPORTANT:** After reporting success (showing the viewer URL and file paths), your task is COMPLETE.

Do NOT continue processing, waiting, or asking follow-up questions. The generation is done when you've displayed:
- The viewer URL
- The enhanced prompt explanation
- The saved file paths

Stop immediately after showing these results. The user can run another command if they want more generations.
