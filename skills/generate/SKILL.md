---
name: generate
description: Generate a single image + splat from prompt
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*), AskUserQuestion
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli generate "USER_PROMPT" [USER_FLAGS]
```

Pass the user's arguments directly. Examples:
- User says `/generate "modern kitchen"` -> run `... generate "modern kitchen"`
- User says `/generate "beach" --no-splat` -> run `... generate "beach" --no-splat"`

The default generator is Nano Banana Pro (use `--generator gemini` to use Gemini instead).

## Handle Provider Failures (IGEN-02)

If the generate command fails with a provider error:

1. Use AskUserQuestion:
   - header: "Provider Unavailable"
   - question: "Nano Banana Pro is unavailable. Try Gemini instead?"
   - options:
     - "yes" - Yes, use Gemini
     - "no" - No, cancel

2. If user chooses "yes":
   ```bash
   PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
   export PYTHONPATH="${PLUGIN_ROOT}" && python3 -m splatworld_agent.cli generate "PROMPT" --generator gemini
   ```

3. If user chooses "no":
   Report that generation was cancelled.

## FORBIDDEN ACTIONS

- Do NOT enhance or modify the user's prompt
- Do NOT summarize or interpret the output
- Do NOT ask your own confirmation questions
- Do NOT intercept the CLI interaction
- Do NOT describe what the CLI is doing
- Do NOT automatically switch providers without asking user (IGEN-02 requires consent)

## CORRECT BEHAVIOR

1. Parse user arguments
2. Run the single bash command above
3. If provider fails, ask user with AskUserQuestion before trying alternate
4. The CLI handles EVERYTHING else

The CLI handles prompt enhancement, generation, and output. You do not control this.
