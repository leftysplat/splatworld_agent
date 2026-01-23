---
name: splatworld:migrate-data
description: Migrate data from old .splatworld_agent/ folder
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

# CRITICAL: Just run the CLI

**Your ONLY job is to run this ONE bash command:**

```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli migrate-data
```

## Options

If user specifies a source directory:
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli migrate-data --from-dir "/path/to/old/project"
```

For dry-run (preview only):
```bash
export PYTHONPATH=~/.claude/splatworld && python3 -m splatworld_agent.cli migrate-data --dry-run
```

## FORBIDDEN ACTIONS

- Do NOT summarize or interpret the output
- Do NOT ask confirmation questions (CLI handles consent)
- Do NOT run file operations manually
- Do NOT delete the old .splatworld_agent/ folder

## CORRECT BEHAVIOR

1. Run the bash command above
2. The CLI handles EVERYTHING:
   - Detects old .splatworld_agent/ folder
   - Shows what will be migrated
   - Prompts user for consent
   - Copies files preserving timestamps
   - Shows completion message

The CLI handles migration internally. You do not control this.
