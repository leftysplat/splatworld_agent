---
name: splatworld-agent:prompt-history
description: View prompt variant history from training sessions
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
View all prompt variants tried during training, their ratings, and lineage.
</objective>

## Your task

View recent prompt history:
```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli prompt-history
```

View statistics:
```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli prompt-history --stats
```

View lineage for a specific variant:
```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli prompt-history --lineage <variant-id>
```

Filter by training session:
```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli prompt-history --session <session-id>
```

## Options

- `--limit, -n`: Number of entries to show (default: 20)
- `--session, -s`: Filter to a specific training session ID
- `--lineage, -l`: Show lineage for a specific variant ID
- `--stats`: Show statistics only
- `--json`: Output as JSON for programmatic use

## Use cases

1. **Review training progress**: See which variants were tried and how they were rated
2. **Understand prompt evolution**: Use `--lineage` to see how a successful prompt evolved
3. **Audit training sessions**: Filter by session to review specific training runs
4. **Export data**: Use `--json` to export history for analysis
