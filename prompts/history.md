<purpose>
Browse past generations in the current project.
Shows recent generations with their prompts and feedback.
</purpose>

<arguments>
- --limit/-n: Number of generations to show (default: 10)
- generation_id: View a specific generation in detail
</arguments>

<execution>
For listing recent:
```bash
splatworld-agent history [--limit <n>]
```

For viewing specific generation:
```bash
splatworld-agent history <generation_id>
```

After displaying:
- Show where generation files are stored
- If viewing specific generation, show full details including file paths
- Offer to provide feedback if the generation doesn't have any
</execution>
