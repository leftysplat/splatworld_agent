<purpose>
View or edit the current taste profile.
Shows learned preferences, exemplars, and statistics.
</purpose>

<arguments>
- edit: Open profile.json for manual editing
- --json: Output as raw JSON
</arguments>

<process>
1. Load the taste profile
2. Display in requested format (pretty or JSON)
3. If edit mode, open in editor
</process>

<execution>
Determine what the user wants:

For viewing:
```bash
splatworld profile
```

For JSON output:
```bash
splatworld profile --json
```

For editing:
```bash
splatworld profile --edit
```

After displaying, explain:
- What each section means
- How confidence scores work (0-1, higher = more consistent pattern)
- How to manually adjust preferences if desired
- That preferences are automatically updated when running /splatworld:learn
</execution>
