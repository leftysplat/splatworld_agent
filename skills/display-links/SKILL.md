---
name: display-links
description: Display World Labs viewer links for all converted splats
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}*python3*splatworld_agent.cli*)
---

Display all World Labs viewer links for converted splats.

Run this command:

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
cd {{cwd}} && PYTHONPATH="${PLUGIN_ROOT}" python3 -m splatworld_agent.cli splats
```

This shows all 3D splats that have been converted, with clickable links to view them in the World Labs viewer.

To open a specific splat's viewer directly:

```bash
PLUGIN_ROOT=$("${CLAUDE_PLUGIN_ROOT}/.resolver.sh" 2>/dev/null || echo "${HOME}/.claude/splatworld")
cd {{cwd}} && PYTHONPATH="${PLUGIN_ROOT}" python3 -m splatworld_agent.cli splats --open <generation-id>
```
