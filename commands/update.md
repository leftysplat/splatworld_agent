---
name: splatworld-agent:update
description: Update SplatWorld Agent to latest version
allowed-tools: Bash(PYTHONPATH*python3*splatworld_agent.cli*)
---

<objective>
Update SplatWorld Agent by pulling the latest changes from the git repository.
</objective>

## Your task

```bash
export PYTHONPATH=~/Documents/splatworld_agent
python3 -m splatworld_agent.cli update
```

This will:
1. Fetch updates from the remote repository
2. Show new commits available
3. Pull the latest changes (fast-forward only)
4. Display update summary

### If you have local changes

If the update fails due to local changes, you can:
```bash
cd ~/Documents/splatworld_agent
git stash
git pull
git stash pop
```

### After updating

Run `/splatworld-agent:help` to see any new commands or features.
