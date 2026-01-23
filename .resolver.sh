#!/bin/bash
# .resolver.sh - Dynamic plugin root path resolution for dual install support
# Resolves plugin root with three fallback levels:
# 1. CLAUDE_PLUGIN_ROOT environment variable (future-proof)
# 2. Parse ~/.claude/plugins/installed_plugins.json
# 3. Fall back to manual install at ~/.claude/splatworld

PLUGIN_NAME="splatworld"

# Level 1: Check CLAUDE_PLUGIN_ROOT environment variable
if [ -n "${CLAUDE_PLUGIN_ROOT}" ] && [ -d "${CLAUDE_PLUGIN_ROOT}" ]; then
    echo "${CLAUDE_PLUGIN_ROOT}" | sed 's:/$::'
    exit 0
fi

# Level 2: Parse installed_plugins.json
INSTALLED_PLUGINS="${HOME}/.claude/plugins/installed_plugins.json"
if [ -f "${INSTALLED_PLUGINS}" ]; then
    PLUGIN_PATH=$(python3 -c "
import json
import sys
try:
    with open('${INSTALLED_PLUGINS}', 'r') as f:
        data = json.load(f)
    if 'installedPlugins' in data:
        for plugin in data['installedPlugins']:
            if plugin.get('name') == '${PLUGIN_NAME}':
                path = plugin.get('installPath', '').rstrip('/')
                if path:
                    print(path)
                    sys.exit(0)
except:
    pass
" 2>/dev/null)

    if [ -n "${PLUGIN_PATH}" ] && [ -d "${PLUGIN_PATH}" ]; then
        echo "${PLUGIN_PATH}"
        exit 0
    fi
fi

# Level 3: Fall back to manual install location
MANUAL_PATH="${HOME}/.claude/${PLUGIN_NAME}"
if [ -d "${MANUAL_PATH}" ]; then
    echo "${MANUAL_PATH}"
    exit 0
fi

# If all fallbacks fail, return manual path anyway (last resort)
echo "${MANUAL_PATH}"
exit 0
