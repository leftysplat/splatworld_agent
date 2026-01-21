#!/bin/bash
#
# SplatWorld Agent Installer
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/leftysplat/splatworld_agent/main/install.sh | bash
#   OR
#   git clone ... && cd splatworld_agent && ./install.sh
#

set -e

# Save original directory to return to after install
ORIGINAL_DIR="$(pwd)"

# Parse arguments
AUTO_YES=false
for arg in "$@"; do
    case $arg in
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
    esac
done

# Auto-yes if not running in a terminal (e.g., piped or in CI)
if [ ! -t 0 ]; then
    AUTO_YES=true
fi

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
RESET='\033[0m'

# Default install location (inside ~/.claude like GSD)
DEFAULT_INSTALL_DIR="$HOME/.claude/splatworld-agent"
CLAUDE_COMMANDS_DIR="$HOME/.claude/commands"

echo ""
echo -e "${CYAN}   ███████╗██████╗ ██╗      █████╗ ████████╗"
echo -e "   ██╔════╝██╔══██╗██║     ██╔══██╗╚══██╔══╝"
echo -e "   ███████╗██████╔╝██║     ███████║   ██║   "
echo -e "   ╚════██║██╔═══╝ ██║     ██╔══██║   ██║   "
echo -e "   ███████║██║     ███████╗██║  ██║   ██║   "
echo -e "   ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝${RESET}"
echo ""
echo -e "   SplatWorld Agent Installer"
echo -e "   ${DIM}Iterative 3D splat generation with taste learning${RESET}"
echo ""

# Check for required tools
check_requirements() {
    local missing=()

    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi

    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi

    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        missing+=("pip")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}Missing required tools: ${missing[*]}${RESET}"
        echo -e "  Please install them and try again."
        exit 1
    fi
}

# Determine if running from within repo or standalone
detect_mode() {
    if [ -f "$(dirname "$0")/splatworld_agent/cli.py" ]; then
        # Running from within the repo
        REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
        echo -e "  ${DIM}Detected: Running from cloned repo${RESET}"
        MODE="local"
    else
        # Running standalone (e.g., curl | bash)
        MODE="download"
    fi
}

# Prompt for install location
prompt_location() {
    # Always install to ~/.claude/splatworld-agent for global access
    INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    echo -e "  ${DIM}Installing to: $INSTALL_DIR${RESET}"
}

# Clone or copy the repo to ~/.claude/splatworld-agent
install_repo() {
    local SOURCE_DIR=""

    if [ "$MODE" = "download" ]; then
        # Clone to temp location first, then copy
        echo -e "  Cloning repository..."
        TEMP_DIR=$(mktemp -d)
        git clone --quiet https://github.com/leftysplat/splatworld_agent.git "$TEMP_DIR/splatworld_agent"
        SOURCE_DIR="$TEMP_DIR/splatworld_agent"
        echo -e "  ${GREEN}✓${RESET} Cloned repository"
    else
        # Use current repo as source
        SOURCE_DIR="$REPO_DIR"
    fi

    # Copy/update to install location
    if [ -d "$INSTALL_DIR" ]; then
        echo -e "  ${DIM}Updating existing installation...${RESET}"
        # Preserve .git if it exists for updates
        if [ -d "$INSTALL_DIR/.git" ]; then
            cd "$INSTALL_DIR"
            git pull --ff-only 2>/dev/null || {
                # If pull fails, do a fresh copy
                rm -rf "$INSTALL_DIR"
                cp -r "$SOURCE_DIR" "$INSTALL_DIR"
            }
            echo -e "  ${GREEN}✓${RESET} Updated installation"
        else
            rm -rf "$INSTALL_DIR"
            cp -r "$SOURCE_DIR" "$INSTALL_DIR"
            echo -e "  ${GREEN}✓${RESET} Reinstalled to $INSTALL_DIR"
        fi
    else
        mkdir -p "$(dirname "$INSTALL_DIR")"
        cp -r "$SOURCE_DIR" "$INSTALL_DIR"
        echo -e "  ${GREEN}✓${RESET} Installed to $INSTALL_DIR"
    fi

    # Clean up temp dir if used
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Install Python package
install_python() {
    echo -e "  Installing Python package..."

    cd "$INSTALL_DIR"

    # Try pip install (non-editable for better compatibility)
    local pip_cmd="pip3"
    if ! command -v pip3 &> /dev/null; then
        pip_cmd="pip"
    fi

    # Uninstall any existing installation first (avoids conflicts)
    $pip_cmd uninstall -y splatworld-agent 2>/dev/null || true

    # Try with --user flag for permission issues
    if $pip_cmd install . --user --quiet 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} Installed Python package"
    elif $pip_cmd install . --user 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} Installed Python package"
    else
        echo -e "  ${YELLOW}⚠${RESET} Python package install failed (optional)"
        echo -e "    ${DIM}The slash commands will still work via PYTHONPATH${RESET}"
        echo -e "    ${DIM}To fix: upgrade pip with 'pip3 install --upgrade pip'${RESET}"
    fi
}

# Set up Claude Code integration
setup_claude() {
    echo -e "  Setting up Claude Code integration..."

    # Create commands directory if needed
    mkdir -p "$CLAUDE_COMMANDS_DIR"

    # Remove existing (file or symlink)
    if [ -e "$CLAUDE_COMMANDS_DIR/splatworld-agent" ] || [ -L "$CLAUDE_COMMANDS_DIR/splatworld-agent" ]; then
        rm -rf "$CLAUDE_COMMANDS_DIR/splatworld-agent"
    fi

    # Create symlink
    ln -s "$INSTALL_DIR/commands" "$CLAUDE_COMMANDS_DIR/splatworld-agent"

    echo -e "  ${GREEN}✓${RESET} Linked commands to ~/.claude/commands/splatworld-agent"

    # Update PYTHONPATH in all command files to point to install location
    echo -e "  Updating command paths..."
    for cmd_file in "$INSTALL_DIR/commands"/*.md; do
        if [ -f "$cmd_file" ]; then
            # Replace the PYTHONPATH line with the correct install dir
            sed -i '' "s|PYTHONPATH=~/Documents/splatworld_agent|PYTHONPATH=$INSTALL_DIR|g" "$cmd_file"
            sed -i '' "s|PYTHONPATH=.*/splatworld_agent|PYTHONPATH=$INSTALL_DIR|g" "$cmd_file"
        fi
    done
    echo -e "  ${GREEN}✓${RESET} Updated command paths to $INSTALL_DIR"
}

# Print completion message
print_done() {
    # Return to original directory if we cd'd into the repo
    if [ -n "$ORIGINAL_DIR" ] && [ "$ORIGINAL_DIR" != "$(pwd)" ]; then
        cd "$ORIGINAL_DIR"
    fi

    echo ""
    echo -e "  ${GREEN}Installation complete!${RESET}"
    echo ""
    echo -e "  ${YELLOW}Installed to:${RESET} ~/.claude/splatworld-agent"
    echo -e "  ${YELLOW}Commands at:${RESET} ~/.claude/commands/splatworld-agent"
    echo -e "  ${YELLOW}Available from:${RESET} Any project directory"
    echo ""
    echo -e "  ${YELLOW}Next steps:${RESET}"
    echo -e "  1. Run ${CYAN}/clear${RESET} or restart Claude Code"
    echo -e "  2. Run ${CYAN}/splatworld-agent:help${RESET} to see available commands"
    echo -e "  3. Run ${CYAN}/splatworld-agent:init${RESET} in any project to start"
    echo ""
    echo -e "  ${YELLOW}To update later:${RESET}"
    echo -e "  Run ${CYAN}/splatworld-agent:update${RESET} from anywhere"
    echo ""
}

# Main
main() {
    check_requirements
    detect_mode

    # Always install to ~/.claude/splatworld-agent
    prompt_location
    install_repo
    install_python
    setup_claude
    print_done
}

main "$@"
