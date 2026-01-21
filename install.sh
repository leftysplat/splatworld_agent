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

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
RESET='\033[0m'

# Default install location
DEFAULT_INSTALL_DIR="$HOME/.local/share/splatworld_agent"
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
    echo -e "  ${YELLOW}Where would you like to install?${RESET}"
    echo ""
    echo -e "  ${CYAN}1${RESET}) Default ${DIM}($DEFAULT_INSTALL_DIR)${RESET}"
    echo -e "  ${CYAN}2${RESET}) Current directory ${DIM}($(pwd)/splatworld_agent)${RESET}"
    echo -e "  ${CYAN}3${RESET}) Custom location"
    echo ""
    read -p "  Choice [1]: " choice
    choice=${choice:-1}

    case $choice in
        1)
            INSTALL_DIR="$DEFAULT_INSTALL_DIR"
            ;;
        2)
            INSTALL_DIR="$(pwd)/splatworld_agent"
            ;;
        3)
            read -p "  Enter path: " custom_path
            INSTALL_DIR="${custom_path/#\~/$HOME}"
            ;;
        *)
            INSTALL_DIR="$DEFAULT_INSTALL_DIR"
            ;;
    esac
}

# Clone or copy the repo
install_repo() {
    if [ "$MODE" = "download" ]; then
        echo -e "  Cloning repository..."

        if [ -d "$INSTALL_DIR" ]; then
            echo -e "  ${YELLOW}Directory exists: $INSTALL_DIR${RESET}"
            read -p "  Update existing installation? [Y/n]: " update
            update=${update:-Y}
            if [[ $update =~ ^[Yy] ]]; then
                cd "$INSTALL_DIR"
                git pull --ff-only
                echo -e "  ${GREEN}✓${RESET} Updated repository"
            fi
        else
            mkdir -p "$(dirname "$INSTALL_DIR")"
            git clone https://github.com/leftysplat/splatworld_agent.git "$INSTALL_DIR"
            echo -e "  ${GREEN}✓${RESET} Cloned repository"
        fi
    else
        # Running from within repo - use current location or copy
        if [ "$REPO_DIR" != "$INSTALL_DIR" ]; then
            if [ -d "$INSTALL_DIR" ]; then
                echo -e "  ${YELLOW}Directory exists: $INSTALL_DIR${RESET}"
                read -p "  Replace? [y/N]: " replace
                if [[ $replace =~ ^[Yy] ]]; then
                    rm -rf "$INSTALL_DIR"
                    cp -r "$REPO_DIR" "$INSTALL_DIR"
                    echo -e "  ${GREEN}✓${RESET} Copied to $INSTALL_DIR"
                fi
            else
                mkdir -p "$(dirname "$INSTALL_DIR")"
                cp -r "$REPO_DIR" "$INSTALL_DIR"
                echo -e "  ${GREEN}✓${RESET} Copied to $INSTALL_DIR"
            fi
        else
            INSTALL_DIR="$REPO_DIR"
            echo -e "  ${GREEN}✓${RESET} Using current directory"
        fi
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
}

# Print completion message
print_done() {
    echo ""
    echo -e "  ${GREEN}Installation complete!${RESET}"
    echo ""
    echo -e "  ${YELLOW}Installed to:${RESET} $INSTALL_DIR"
    echo -e "  ${YELLOW}Commands at:${RESET} ~/.claude/commands/splatworld-agent"
    echo ""
    echo -e "  ${YELLOW}Next steps:${RESET}"
    echo -e "  1. Restart Claude Code (or start a new conversation)"
    echo -e "  2. Run ${CYAN}/splatworld-agent:help${RESET} to see available commands"
    echo -e "  3. Run ${CYAN}/splatworld-agent:init${RESET} in a project directory"
    echo ""
    echo -e "  ${YELLOW}To update later:${RESET}"
    echo -e "  Run ${CYAN}/splatworld-agent:update${RESET} in Claude Code"
    echo ""
}

# Main
main() {
    check_requirements
    detect_mode

    if [ "$MODE" = "local" ]; then
        # Running from repo - ask if they want to install here or elsewhere
        echo -e "  ${DIM}Running from: $REPO_DIR${RESET}"
        echo ""
        read -p "  Install from current location? [Y/n]: " use_current
        use_current=${use_current:-Y}

        if [[ $use_current =~ ^[Yy] ]]; then
            INSTALL_DIR="$REPO_DIR"
        else
            prompt_location
            install_repo
        fi
    else
        prompt_location
        install_repo
    fi

    install_python
    setup_claude
    print_done
}

main "$@"
