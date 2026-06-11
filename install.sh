#!/usr/bin/env bash
# ============================================================
# PiEEG Agent: Remote installer
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/pieeg-club/PiEEG-agent/main/install.sh | bash
#
# This script:
#   1. Checks for git, installs if needed
#   2. Clones the PiEEG-agent repository
#   3. Runs setup.sh (which does the real work)
#
# Safe to re-run — it will pull the latest code if already cloned.
# ============================================================
set -euo pipefail

# --- Colors ---
if [ -t 1 ] && command -v tput &>/dev/null; then
    BOLD=$(tput bold)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RED=$(tput setaf 1)
    CYAN=$(tput setaf 6)
    RESET=$(tput sgr0)
else
    BOLD="" GREEN="" YELLOW="" RED="" CYAN="" RESET=""
fi

REPO_URL="https://github.com/pieeg-club/PiEEG-agent.git"
INSTALL_DIR="$HOME/PiEEG-agent"

echo ""
echo "${BOLD}${CYAN}╔══════════════════════════════════════════╗${RESET}"
echo "${BOLD}${CYAN}║       PiEEG Agent Installer             ║${RESET}"
echo "${BOLD}${CYAN}╚══════════════════════════════════════════╝${RESET}"
echo ""

# --- Check OS ---
OS_NAME="$(uname)"
echo "Detected OS: $OS_NAME"
echo ""

# --- Install git if needed ---
if ! command -v git &>/dev/null; then
    echo "Installing git..."
    case "$OS_NAME" in
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y -qq git
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y git
            elif command -v yum &>/dev/null; then
                sudo yum install -y git
            elif command -v pacman &>/dev/null; then
                sudo pacman -S --noconfirm git
            else
                echo "${RED}ERROR:${RESET} Could not detect package manager. Please install git manually."
                exit 1
            fi
            ;;
        Darwin)
            if ! command -v brew &>/dev/null; then
                echo "${RED}ERROR:${RESET} git not found and Homebrew not installed."
                echo "  Install Homebrew: https://brew.sh"
                echo "  Or install git from: https://git-scm.com/download/mac"
                exit 1
            fi
            brew install git
            ;;
        *)
            echo "${RED}ERROR:${RESET} Unsupported OS: $OS_NAME"
            echo "  Clone manually:"
            echo "    git clone $REPO_URL"
            echo "    cd PiEEG-agent"
            echo "    ./setup.sh"
            exit 1
            ;;
    esac
fi

# --- Clone or update ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating existing installation in $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull --ff-only origin main 2>/dev/null || {
        echo "${YELLOW}WARNING:${RESET} Could not update (local changes?). Using existing code."
    }
else
    echo "Cloning PiEEG-agent to $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR" || {
        echo "${RED}ERROR:${RESET} Failed to clone repository."
        echo "  Check your internet connection and try again."
        exit 1
    }
    cd "$INSTALL_DIR"
fi

# --- Run setup ---
echo ""
chmod +x setup.sh
exec ./setup.sh
