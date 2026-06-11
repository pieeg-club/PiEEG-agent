#!/usr/bin/env bash
# ============================================================
# PiEEG Agent: One-command setup script
# Cross-platform installation for Linux, macOS, WSL
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Detects your OS and validates compatibility
#   2. Installs system packages (python3, pip, venv)
#   3. Creates a Python venv and installs pieeg-agent
#   4. Symlinks 'pieeg-agent' to PATH (optional)
#   5. Verifies installation works
#
# Note: Frontend is prebuilt — no Node.js required.
# ============================================================
set -euo pipefail

# --- Colors & formatting ---
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

ok()   { echo "  ${GREEN}✓${RESET} $*"; }
warn() { echo "  ${YELLOW}⚠${RESET} $*"; }
fail() { echo "  ${RED}✗${RESET} $*"; }
step() { echo ""; echo "${BOLD}${CYAN}[$1]${RESET} ${BOLD}$2${RESET}"; }

die() {
    echo ""
    fail "$1"
    echo ""
    echo "  Setup failed. You can re-run this script after fixing the issue above."
    echo "  For help: https://github.com/pieeg-club/PiEEG-agent/issues"
    exit 1
}

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
OS_NAME="$(uname)"

# ============================================================
# Step 1: Detect OS and environment
# ============================================================
step "1/4" "Detecting environment..."

case "$OS_NAME" in
    Linux)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            ok "Linux (WSL on Windows)"
            IS_WSL=1
        else
            ok "Linux"
            IS_WSL=0
        fi
        ;;
    Darwin)
        ok "macOS"
        IS_WSL=0
        ;;
    *)
        die "Unsupported OS: $OS_NAME. This script is for Linux/macOS. On Windows, use setup.cmd or install manually."
        ;;
esac

# ============================================================
# Step 2: Check prerequisites
# ============================================================
step "2/4" "Checking prerequisites..."

# Check for package manager and install tools if needed
install_system_packages() {
    case "$OS_NAME" in
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq 2>&1 | tail -1 || die "apt-get update failed"
                sudo apt-get install -y -qq python3 python3-pip python3-venv curl 2>&1 | tail -1 || \
                    die "Failed to install system packages"
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y python3 python3-pip curl || die "Failed to install system packages"
            elif command -v yum &>/dev/null; then
                sudo yum install -y python3 python3-pip curl || die "Failed to install system packages"
            elif command -v pacman &>/dev/null; then
                sudo pacman -S --noconfirm python python-pip curl || die "Failed to install system packages"
            else
                warn "Could not detect package manager. Make sure python3, pip, curl are installed."
            fi
            ;;
        Darwin)
            # macOS should have Python 3, but check
            if ! command -v python3 &>/dev/null; then
                die "Python 3 not found. Install from https://www.python.org/downloads/ or via Homebrew: brew install python@3"
            fi
            ;;
    esac
}

install_system_packages

# Verify Python version
if ! command -v python3 &>/dev/null; then
    die "python3 not found after installation attempt"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || \
    die "python3 not working"

PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    die "Python 3.10+ is required, but found Python $PY_VERSION. Upgrade Python or your OS."
fi

ok "Python $PY_VERSION"

# ============================================================
# Step 3: Create venv & install pieeg-agent
# ============================================================
step "3/4" "Installing pieeg-agent..."

cd "$INSTALL_DIR"

# Create venv
if [ ! -d ".venv" ]; then
    python3 -m venv .venv || die "Failed to create virtual environment"
fi

# shellcheck source=/dev/null
source .venv/bin/activate || die "Failed to activate virtual environment"

pip install --upgrade pip -q 2>&1 | tail -1 || die "Failed to upgrade pip"

echo "  Installing dependencies..."
pip install -e ".[web,dev]" -q 2>&1 | tail -3 || die "Failed to install pieeg-agent"

ok "Installed to: $INSTALL_DIR/.venv"

# Check frontend is present
if [ -d "$INSTALL_DIR/frontend/dist" ] && [ -n "$(ls -A $INSTALL_DIR/frontend/dist 2>/dev/null)" ]; then
    ok "Frontend: prebuilt React app included"
else
    warn "Frontend dist/ not found. Run: cd frontend && npm install && npm run build"
fi

# ============================================================
# Step 4: Add to PATH (optional)
# ============================================================
step "4/4" "Setting up command-line access..."

# Symlink to /usr/local/bin if we have permission
if [ -w /usr/local/bin ]; then
    ln -sf "$INSTALL_DIR/.venv/bin/pieeg-agent" /usr/local/bin/pieeg-agent 2>/dev/null || true
    if command -v pieeg-agent &>/dev/null; then
        ok "Command linked: /usr/local/bin/pieeg-agent"
    fi
elif command -v sudo &>/dev/null; then
    if sudo -n true 2>/dev/null; then
        sudo ln -sf "$INSTALL_DIR/.venv/bin/pieeg-agent" /usr/local/bin/pieeg-agent
        ok "Command linked: /usr/local/bin/pieeg-agent (with sudo)"
    else
        warn "Cannot link to /usr/local/bin without sudo"
        echo "     Run: sudo ln -s $INSTALL_DIR/.venv/bin/pieeg-agent /usr/local/bin/pieeg-agent"
        echo "     Or use: $INSTALL_DIR/.venv/bin/pieeg-agent"
    fi
fi

# Verify installation
echo ""
echo "${BOLD}Verifying installation...${RESET}"

VERIFY_PASS=true

# Check entry point
if "$INSTALL_DIR/.venv/bin/pieeg-agent" --help &>/dev/null; then
    ok "pieeg-agent --help works"
else
    fail "pieeg-agent --help failed"
    VERIFY_PASS=false
fi

# Check core imports
if "$INSTALL_DIR/.venv/bin/python" -c "
from pieeg_agent.agent import copilot
from pieeg_agent.llm import factory
from pieeg_agent.decode import session
from pieeg_agent.ingest import lsl_inlet
" 2>/dev/null; then
    ok "All Python modules import successfully"
else
    fail "Module import failed"
    VERIFY_PASS=false
fi

# Check web imports if installed
if "$INSTALL_DIR/.venv/bin/python" -c "
from pieeg_agent.web import app
from pieeg_agent.web import engine
" 2>/dev/null; then
    ok "Web server modules available"
else
    warn "Web server modules not available (install with: pip install -e '.[web]')"
fi

# ============================================================
# Done!
# ============================================================
echo ""
echo "${BOLD}${GREEN}=== Setup complete! ===${RESET}"
echo ""

echo "  Start the web interface:"
echo "    ${BOLD}pieeg-agent web${RESET}"
echo ""
echo "  Dashboard: ${CYAN}http://localhost:8080${RESET}"
echo ""
echo "  For LSL stream integration, connect to a PiEEG-server or any LSL source."
echo ""
echo "  Explore commands:"
echo "    ${BOLD}pieeg-agent --help${RESET}"
echo ""

if [ "$VERIFY_PASS" != true ]; then
    echo "${YELLOW}⚠  Some verification checks failed. The installation may still work.${RESET}"
    echo "   If you encounter issues, report at: https://github.com/pieeg-club/PiEEG-agent/issues"
    echo ""
fi
