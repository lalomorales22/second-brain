#!/usr/bin/env bash
set -e

# ============================================================
#  Second Brain — Install Script
#
#  Installs everything needed to run `second-brain` from
#  anywhere in your terminal.
#
#  Usage:
#    chmod +x install.sh && ./install.sh
#
# ============================================================

BOLD='\033[1m'
DIM='\033[2m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

info()  { echo -e "${BLUE}[info]${RESET}  $1"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $1"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $1"; }
fail()  { echo -e "${RED}[fail]${RESET}  $1"; exit 1; }

echo ""
echo -e "${BOLD}  second brain — installer${RESET}"
echo -e "${DIM}  crystals . spreading activation . temporal gravity${RESET}"
echo ""

# ----------------------------------------------------------
# 1. Check Python 3.10+
# ----------------------------------------------------------
info "Checking Python..."

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    fail "Python not found. Install Python 3.10+ from https://python.org"
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    fail "Python 3.10+ required (found $PY_VERSION)"
fi

ok "Python $PY_VERSION ($PYTHON)"

# ----------------------------------------------------------
# 2. Check tkinter (optional, for desktop GUI)
# ----------------------------------------------------------
info "Checking tkinter..."

if $PYTHON -c "import tkinter" 2>/dev/null; then
    ok "tkinter available (desktop GUI ready)"
else
    warn "tkinter not found — desktop GUI won't work, but 3D brain is fine"
fi

# ----------------------------------------------------------
# 3. Install the package
# ----------------------------------------------------------
info "Installing semantic-gravity-memory..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Figure out where pip will put scripts
USER_BIN=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('scripts', 'posix_user'))")

# Try standard pip install first, fall back to --user --break-system-packages
# (needed for Homebrew Python / PEP 668 managed environments)
if $PYTHON -m pip install -e "$SCRIPT_DIR" --quiet 2>/dev/null; then
    ok "Installed via pip"
elif $PYTHON -m pip install --user --break-system-packages -e "$SCRIPT_DIR" --quiet 2>/dev/null; then
    ok "Installed via pip --user"
elif $PYTHON -m pip install --user -e "$SCRIPT_DIR" --quiet 2>/dev/null; then
    ok "Installed via pip --user"
else
    fail "pip install failed — try running: $PYTHON -m pip install -e $SCRIPT_DIR"
fi

# ----------------------------------------------------------
# 4. Ensure scripts directory is in PATH
# ----------------------------------------------------------
info "Checking PATH..."

# Find where second-brain was actually installed
BRAIN_BIN=""
if command -v second-brain &>/dev/null; then
    BRAIN_BIN=$(command -v second-brain)
    ok "'second-brain' is already on PATH: $BRAIN_BIN"
elif [ -f "$USER_BIN/second-brain" ]; then
    BRAIN_BIN="$USER_BIN/second-brain"

    # Detect shell config file
    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi

    # Add to PATH if not already there
    if [ -n "$SHELL_RC" ]; then
        if ! grep -q "$USER_BIN" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Second Brain (semantic-gravity-memory)" >> "$SHELL_RC"
            echo "export PATH=\"$USER_BIN:\$PATH\"" >> "$SHELL_RC"
            ok "Added $USER_BIN to PATH in $SHELL_RC"
        else
            ok "$USER_BIN already in $SHELL_RC"
        fi
        # Also export for current session
        export PATH="$USER_BIN:$PATH"
    else
        warn "Could not find shell config file. Add this to your shell profile:"
        echo ""
        echo -e "    export PATH=\"$USER_BIN:\$PATH\""
        echo ""
    fi
else
    # Check other common locations
    for DIR in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin"; do
        if [ -f "$DIR/second-brain" ]; then
            BRAIN_BIN="$DIR/second-brain"
            ok "Found second-brain at $DIR"
            break
        fi
    done
    if [ -z "$BRAIN_BIN" ]; then
        warn "Could not find 'second-brain' command — try restarting your terminal"
    fi
fi

# ----------------------------------------------------------
# 5. Check Ollama
# ----------------------------------------------------------
echo ""
info "Checking Ollama..."

if command -v ollama &>/dev/null; then
    ok "Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama is running"

        # List available models
        MODELS=$(curl -s http://localhost:11434/api/tags | $PYTHON -c "
import sys, json
data = json.load(sys.stdin)
names = [m['name'] for m in data.get('models', [])]
print(', '.join(names[:8]) if names else 'none')
" 2>/dev/null || echo "could not list")
        info "Models available: $MODELS"

        # Check for embedding model
        HAS_EMBED=$(curl -s http://localhost:11434/api/tags | $PYTHON -c "
import sys, json
data = json.load(sys.stdin)
names = [m['name'] for m in data.get('models', [])]
has = any('minilm' in n or 'embed' in n or 'nomic' in n for n in names)
print('yes' if has else 'no')
" 2>/dev/null || echo "no")

        if [ "$HAS_EMBED" = "no" ]; then
            warn "No embedding model found. Pulling all-minilm..."
            ollama pull all-minilm && ok "Pulled all-minilm" || warn "Could not pull — run 'ollama pull all-minilm' manually"
        else
            ok "Embedding model available"
        fi
    else
        warn "Ollama installed but not running — start it with: ollama serve"
    fi
else
    warn "Ollama not found"
    echo ""
    echo -e "    ${DIM}Ollama gives Second Brain embeddings and chat.${RESET}"
    echo -e "    ${DIM}Install from: https://ollama.com${RESET}"
    echo -e "    ${DIM}Then run:${RESET}"
    echo -e "    ${DIM}  ollama pull all-minilm     # embeddings${RESET}"
    echo -e "    ${DIM}  ollama pull gpt-oss:20b     # chat model${RESET}"
    echo ""
fi

# ----------------------------------------------------------
# 6. Done
# ----------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}  Installation complete!${RESET}"
echo ""
echo -e "  ${YELLOW}NOTE: restart your terminal (or run 'source ~/.zshrc') for PATH changes${RESET}"
echo ""
echo -e "  Start the 3D brain:"
echo -e "    ${BOLD}second-brain${RESET}"
echo -e "    ${DIM}opens http://localhost:8487${RESET}"
echo ""
echo -e "  Or with options:"
echo -e "    ${BOLD}second-brain --chat-model gpt-oss:20b --port 8487${RESET}"
echo ""
echo -e "  Desktop GUI (tkinter):"
echo -e "    ${BOLD}semantic-gravity-lab${RESET}"
echo ""
echo -e "  Python API:"
echo -e "    ${DIM}from semantic_gravity_memory import Memory${RESET}"
echo -e "    ${DIM}m = Memory()${RESET}"
echo -e "    ${DIM}m.ingest('hello world')${RESET}"
echo ""
