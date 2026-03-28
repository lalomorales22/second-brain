#!/usr/bin/env bash
set -e

# ============================================================
#  Second Brain — Universal Installer (macOS / Linux)
#
#  Works two ways:
#    1. From a cloned repo:  ./install.sh
#    2. One-liner from anywhere:
#       curl -fsSL https://raw.githubusercontent.com/lalomorales22/second-brain/main/install.sh | bash
#
#  What it does:
#    - Checks Python 3.10+
#    - Installs the package (from local repo or GitHub)
#    - Checks/installs Ollama
#    - Pulls an embedding model if needed
#    - Detects WebGL issues on Linux and offers fixes
#    - Puts 'second-brain' on your PATH
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
echo -e "${DIM}  persistent memory engine for AI agents${RESET}"
echo ""

OS="$(uname -s)"
ARCH="$(uname -m)"
info "System: $OS $ARCH"

# ----------------------------------------------------------
# 1. Check Python 3.10+
# ----------------------------------------------------------
info "Checking Python..."

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY_MAJOR=$($cmd -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
        PY_MINOR=$($cmd -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    fail "Python 3.10+ required. Install from https://python.org"
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
ok "Python $PY_VERSION ($PYTHON)"

# ----------------------------------------------------------
# 2. Check pip
# ----------------------------------------------------------
info "Checking pip..."

if ! $PYTHON -m pip --version &>/dev/null; then
    warn "pip not found, trying to install..."
    if [ "$OS" = "Linux" ]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip 2>/dev/null || true
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3-pip 2>/dev/null || true
        fi
    fi
    if ! $PYTHON -m pip --version &>/dev/null; then
        fail "pip not found. Install it: $PYTHON -m ensurepip --upgrade"
    fi
fi
ok "pip available"

# ----------------------------------------------------------
# 3. Install the package
# ----------------------------------------------------------
info "Installing second-brain..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" 2>/dev/null || echo ".")" && pwd)"
USER_BIN=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('scripts', 'posix_user'))" 2>/dev/null || echo "$HOME/.local/bin")

# Determine install source: local repo or GitHub
INSTALL_SRC=""
if [ -f "$SCRIPT_DIR/pyproject.toml" ] && grep -q "semantic-gravity-memory" "$SCRIPT_DIR/pyproject.toml" 2>/dev/null; then
    INSTALL_SRC="$SCRIPT_DIR"
    info "Installing from local repo: $SCRIPT_DIR"
else
    INSTALL_SRC="git+https://github.com/lalomorales22/second-brain.git"
    info "Installing from GitHub..."
fi

# Try install methods in order
INSTALLED=false

# Method 1: standard pip install
if [ "$INSTALL_SRC" = "$SCRIPT_DIR" ]; then
    INSTALL_FLAG="-e"
else
    INSTALL_FLAG=""
fi

if $PYTHON -m pip install $INSTALL_FLAG "$INSTALL_SRC" --quiet 2>/dev/null; then
    INSTALLED=true
    ok "Installed via pip"
elif $PYTHON -m pip install --user --break-system-packages $INSTALL_FLAG "$INSTALL_SRC" --quiet 2>/dev/null; then
    INSTALLED=true
    ok "Installed via pip --user"
elif $PYTHON -m pip install --user $INSTALL_FLAG "$INSTALL_SRC" --quiet 2>/dev/null; then
    INSTALLED=true
    ok "Installed via pip --user"
fi

if [ "$INSTALLED" = false ]; then
    fail "pip install failed. Try manually: $PYTHON -m pip install $INSTALL_FLAG $INSTALL_SRC"
fi

# ----------------------------------------------------------
# 4. Verify the install
# ----------------------------------------------------------
info "Verifying..."

if $PYTHON -c "from semantic_gravity_memory import Memory; m = Memory(db_path=':memory:'); m.ingest('test'); print('OK')" 2>/dev/null | grep -q "OK"; then
    ok "Package works"
else
    warn "Package installed but import test failed — may still work"
fi

# ----------------------------------------------------------
# 5. Ensure scripts directory is in PATH
# ----------------------------------------------------------
info "Checking PATH..."

BRAIN_BIN=""
if command -v second-brain &>/dev/null; then
    BRAIN_BIN=$(command -v second-brain)
    ok "'second-brain' is on PATH: $BRAIN_BIN"
elif [ -f "$USER_BIN/second-brain" ]; then
    BRAIN_BIN="$USER_BIN/second-brain"

    # Detect shell config file
    SHELL_RC=""
    if [ -n "$ZSH_VERSION" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi

    if [ -n "$SHELL_RC" ]; then
        if ! grep -q "$USER_BIN" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Second Brain" >> "$SHELL_RC"
            echo "export PATH=\"$USER_BIN:\$PATH\"" >> "$SHELL_RC"
            ok "Added $USER_BIN to PATH in $SHELL_RC"
        else
            ok "$USER_BIN already in $SHELL_RC"
        fi
        export PATH="$USER_BIN:$PATH"
    else
        warn "Add this to your shell profile:"
        echo -e "    export PATH=\"$USER_BIN:\$PATH\""
    fi
else
    for DIR in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin"; do
        if [ -f "$DIR/second-brain" ]; then
            BRAIN_BIN="$DIR/second-brain"
            ok "Found second-brain at $DIR"
            break
        fi
    done
    if [ -z "$BRAIN_BIN" ]; then
        warn "Could not find 'second-brain' command — restart your terminal"
    fi
fi

# ----------------------------------------------------------
# 6. Check/install Ollama
# ----------------------------------------------------------
echo ""
info "Checking Ollama..."

OLLAMA_RUNNING=false
OLLAMA_INSTALLED=false

if command -v ollama &>/dev/null; then
    OLLAMA_INSTALLED=true
    ok "Ollama is installed"
fi

if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    OLLAMA_RUNNING=true
    ok "Ollama is running"
fi

if [ "$OLLAMA_INSTALLED" = false ]; then
    echo ""
    info "Ollama not found. Installing..."

    if [ "$OS" = "Darwin" ]; then
        # macOS — check for Homebrew first
        if command -v brew &>/dev/null; then
            brew install ollama 2>/dev/null && OLLAMA_INSTALLED=true && ok "Installed via Homebrew" || true
        fi
        if [ "$OLLAMA_INSTALLED" = false ]; then
            warn "Install Ollama manually from https://ollama.com"
        fi
    elif [ "$OS" = "Linux" ]; then
        # Linux — use the official install script
        if curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null; then
            OLLAMA_INSTALLED=true
            ok "Installed Ollama"
        else
            warn "Could not auto-install Ollama. Get it from https://ollama.com"
        fi
    fi
fi

# Start Ollama if installed but not running
if [ "$OLLAMA_INSTALLED" = true ] && [ "$OLLAMA_RUNNING" = false ]; then
    info "Starting Ollama..."
    if [ "$OS" = "Darwin" ]; then
        # macOS: open the app or start serve
        open -a Ollama 2>/dev/null || ollama serve &>/dev/null &
    else
        # Linux: check systemd, or start in background
        if systemctl is-active --quiet ollama 2>/dev/null; then
            OLLAMA_RUNNING=true
        else
            systemctl start ollama 2>/dev/null || (ollama serve &>/dev/null &)
        fi
    fi

    # Wait for it to come up
    for i in $(seq 1 10); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            OLLAMA_RUNNING=true
            ok "Ollama is running"
            break
        fi
        sleep 1
    done

    if [ "$OLLAMA_RUNNING" = false ]; then
        warn "Ollama installed but could not start. Try: ollama serve"
    fi
fi

# ----------------------------------------------------------
# 7. Pull embedding model if needed
# ----------------------------------------------------------
if [ "$OLLAMA_RUNNING" = true ]; then
    HAS_EMBED=$($PYTHON -c "
import json, urllib.request
data = json.loads(urllib.request.urlopen('http://localhost:11434/api/tags').read())
names = [m['name'] for m in data.get('models', [])]
print('yes' if any('minilm' in n or 'embed' in n or 'nomic' in n for n in names) else 'no')
" 2>/dev/null || echo "no")

    if [ "$HAS_EMBED" = "no" ]; then
        info "No embedding model found. Pulling all-minilm (23MB)..."
        ollama pull all-minilm 2>/dev/null && ok "Pulled all-minilm" || warn "Could not pull — run 'ollama pull all-minilm' manually"
    else
        ok "Embedding model available"
    fi

    # Show available models
    MODELS=$($PYTHON -c "
import json, urllib.request
data = json.loads(urllib.request.urlopen('http://localhost:11434/api/tags').read())
names = [m['name'] for m in data.get('models', [])]
print(', '.join(names[:6]))
" 2>/dev/null || echo "could not list")
    info "Available models: $MODELS"
fi

# ----------------------------------------------------------
# 8. Check WebGL (Linux only)
# ----------------------------------------------------------
if [ "$OS" = "Linux" ]; then
    echo ""
    info "Checking WebGL support (needed for 3D brain)..."

    WEBGL_OK=true
    GPU_INFO=""

    # Check if we're on a Jetson or ARM device with limited GPU
    if [ -f /proc/device-tree/model ]; then
        DEVICE_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
        info "Device: $DEVICE_MODEL"
    fi

    # Check for GPU rendering support
    if command -v glxinfo &>/dev/null; then
        RENDERER=$(glxinfo 2>/dev/null | grep "OpenGL renderer" | head -1 || echo "")
        if echo "$RENDERER" | grep -qi "llvmpipe\|swrast\|software\|disabled"; then
            WEBGL_OK=false
            GPU_INFO="software rendering detected"
        elif [ -n "$RENDERER" ]; then
            ok "GPU: $RENDERER"
        fi
    fi

    if [ "$WEBGL_OK" = false ] || [ -z "$GPU_INFO" ]; then
        warn "WebGL may not work in your browser (3D brain needs it)"
        echo ""
        echo -e "  ${YELLOW}Fixes to try:${RESET}"
        echo ""
        echo -e "  ${BOLD}1. Launch Chrome/Chromium with GPU flags:${RESET}"
        echo -e "     ${DIM}google-chrome --ignore-gpu-blocklist --enable-gpu-rasterization --enable-webgl${RESET}"
        echo -e "     ${DIM}chromium-browser --ignore-gpu-blocklist --enable-gpu-rasterization --enable-webgl${RESET}"
        echo ""
        echo -e "  ${BOLD}2. Or in Chrome, go to:${RESET}"
        echo -e "     ${DIM}chrome://flags/#ignore-gpu-blocklist  → Enable${RESET}"
        echo -e "     ${DIM}chrome://flags/#enable-webgl-draft-extensions  → Enable${RESET}"
        echo -e "     ${DIM}Then restart Chrome${RESET}"
        echo ""
        echo -e "  ${BOLD}3. Or try Firefox (often has better WebGL on Linux):${RESET}"
        echo -e "     ${DIM}firefox http://localhost:8487${RESET}"
        echo ""
        echo -e "  ${BOLD}4. Or access from another device on your network:${RESET}"
        echo -e "     ${DIM}second-brain binds to 0.0.0.0 — open http://$(hostname -I 2>/dev/null | awk '{print $1}'):8487${RESET}"
        echo -e "     ${DIM}from your phone, tablet, or another computer${RESET}"
        echo ""
    else
        ok "WebGL should work"
    fi
fi

# ----------------------------------------------------------
# 9. Done
# ----------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}  Installation complete!${RESET}"
echo ""

if [ -n "$SHELL_RC" ] && ! command -v second-brain &>/dev/null; then
    echo -e "  ${YELLOW}Restart your terminal (or run 'source $SHELL_RC') for PATH changes${RESET}"
    echo ""
fi

echo -e "  Start the 3D brain:"
echo -e "    ${BOLD}second-brain${RESET}"
echo -e "    ${DIM}opens http://localhost:8487${RESET}"
echo -e "    ${DIM}auto-detects your Ollama models${RESET}"
echo ""
echo -e "  Python API:"
echo -e "    ${DIM}from semantic_gravity_memory import Memory${RESET}"
echo -e "    ${DIM}m = Memory()${RESET}"
echo -e "    ${DIM}m.ingest('hello world')${RESET}"
echo ""

# Offer to launch
if command -v second-brain &>/dev/null || [ -n "$BRAIN_BIN" ]; then
    echo -e -n "  Launch now? [Y/n] "
    read -r LAUNCH
    if [ "$LAUNCH" != "n" ] && [ "$LAUNCH" != "N" ]; then
        echo ""
        exec "${BRAIN_BIN:-second-brain}"
    fi
fi
