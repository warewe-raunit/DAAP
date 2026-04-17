#!/bin/bash
# First-time setup for DAAP CLI
# Usage: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[daap] Setting up virtual environment..."
python3 -m venv ~/.daap-venv

echo "[daap] Installing dependencies..."
~/.daap-venv/bin/pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Make the daap wrapper executable
chmod +x "$SCRIPT_DIR/scripts/daap"

# Install daap command to PATH
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"
ln -sf "$SCRIPT_DIR/scripts/daap" "$INSTALL_DIR/daap"

# Ensure ~/.local/bin is on PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "[daap] Adding ~/.local/bin to PATH..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi

# Save server URL
mkdir -p ~/.daap
SERVER_FILE="$HOME/.daap/server"
if [ ! -f "$SERVER_FILE" ]; then
    echo ""
    read -p "[daap] Server URL (e.g. http://107.174.35.26:8000): " SERVER_URL
    if [ -n "$SERVER_URL" ]; then
        echo "$SERVER_URL" > "$SERVER_FILE"
        echo "[daap] Server saved to $SERVER_FILE"
    fi
else
    echo "[daap] Server: $(cat $SERVER_FILE)"
fi

# Save API key hint
if [ -z "$DAAP_API_KEY" ]; then
    echo ""
    read -p "[daap] DAAP_API_KEY (leave blank to set later): " INPUT_KEY
    if [ -n "$INPUT_KEY" ]; then
        # Add to ~/.bashrc if not already there
        if ! grep -q "DAAP_API_KEY" ~/.bashrc; then
            echo "export DAAP_API_KEY=\"$INPUT_KEY\"" >> ~/.bashrc
            echo "[daap] Key saved to ~/.bashrc"
        fi
        export DAAP_API_KEY="$INPUT_KEY"
    fi
else
    echo "[daap] DAAP_API_KEY already set."
fi

echo ""
echo "[daap] Setup complete!"
echo ""
echo "Start chatting:"
echo "  source ~/.bashrc   # load PATH + API key (first time only)"
echo "  daap               # new session"
echo "  daap <session_id>  # resume session"
echo ""
echo "To update in future:"
echo "  bash update.sh"
