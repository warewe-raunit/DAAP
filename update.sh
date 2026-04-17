#!/bin/bash
# Run this to get latest DAAP updates
# Usage: bash update.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[daap] Pulling latest changes..."
git pull

echo "[daap] Installing any new dependencies..."
if [ -d ~/.daap-venv ]; then
    ~/.daap-venv/bin/pip install -q -r requirements.txt
else
    echo "[daap] No venv found at ~/.daap-venv — run setup first."
    exit 1
fi

echo "[daap] Done. Run: source ~/.daap-venv/bin/activate && python3 scripts/chat.py --api-url http://107.174.35.26:8000"
