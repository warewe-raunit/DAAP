#!/bin/bash
# First-time setup for DAAP CLI
# Usage: bash setup.sh

set -e

echo "[daap] Setting up virtual environment..."
python3 -m venv ~/.daap-venv

echo "[daap] Installing dependencies..."
~/.daap-venv/bin/pip install -q -r requirements.txt

echo ""
echo "[daap] Setup complete!"
echo ""
echo "To start chatting:"
echo "  source ~/.daap-venv/bin/activate"
echo "  python3 scripts/chat.py --api-url http://107.174.35.26:8000"
echo ""
echo "To update in future:"
echo "  bash update.sh"
