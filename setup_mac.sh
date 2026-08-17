#!/bin/bash

sed -i '' 's/\r//' "$0" 2>/dev/null || true

echo "--- Starting Automatic Configuration (macOS) ---"


if ! command -v python3 &>/dev/null; then
  echo "[ERROR] Python 3 is not installed. Install Python from python.org or via Homebrew."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Creating virtual environment (venv)..."
  python3 -m venv venv
fi

source venv/bin/activate


if [ -f "install_deps.py" ]; then
  python3 install_deps.py
else
  echo "[ERROR] File install_deps.py not found in the current folder!"
fi

echo "--- Procedure completed ---"
echo "To start the program in the future, use:"
echo "source venv/bin/activate && python3 main.py"
