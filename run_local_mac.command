#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "Starting SignalForge..."
echo

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python was not found."
  echo "Install Python 3, then run this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if ! "$PYTHON_BIN" -m uvicorn --version >/dev/null 2>&1; then
  echo "Uvicorn is not installed for $PYTHON_BIN."
  echo "Installing app requirements now..."
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo
echo "Open this URL in your browser:"
echo "http://127.0.0.1:8000"
echo
echo "Leave this window open while using SignalForge."
echo "Press Control+C to stop the app."
echo

"$PYTHON_BIN" -m uvicorn app.main:app --reload

echo
read -r -p "SignalForge stopped. Press Enter to close..."
