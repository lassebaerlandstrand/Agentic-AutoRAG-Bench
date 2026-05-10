#!/usr/bin/env bash
# Provision a separate venv with Marker-Inc AutoRAG installed.
#
# AutoRAG 0.3.x pins ``numpy<2`` which conflicts with the bench's base deps
# (opencv-python-headless>=4.13 needs numpy>=2). We isolate it in a sibling
# venv and pass the interpreter path via AUTORAG_PYTHON to the bench's
# autorag method driver.
#
# Usage:
#   bash scripts/setup_autorag_venv.sh
#   export AUTORAG_PYTHON="$(pwd)/.autorag-venv/bin/python"

set -euo pipefail

VENV_DIR=".autorag-venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d "$VENV_DIR" ]]; then
    echo "venv already exists at $VENV_DIR — skipping creation"
else
    echo "creating venv at $VENV_DIR (using $PYTHON_BIN)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "upgrading pip"
"$VENV_PIP" install --quiet --upgrade pip

echo "installing AutoRAG"
"$VENV_PIP" install --quiet "AutoRAG>=0.3,<0.4"

echo "verifying autorag CLI"
if ! "$VENV_PYTHON" -m autorag --help >/dev/null 2>&1; then
    echo "ERROR: 'autorag --help' failed inside $VENV_DIR" >&2
    echo "Try a different base Python: PYTHON_BIN=python3.11 bash scripts/setup_autorag_venv.sh" >&2
    exit 1
fi

ABS_PYTHON="$(cd "$(dirname "$VENV_PYTHON")/.." && pwd)/bin/python"
echo
echo "AutoRAG installed successfully."
echo "Add this to your shell environment before running bench tasks:"
echo
echo "  export AUTORAG_PYTHON=\"$ABS_PYTHON\""
echo
