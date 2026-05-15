#!/usr/bin/env bash
# Provision a separate venv with Marker-Inc AutoRAG installed.
#
# AutoRAG 0.3.x pins ``numpy<2`` which conflicts with the bench's base deps
# (opencv-python-headless>=4.13 needs numpy>=2). We isolate it in a sibling
# .autorag-venv/, which the bench driver auto-discovers — no env var needed.
# Only set AUTORAG_PYTHON if you've placed the venv somewhere else.
#
# Usage:
#   bash scripts/setup_autorag_venv.sh

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

echo "installing AutoRAG (with [gpu] extras: sentence-transformers, FlagEmbedding,"
echo "  torch — required by the paper's HF embedders & rerankers)"
"$VENV_PIP" install --quiet "AutoRAG[gpu]>=0.3,<0.4"

# AutoRAG 0.3 still pulls in the deprecated llama-index-llms-bedrock package,
# which restricts ``model`` to a fixed pre-2024 registry (no Llama 3.1, Nova 2,
# Claude Haiku 4.5). The paper's bedrock entries are all 2024+, so we add the
# modern bedrock-converse LLM and register it as a new AutoRAG provider via
# the .pth patch (see autorag_patches.py:_patch_register_bedrock_converse).
echo "installing llama-index-llms-bedrock-converse (modern Bedrock API for newer models)"
"$VENV_PIP" install --quiet "llama-index-llms-bedrock-converse>=0.4"

echo "verifying autorag CLI"
# AutoRAG installs an entry-point ``autorag`` script next to python — ``python
# -m autorag`` doesn't work because the package has no __main__.
if ! "$VENV_DIR/bin/autorag" --help >/dev/null 2>&1; then
    echo "ERROR: '$VENV_DIR/bin/autorag --help' failed" >&2
    echo "Try a different base Python: PYTHON_BIN=python3.11 bash scripts/setup_autorag_venv.sh" >&2
    exit 1
fi

echo "installing bench's AutoRAG runtime patches (.pth-driven)"
# ``Chroma.add_embedding`` doesn't chunk, so corpora >5461 docs blow up on
# Chroma's SQLite batch cap. The patch is in scripts/autorag_patches.py;
# we drop it into site-packages plus a .pth line so every interpreter
# invocation (including the ``autorag`` CLI subprocess) picks it up.
SITE_PACKAGES="$("$VENV_PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
cp "$(dirname "$0")/autorag_patches.py" "$SITE_PACKAGES/bench_autorag_patches.py"
echo "import bench_autorag_patches" > "$SITE_PACKAGES/bench_autorag_patches.pth"

echo
echo "AutoRAG installed successfully at $VENV_DIR/."
echo "The bench's matrix runner auto-discovers this location — no export needed."
echo "Run e.g.: uv run agentic-autorag-bench run --config configs/hotpot_paper.yaml -m autorag_ragas"
echo
echo "Only set AUTORAG_PYTHON if you've moved the venv elsewhere, or you want to"
echo "exercise the AUTORAG_PYTHON-gated equivalence tests (tests/test_autorag_equivalence.py)."
echo
