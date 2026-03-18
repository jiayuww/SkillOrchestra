#!/bin/bash
# =============================================================================
# Setup Retriever Environment (for retrieval_wiki.py / retrieval_hle.py)
# =============================================================================
#
# Creates the 'retriever' conda env with all deps for Qwen3-Embedding-8B
# and FAISS-based retrieval.
#
# Usage:
#   ./scripts/setup/retriever.sh
#
# After setup:
#   The retriever is launched by serve_orchestration.sh (uses conda env retriever):
#   ./scripts/serve/serve_orchestration.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="retriever"

echo "=========================================="
echo "Setting up Retriever Environment"
echo "=========================================="

# Remove existing env if present
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Removing existing $ENV_NAME env..."
    conda env remove -n $ENV_NAME -y
fi

# 1. Create env with Python 3.12
echo ""
echo "Creating conda env: $ENV_NAME (python 3.12)"
conda create -n $ENV_NAME python=3.12 -y

# 2. PyTorch 2.4 + CUDA 12.1
echo ""
echo "Installing PyTorch 2.4.0 + CUDA 12.1..."
conda install -n $ENV_NAME pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. FAISS-GPU (for index search)
echo ""
echo "Installing faiss-gpu..."
conda install -n $ENV_NAME -c pytorch -c nvidia faiss-gpu -y

# 4. Pip packages (skip pyserini - causes numpy/transformers conflicts)
echo ""
echo "Installing pip packages..."
conda run -n $ENV_NAME pip install \
    "transformers>=4.51" \
    "numpy<2" \
    "huggingface_hub>=0.23.0" \
    datasets \
    uvicorn \
    fastapi \
    tavily-python

# 5. Flash Attention (required for Qwen3-Embedding-8B)
echo ""
echo "Installing flash-attn..."
conda run -n $ENV_NAME pip install flash-attn --no-build-isolation

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Activate: conda activate $ENV_NAME"
echo ""
echo "Launch retriever (via orchestration):"
echo "  ./scripts/serve/serve_orchestration.sh       # Launches retriever + all orchestration models"
echo "  # Set RETRIEVAL_BACKEND=wiki in scripts/env.sh for FRAMES"
echo ""
echo "=========================================="
