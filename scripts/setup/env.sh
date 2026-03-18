#!/bin/bash
# =============================================================================
# Setup SkillOrchestra Environment
# =============================================================================
#
# Creates two conda environments:
#   1. so_env      - For running the SkillOrchestra pipeline (learning, selection)
#   2. sglang_env  - For serving models via SGLang, can skip it if not calling models via SGLang
#
# Usage:
#   ./scripts/setup/env.sh              # Setup both environments
#   ./scripts/setup/env.sh --pipeline   # Pipeline env only
#   ./scripts/setup/env.sh --sglang     # SGLang env only
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source $(conda info --base)/etc/profile.d/conda.sh

setup_pipeline_env() {
    local ENV_NAME="so_env"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '$ENV_NAME' already exists. Updating..."
        conda activate $ENV_NAME
    else
        echo "Creating environment '$ENV_NAME'..."
        conda create -n $ENV_NAME python=3.11 -y
        conda activate $ENV_NAME
    fi

    echo "Installing pipeline dependencies..."
    pip install -q \
        openai>=1.0.0 \
        pydantic>=2.0.0 \
        requests \
        python-dotenv \
        tqdm \
        sympy \
        pylatexenc \
        datasets

    echo ""
    echo "Pipeline environment '$ENV_NAME' ready."
    echo "  Activate: conda activate $ENV_NAME"
    echo "  Run:      python scripts/pipeline.py --help"
}

setup_sglang_env() {
    local ENV_NAME="sglang_env"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '$ENV_NAME' already exists."
        echo "  To update: conda activate $ENV_NAME && pip install -U 'sglang[all]'"
        return 0
    fi

    echo "Creating environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.11 -y
    conda activate $ENV_NAME

    echo "Installing SGLang..."
    pip install 'sglang[all]'

    # Verify installation
    python -c "import sglang; print(f'SGLang version: {sglang.__version__}')" 2>/dev/null || \
        echo "WARNING: SGLang import check failed, but install may still be ok"

    echo ""
    echo "SGLang environment '$ENV_NAME' ready."
    echo "  Activate: conda activate $ENV_NAME"
    echo "  Launch:   ./scripts/serve/serve_routing.sh --help"
}

case "${1:---all}" in
    --pipeline)
        setup_pipeline_env
        ;;
    --sglang)
        setup_sglang_env
        ;;
    --all|*)
        setup_pipeline_env
        echo ""
        echo "=========================================="
        echo ""
        setup_sglang_env
        echo ""
        echo "=========================================="
        echo "Setup complete!"
        echo ""
        echo "Quick start:"
        echo "  # 1. Start model servers"
        echo "  conda activate sglang_env"
        echo "  ./scripts/serve/serve_routing.sh"
        echo ""
        echo "  # 2. Wait for servers"
        echo "  conda activate so_env"
        echo "  python scripts/check_servers.py --mode routing --wait"
        echo ""
        echo "  # 3. Run full pipeline"
        echo "  python scripts/pipeline.py model-routing \\"
        echo "      --dataset nq_validation_qwen \\"
        echo "      --output-dir /tmp/so_pipeline/nq \\"
        echo "      --phases explore,learn,select,test \\"
        echo "      --test-dataset nq_test_qwen"
        echo "=========================================="
        ;;
esac
