#!/bin/bash
# =============================================================================
# Setup Agent Orchestration Environment (so-internal)
# =============================================================================
#
# Reproduces the full environment for agent evaluation (FRAMES, HLE) from
# so-internal, integrating with agentic-router/ToolOrchestra/evaluation.
#
# Usage:
#   ./scripts/setup/agent_orchestration.sh             # Full setup (verify only)
#   ./scripts/setup/agent_orchestration.sh --setup-envs  # Create conda envs (sglang_env, retriever)
#   ./scripts/setup/agent_orchestration.sh --env-only    # Just create env.sh
#   ./scripts/setup/agent_orchestration.sh --check      # Verify setup
#
# Prerequisites:
#   - INDEX_DIR set in .env or env.sh (index files: eval.index/eval.jsonl for HLE; wiki.index/wiki.jsonl for FRAMES)
#   - Retrieval scripts in scripts/retrieval/
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SETUP_DIR="$(cd "$SCRIPT_DIR" && pwd)"

# Load .env and env.sh early so INDEX_DIR etc. are available
[ -f "$SO_ROOT/.env" ] && set -a && source "$SO_ROOT/.env" 2>/dev/null && set +a
[ -f "$SO_ROOT/scripts/env.sh" ] && source "$SO_ROOT/scripts/env.sh"

EVAL_SCRIPTS_DIR="${EVAL_SCRIPTS_DIR:-$SO_ROOT/scripts/retrieval}"
INDEX_DIR="${INDEX_DIR:-}"

do_env_only=false
do_check=false
do_setup_envs=false
for arg in "$@"; do
    case "$arg" in
        --env-only)    do_env_only=true ;;
        --check)       do_check=true ;;
        --setup-envs)  do_setup_envs=true ;;
    esac
done

echo "=========================================="
echo "Agent Orchestration Setup (so-internal)"
echo "=========================================="
echo "  SO_ROOT:         $SO_ROOT"
echo "  EVAL_SCRIPTS_DIR: ${EVAL_SCRIPTS_DIR:-[not set]}"
echo "  INDEX_DIR:       ${INDEX_DIR:-[not set]}"
echo ""

# -----------------------------------------------------------------------------
# 1. Create env.sh from example if missing
# -----------------------------------------------------------------------------
if [ ! -f "$SO_ROOT/scripts/env.sh" ]; then
    echo "Creating scripts/env.sh from env.example.sh..."
    cp "$SO_ROOT/scripts/env.example.sh" "$SO_ROOT/scripts/env.sh"
    echo "  Done. Edit scripts/env.sh to customize paths and API keys."
else
    echo "scripts/env.sh already exists."
fi

# Re-load env after creating env.sh (in case it was just created)
[ -f "$SO_ROOT/scripts/env.sh" ] && source "$SO_ROOT/scripts/env.sh"
[ -f "$SO_ROOT/.env" ] && set -a && source "$SO_ROOT/.env" 2>/dev/null && set +a
export EVAL_SCRIPTS_DIR="${EVAL_SCRIPTS_DIR:-$SO_ROOT/scripts/retrieval}"
export INDEX_DIR="${INDEX_DIR:-}"

if [ "$do_env_only" = true ]; then
    echo ""
    echo "Done (--env-only). Run again without --env-only for full setup."
    exit 0
fi

# -----------------------------------------------------------------------------
# 2. Verify evaluation scripts (retrieval_wiki.py, retrieval_hle.py, examples.json)
# -----------------------------------------------------------------------------
echo ""
echo "Checking retrieval scripts..."

eval_dir="${EVAL_SCRIPTS_DIR:-$SO_ROOT/scripts/retrieval}"
if [ -z "$eval_dir" ] || [ ! -d "$eval_dir" ]; then
    echo "  ERROR: EVAL_SCRIPTS_DIR not set or missing: $eval_dir"
    echo "  Default: scripts/retrieval. Run setup to copy scripts from agentic-router if needed."
    exit 1
fi
for script in retrieval_hle.py retrieval_wiki.py examples.json; do
    if [ ! -f "$eval_dir/$script" ]; then
        echo "  ERROR: Missing $eval_dir/$script"
        exit 1
    fi
done
echo "  ✓ retrieval_hle.py, retrieval_wiki.py, examples.json found"

# -----------------------------------------------------------------------------
# 3. Verify index directory
# -----------------------------------------------------------------------------
echo ""
echo "Checking index directory..."

if [ -z "$INDEX_DIR" ] || [ ! -d "$INDEX_DIR" ]; then
    echo "  WARNING: INDEX_DIR not set or missing: $INDEX_DIR"
    echo "  HLE needs: eval.index, eval.jsonl"
    echo "  FRAMES needs: wiki.index, wiki.jsonl"
    echo "  Set INDEX_DIR in .env or scripts/env.sh"
else
    for f in eval.index eval.jsonl wiki.index wiki.jsonl; do
        if [ -f "$INDEX_DIR/$f" ]; then
            echo "  ✓ $INDEX_DIR/$f"
        else
            echo "  - $INDEX_DIR/$f (optional for some workflows)"
        fi
    done
fi

# -----------------------------------------------------------------------------
# 4. Conda environments (create if --setup-envs)
# -----------------------------------------------------------------------------
echo ""
echo "Conda environments..."

if ! command -v conda &>/dev/null; then
    echo "  WARNING: conda not found. Install Miniconda/Anaconda first."
else
    source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || true

    # Create sglang_env
    if conda info --envs 2>/dev/null | grep -q "^sglang_env "; then
        echo "  ✓ sglang_env (exists)"
    else
        if [ "$do_setup_envs" = true ]; then
            echo "  Creating sglang_env..."
            "$SETUP_DIR/env.sh" --sglang
            echo "  ✓ sglang_env created"
        else
            echo "  - sglang_env (run: ./scripts/setup/env.sh --sglang)"
        fi
    fi

    # Create retriever env
    if conda info --envs 2>/dev/null | grep -q "^retriever "; then
        echo "  ✓ retriever (exists)"
    else
        if [ "$do_setup_envs" = true ]; then
            echo "  Creating retriever env..."
            "$SETUP_DIR/retriever.sh"
            echo "  ✓ retriever created"
        else
            echo "  - retriever (run: ./scripts/setup/retriever.sh)"
            echo "    Or: ./scripts/setup/agent_orchestration.sh --setup-envs"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 5. Nemotron checkpoint (optional)
# -----------------------------------------------------------------------------
CKPT_DIR="${CKPT_DIR:-$EVAL_SCRIPTS_DIR/Nemotron-Orchestrator-8B}"
if [ -d "$CKPT_DIR" ] && [ -f "$CKPT_DIR/config.json" ]; then
    echo ""
    echo "  ✓ Nemotron-Orchestrator-8B at $CKPT_DIR"
else
    echo ""
    echo "  - Nemotron-Orchestrator-8B not found at $CKPT_DIR"
    echo "    Download or symlink for orchestrator model."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setup complete. Next steps:"
echo "=========================================="
echo ""
echo "  1. Edit .env or scripts/env.sh (INDEX_DIR, TAVILY_KEY, etc.)"
echo ""
echo "  2. Launch orchestration (FRAMES) servers (1 H200 node, 8 GPUs):"
echo "     ./scripts/serve/serve_orchestration.sh"
echo "     ./scripts/serve/serve_orchestration.sh --status"
echo "     ./scripts/serve/serve_orchestration.sh --stop"
echo ""
echo "  3. Launch model routing (separate node):"
echo "     ./scripts/serve/serve_routing.sh             # All 8 models"
echo "     ./scripts/serve/serve_routing.sh --minimal   # 5 models (skip large)"
echo ""
echo "  4. Run pipeline:"
echo "     python scripts/pipeline.py frames --model-config config/eval_config.json ..."
echo ""
echo "=========================================="
