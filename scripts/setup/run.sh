#!/bin/bash
# =============================================================================
# Environment Setup - Unified Entry Point
# =============================================================================
#
# Usage:
#   ./scripts/setup/run.sh              # Run all (so_env, sglang_env, retriever)
#   ./scripts/setup/run.sh --env        # Pipeline + SGLang only
#   ./scripts/setup/run.sh --sglang     # SGLang only
#   ./scripts/setup/run.sh --retriever  # Retriever only
#   ./scripts/setup/run.sh --orchestration [--setup-envs]  # Agent orchestration
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:---all}" in
    --env)
        "$SCRIPT_DIR/env.sh" --all
        ;;
    --sglang)
        "$SCRIPT_DIR/env.sh" --sglang
        ;;
    --retriever)
        "$SCRIPT_DIR/retriever.sh"
        ;;
    --orchestration)
        shift
        "$SCRIPT_DIR/agent_orchestration.sh" "$@"
        ;;
    --all|*)
        echo "Setting up all environments..."
        echo ""
        "$SCRIPT_DIR/env.sh" --all
        echo ""
        echo "=========================================="
        echo ""
        "$SCRIPT_DIR/retriever.sh"
        echo ""
        echo "=========================================="
        echo "All environments ready: so_env, sglang_env, retriever"
        echo "=========================================="
        ;;
esac
