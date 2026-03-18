#!/bin/bash
# =============================================================================
# Model Routing - Sequential Launcher (1 H200 node, 8 GPUs)
# =============================================================================
#
# Usage:
#   ./scripts/serve/serve_routing.sh           # Default: all models
#   ./scripts/serve/serve_routing.sh --minimal # 5 models (skip large)
#   ./scripts/serve/serve_routing.sh --all     # All models
#   ./scripts/serve/serve_routing.sh --stop
#   ./scripts/serve/serve_routing.sh --status
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sglang_env

[ -f "$SO_ROOT/scripts/env.sh" ] && source "$SO_ROOT/scripts/env.sh"
[ -f "$SO_ROOT/.env" ] && set -a && source "$SO_ROOT/.env" 2>/dev/null && set +a
export HF_HOME="${HF_HOME:-/export/xgen-small/models}"
export PATH="/opt/conda/envs/sglang_env/bin:$PATH"

LOG_DIR="$SO_ROOT/logs/sglang"
mkdir -p "$LOG_DIR"

HEALTH_TIMEOUT=300
HEALTH_POLL=5

# Model configs: model_key|model_path|port|tp_size|mem_fraction
declare -A MODELS=(
    ["qwen"]="Qwen/Qwen2.5-7B-Instruct|30006|1|0.25"
    ["llama8b"]="meta-llama/Llama-3.1-8B-Instruct|30001|1|0.5"
    ["mistral"]="mistralai/Mistral-7B-Instruct-v0.3|30003|1|0.75"
    ["gemma"]="google/gemma-2-27b-it|30005|1|0.55"
    ["qwen3b"]="Qwen/Qwen2.5-3B-Instruct|30008|1|0.5"
    ["llama70b"]="meta-llama/Llama-3.1-70B-Instruct|30002|2|0.9"
    ["mixtral"]="mistralai/Mixtral-8x22B-Instruct-v0.1|30004|4|0.9"
)

# Original launch order and GPU assignment (do not change)
LAUNCH_ORDER=(
    "qwen|0"
    "llama8b|0"
    "mistral|0"
    "gemma|1"
    "llama70b|4,5"
    "mixtral|2,3,6,7"
    "qwen3b|1"
)

# Minimal: skip large models (llama70b, mixtral)
LAUNCH_ORDER_SKIP_LARGE=(
    "qwen|0"
    "llama8b|1"
    "mistral|2"
    "gemma|3"
    "qwen3b|4"
)

wait_for_health() {
    local port=$1
    local label=$2
    local elapsed=0

    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ $label ready (${elapsed}s)"
            return 0
        fi
        sleep $HEALTH_POLL
        elapsed=$((elapsed + HEALTH_POLL))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... waiting (${elapsed}s)"
        fi
    done

    echo "  ✗ $label TIMEOUT after ${HEALTH_TIMEOUT}s"
    echo "  Check log: tail -f $LOG_DIR/${label}.log"
    return 1
}

launch_and_wait() {
    local model_key=$1
    local gpus=$2

    IFS='|' read -r model_path port tp_size mem_fraction <<< "${MODELS[$model_key]}"

    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "[$model_key] Already running on port $port, skipping"
        echo ""
        return 0
    fi

    local gpu_count=$(echo "$gpus" | tr ',' '\n' | wc -l)
    [ $gpu_count -gt 1 ] && tp_size=$gpu_count

    local log_file="$LOG_DIR/${model_key}.log"

    echo "[$model_key] Launching..."
    echo "  Model: $model_path"
    echo "  Port: $port | GPUs: $gpus (tp=$tp_size) | Mem: $mem_fraction"

    CUDA_VISIBLE_DEVICES=$gpus nohup python -m sglang.launch_server \
        --model-path "$model_path" \
        --port $port \
        --host 0.0.0.0 \
        --tp $tp_size \
        --mem-fraction-static $mem_fraction \
        --trust-remote-code \
        > "$log_file" 2>&1 &

    echo "  PID: $!"
    wait_for_health $port $model_key
    local status=$?
    echo ""
    return $status
}

show_status() {
    echo "=========================================="
    echo "Server Status"
    echo "=========================================="
    local running=0
    local total=0
    for entry in "${LAUNCH_ORDER[@]}"; do
        IFS='|' read -r model_key gpus <<< "$entry"
        IFS='|' read -r _ port _ _ <<< "${MODELS[$model_key]}"
        total=$((total + 1))
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ $model_key (port $port)"
            running=$((running + 1))
        else
            echo "  ✗ $model_key (port $port)"
        fi
    done
    echo ""
    echo "$running/$total servers running"
    echo "=========================================="
}

stop_all() {
    echo "Stopping all sglang servers..."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
    echo "Done."
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
    --stop)
        stop_all
        ;;
    --status)
        show_status
        ;;
    --minimal)
        echo "=========================================="
        echo "Minimal Launch (skip large)"
        echo "=========================================="
        echo "  qwen|0, llama8b|1, mistral|2, gemma|3, qwen3b|4"
        echo ""
        failed=0
        for entry in "${LAUNCH_ORDER_SKIP_LARGE[@]}"; do
            IFS='|' read -r model_key gpus <<< "$entry"
            launch_and_wait "$model_key" "$gpus" || failed=$((failed + 1))
        done
        echo "=========================================="
        echo "Launch complete. Failures: $failed"
        show_status
        ;;
    --all)
        echo "=========================================="
        echo "Sequential Launch (all models)"
        echo "=========================================="
        echo "  GPU 0: qwen, llama8b, mistral"
        echo "  GPU 1: gemma, qwen3b"
        echo "  GPU 4,5: llama70b"
        echo "  GPU 2,3,6,7: mixtral"
        echo ""
        failed=0
        for entry in "${LAUNCH_ORDER[@]}"; do
            IFS='|' read -r model_key gpus <<< "$entry"
            launch_and_wait "$model_key" "$gpus" || failed=$((failed + 1))
        done
        echo "=========================================="
        echo "Launch complete. Failures: $failed"
        show_status
        ;;
    *)
        # Default: all models
        echo "=========================================="
        echo "Sequential Launch (all models)"
        echo "=========================================="
        echo "  GPU 0: qwen, llama8b, mistral"
        echo "  GPU 1: gemma, qwen3b"
        echo "  GPU 4,5: llama70b"
        echo "  GPU 2,3,6,7: mixtral"
        echo ""
        failed=0
        for entry in "${LAUNCH_ORDER[@]}"; do
            IFS='|' read -r model_key gpus <<< "$entry"
            launch_and_wait "$model_key" "$gpus" || failed=$((failed + 1))
        done
        echo "=========================================="
        echo "Launch complete. Failures: $failed"
        show_status
        ;;
esac
