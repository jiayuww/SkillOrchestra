#!/bin/bash
# =============================================================================
# Orchestration - Sequential Launcher (1 H200 node, 8 GPUs)
# =============================================================================
#
# Launches orchestration models in order.
#
# Usage:
#   ./scripts/serve/serve_orchestration.sh      # Default: all models
#   ./scripts/serve/serve_orchestration.sh --stop
#   ./scripts/serve/serve_orchestration.sh --status
#
# Order
#   math72b 0,1 | llama3_3_70b 2,3 | qwen32b 4 | coder32b 5
#   retriever 6 | nemotron_8b 6 | qwen3_8b 7 | math7b 7
#
# Note: retriever uses conda env 'retriever' (not sglang_env).
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

export EVAL_SCRIPTS_DIR="${EVAL_SCRIPTS_DIR:-$SO_ROOT/scripts/retrieval}"

LOG_DIR="$SO_ROOT/logs"
mkdir -p "$LOG_DIR"

HEALTH_TIMEOUT=300
HEALTH_POLL=5

# Model configs: model_key|model_path|port|tp_size|mem_fraction|extra_args
# retriever is special (uses retrieval_wiki.py)
declare -A MODELS=(
    ["math72b"]="Qwen/Qwen2.5-Math-72B-Instruct|1402|2|0.9|"
    ["llama3_3_70b"]="meta-llama/Llama-3.3-70B-Instruct|1405|2|0.9|"
    ["qwen32b"]="Qwen/Qwen3-32B|1403|1|0.9|"
    ["coder32b"]="Qwen/Qwen2.5-Coder-32B-Instruct|1407|1|0.9|"
    ["retriever"]="RETRIEVER|1401|1|0.4|"
    ["qwen3_8b"]="Qwen/Qwen3-8B|1408|1|0.8|--tool-call-parser qwen25"
    ["math7b"]="Qwen/Qwen2.5-Math-7B-Instruct|1404|1|0.8|"
)


# Launch order (do not change)
LAUNCH_ORDER=(
    "math72b|0,1"
    "llama3_3_70b|2,3"
    "qwen32b|4"
    "coder32b|5"
    "retriever|6"
    "nemotron_8b|6"
    "qwen3_8b|7"
    "math7b|7"
)

wait_for_health() {
    local port=$1
    local label=$2
    local elapsed=0

    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1 || \
           curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1 || \
           curl -sf "http://localhost:$port/" > /dev/null 2>&1; then
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

launch_retriever() {
    # Uses conda env 'retriever' (not sglang_env)
    local gpu=$1
    local backend="${RETRIEVAL_BACKEND:-wiki}"
    local log_file="$LOG_DIR/retriever.log"
    local eval_dir="${EVAL_SCRIPTS_DIR:-}"
    local cache_dir="${RETRIEVER_CACHE_DIR:-$eval_dir/cache/v1/$backend}"
    local script="retrieval_${backend}.py"

    if [ -z "$eval_dir" ] || [ ! -d "$eval_dir" ]; then
        echo "  ERROR: EVAL_SCRIPTS_DIR not set or missing: $eval_dir"
        return 1
    fi
    if [ ! -f "$eval_dir/$script" ]; then
        echo "  ERROR: $script not found at $eval_dir"
        return 1
    fi
    if [ -z "$INDEX_DIR" ] || [ ! -d "$INDEX_DIR" ]; then
        echo "  ERROR: INDEX_DIR not set or missing. Set in env.sh"
        return 1
    fi

    echo "[retriever] Launching ($backend)..."
    echo "  Port: 1401 | GPU: $gpu"

    if [ "$backend" = "hle" ]; then
        CUDA_VISIBLE_DEVICES=$gpu nohup conda run -n retriever python "$eval_dir/$script" \
            --port 1401 \
            --new_cache_dir "$cache_dir" \
            --example_id_file "${eval_dir}/examples.json" \
            --tavily_key "${TAVILY_KEY:-}" \
            > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$gpu nohup conda run -n retriever python "$eval_dir/$script" \
            --port 1401 \
            ${RETRIEVAL_NO_FAISS_GPU:+--no-faiss-gpu} \
            > "$log_file" 2>&1 &
    fi

    echo "  PID: $!"
    wait_for_health 1401 "retriever"
    local status=$?
    echo ""
    return $status
}

launch_and_wait() {
    local model_key=$1
    local gpus=$2

    if [[ "$model_key" == "retriever" ]]; then
        launch_retriever "$gpus"
        return $?
    fi

    IFS='|' read -r model_path port tp_size mem_fraction extra_args <<< "${MODELS[$model_key]}"

    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1 || \
       curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1; then
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
        $extra_args \
        > "$log_file" 2>&1 &

    echo "  PID: $!"
    wait_for_health $port $model_key
    local status=$?
    echo ""
    return $status
}

show_status() {
    echo "=========================================="
    echo "Orchestration Server Status"
    echo "=========================================="
    local running=0
    local total=0
    for entry in "${LAUNCH_ORDER[@]}"; do
        IFS='|' read -r model_key gpus <<< "$entry"
        IFS='|' read -r _ port _ _ _ <<< "${MODELS[$model_key]}"
        [[ "$model_key" == "retriever" ]] && port=1401
        total=$((total + 1))
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1 || \
           curl -sf "http://localhost:$port/v1/models" > /dev/null 2>&1 || \
           curl -sf "http://localhost:$port/" > /dev/null 2>&1; then
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
    echo "Stopping orchestration servers..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "retrieval_wiki.py" 2>/dev/null || true
    pkill -f "retrieval_hle.py" 2>/dev/null || true
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
    *)
        echo "=========================================="
        echo "Orchestration Sequential Launch"
        echo "=========================================="
        echo "  math72b 0,1 | llama3_3_70b 2,3 | qwen32b 4 | coder32b 5"
        echo "  retriever 6 | nemotron_8b 6 | qwen3_8b 7 | math7b 7"
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
