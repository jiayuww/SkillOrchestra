"""
Pipeline paths and default model configuration.

Used by scripts/pipeline.py for model-routing and FRAMES pipelines.
"""

from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_REPO_DIR = _CONFIG_DIR.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = _REPO_DIR / "data"
CONFIGS_DIR = _REPO_DIR / "config"
DEFAULT_OUTPUT_DIR = _REPO_DIR / "output"

# FRAMES: exploration data and samples
FRAMES_EXPLORATION_DIR = DATA_DIR
FRAMES_SAMPLES_PATH = FRAMES_EXPLORATION_DIR / "frames_train.jsonl"
FRAMES_TEST_PATH = DATA_DIR / "frames_test.jsonl"
DEFAULT_MODEL_CONFIG = CONFIGS_DIR / "eval_config.json"
DEFAULT_EVAL_SCRIPT = _REPO_DIR / "orchestration" / "eval_frames.py"

# Model routing: RSL results (when using existing exploration)
RSL_RESULTS_DIR = DATA_DIR / "rsl_results"

# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

# Pool models for model routing
DEFAULT_POOL_MODELS = [
    "qwen2.5-7b-instruct",
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "mistral-7b-instruct",
    "mixtral-8x22b-instruct",
    "gemma2-27b-it",
]

# Models for FRAMES exploration and routing (must match eval_frames ALL_TOOLS)
DEFAULT_SEARCH_MODELS = ["search-1", "search-2", "search-3"]
DEFAULT_CODE_MODELS = ["reasoner-1", "reasoner-2", "reasoner-3"]
DEFAULT_ANSWER_MODELS = [
    "answer-1",
    "answer-2",
    "answer-3",
    "answer-4",
    "answer-math-1",
    "answer-math-2",
]

# FRAMES model routing pool (models the orchestrator routes between per stage)
DEFAULT_FRAMES_POOL_MODELS = (
    DEFAULT_SEARCH_MODELS + DEFAULT_CODE_MODELS + DEFAULT_ANSWER_MODELS
)
