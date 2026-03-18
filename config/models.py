# Re-export from pool config (single source of truth)
from .pool import POOL_MODEL_DISPLAY_NAMES, display_name as pool_display_name

# agent_id -> actual model path/endpoint
MODEL_MAPPING = {
    "search-1": "gpt-5",
    "search-2": "gpt-5-mini",
    "search-3": "Qwen/Qwen3-32B",
    "reasoner-1": "gpt-5",
    "reasoner-2": "gpt-5-mini",
    "reasoner-3": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "answer-math-1": "Qwen/Qwen2.5-Math-72B-Instruct",
    "answer-math-2": "Qwen/Qwen2.5-Math-7B-Instruct",
    "answer-1": "gpt-5",
    "answer-2": "gpt-5-mini",
    "answer-3": "meta-llama/Llama-3.3-70B-Instruct",
    "answer-4": "Qwen/Qwen3-32B",
}


def resolve_model(agent_id: str) -> str:
    """Resolve agent_id to actual model endpoint.
    If agent_id is not in MODEL_MAPPING, returns agent_id unchanged (e.g. already a model path/endpoint).
    """
    return MODEL_MAPPING.get(agent_id, agent_id)