"""
SGLang pool model configuration.

Loads from config/pool_config.json. Host/port can be overridden via:
- SGLANG_DEFAULT_HOST (fallback for all models)
- Per-model env vars: SGLANG_QWEN7B_HOST, SGLANG_LLAMA70B_HOST, etc.
- load_distributed_config(path) for JSON overrides
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, List

_CONFIG_DIR = Path(__file__).resolve().parent
_POOL_CONFIG_PATH = _CONFIG_DIR / "pool_config.json"


def _get_primary_ip() -> str:
    """Get this machine's primary IP (for outbound connections)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "localhost"


def _load_pool_config() -> Dict[str, Any]:
    """Load pool config from JSON."""
    with open(_POOL_CONFIG_PATH) as f:
        return json.load(f)


def _build_model_configs(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build MODEL_CONFIGS with env var, config file, and auto-detect resolution."""
    default_host = (
        os.environ.get("SGLANG_DEFAULT_HOST")
        or raw.get("default_host")
        or _get_primary_ip()
    )
    configs: Dict[str, Dict[str, Any]] = {}

    for model_key, m in raw["models"].items():
        env_var = m.get("env_host")
        ip_addr = os.environ.get(env_var, default_host) if env_var else default_host
        configs[model_key] = {
            "model_path": m["model_path"],
            "ip_addr": ip_addr,
            "port": m["port"],
            "display_name": m["display_name"],
        }
    return configs


# Load once at import
_RAW = _load_pool_config()
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = _build_model_configs(_RAW)
POOL_MODEL_KEYS: List[str] = _RAW["pool_model_keys"]
API_PRICE_1M_TOKENS: Dict[str, Dict[str, float]] = _RAW["pricing"]
POOL_PROMPT: str = _RAW["default_prompt"]

# Display name → model key mapping (case-insensitive)
_DISPLAY_NAME_MAP: Dict[str, str] = {}
for _k, _v in MODEL_CONFIGS.items():
    _DISPLAY_NAME_MAP[_v["display_name"].lower()] = _k
    _DISPLAY_NAME_MAP[_k.lower()] = _k

# Backward compat: POOL_MODEL_DISPLAY_NAMES for config.models consumers
POOL_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    k: v["display_name"] for k, v in MODEL_CONFIGS.items()
}

DEFAULT_HOST = next(
    (c["ip_addr"] for c in MODEL_CONFIGS.values() if c.get("ip_addr")),
    "localhost",
)


def display_name(model_key: str) -> str:
    """Return the human-readable display name for a model key."""
    cfg = MODEL_CONFIGS.get(model_key)
    return cfg["display_name"] if cfg else model_key


def load_distributed_config(path: str) -> None:
    """Load host/port overrides from a JSON file (same format as agentic-router)."""
    with open(path) as f:
        cfg = json.load(f)
    for model_key, settings in cfg.items():
        if model_key in MODEL_CONFIGS:
            if "ip_addr" in settings:
                MODEL_CONFIGS[model_key]["ip_addr"] = settings["ip_addr"]
            if "port" in settings:
                MODEL_CONFIGS[model_key]["port"] = settings["port"]
