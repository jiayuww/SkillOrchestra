"""Shared configuration for model routing scripts."""

from typing import Dict

from config.pool import (
    API_PRICE_1M_TOKENS,
    MODEL_CONFIGS,
    POOL_MODEL_KEYS,
    display_name,
)

MAX_TURNS = 4

# Model key -> display name (derived from pool config)
MODEL_KEY_TO_DISPLAY: Dict[str, str] = {
    k: v["display_name"] for k, v in MODEL_CONFIGS.items()
}
DISPLAY_TO_KEY = {v: k for k, v in MODEL_KEY_TO_DISPLAY.items()}

# Pool models for exploration (same as POOL_MODEL_KEYS, excluding router)
POOL_MODELS = list(POOL_MODEL_KEYS)

# Fallback relative costs for weighted_avg when handbook has no cost data
# (output price per 1M tokens, derived from pool config)
OUTPUT_PRICES: Dict[str, float] = {
    display_name(k): p.get("output", 0) for k, p in API_PRICE_1M_TOKENS.items()
    if k in MODEL_CONFIGS
}
_MAX_PRICE = max(OUTPUT_PRICES.values()) or 1.0
MODEL_RELATIVE_COST_FALLBACK = {m: p / _MAX_PRICE for m, p in OUTPUT_PRICES.items()}
