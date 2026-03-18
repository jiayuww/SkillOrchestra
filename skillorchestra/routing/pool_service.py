"""
SGLang pool model service for SkillOrchestra.

Calls pool models via the OpenAI-compatible /v1/chat/completions endpoint,
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from config.pool import (
    API_PRICE_1M_TOKENS,
    DEFAULT_HOST,
    MODEL_CONFIGS,
    POOL_MODEL_KEYS,
    POOL_PROMPT,
    _DISPLAY_NAME_MAP,
    display_name,
    load_distributed_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PoolModelCost:
    """Cost breakdown for a single pool model call."""
    model_key: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class PoolCallResult:
    """Result from calling a pool model."""
    model_key: str
    response: str
    cost: PoolModelCost
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def resolve_model_key(name: str) -> str:
    """Map a display name / model key to the canonical model_key."""
    key = name.strip().lower()
    if key in _DISPLAY_NAME_MAP:
        return _DISPLAY_NAME_MAP[key]

    cleaned = key.replace("-instruct", "").replace("_instruct", "")
    if "qwen" in cleaned and "3b" not in cleaned:
        return "qwen2.5-7b-instruct"
    if "llama" in cleaned:
        return "llama3.1-70b-instruct" if "70b" in cleaned else "llama3.1-8b-instruct"
    if "mixtral" in cleaned:
        return "mixtral-8x22b-instruct"
    if "mistral" in cleaned:
        return "mistral-7b-instruct"
    if "gemma" in cleaned:
        return "gemma2-27b-it"
    return ""


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_server_health(model_key: str) -> bool:
    if model_key not in MODEL_CONFIGS:
        return False
    cfg = MODEL_CONFIGS[model_key]
    host = cfg.get("ip_addr", DEFAULT_HOST)
    try:
        r = requests.get(f"http://{host}:{cfg['port']}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def check_all_servers() -> Dict[str, bool]:
    return {k: check_server_health(k) for k in MODEL_CONFIGS}


# ---------------------------------------------------------------------------
# Model calling via /v1/chat/completions
# ---------------------------------------------------------------------------

def call_pool_model(
    model_key: str,
    query: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    timeout: int = 120,
    max_retries: int = 3,
    prompt_template: str = POOL_PROMPT,
) -> PoolCallResult:
    """Call a single pool model via /v1/chat/completions.

    The chat-completions endpoint applies the model's chat template
    server-side, avoiding the empty-response bug with raw /generate
    for models like Gemma-2.
    """
    if model_key not in MODEL_CONFIGS:
        return PoolCallResult(model_key=model_key, response="", cost=PoolModelCost(),
                              success=False, error=f"Unknown model: {model_key}")

    cfg = MODEL_CONFIGS[model_key]
    host = cfg.get("ip_addr", DEFAULT_HOST)
    url = f"http://{host}:{cfg['port']}/v1/chat/completions"

    user_content = prompt_template.format(query=query)
    payload: Dict[str, Any] = {
        "model": model_key,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9 if temperature > 0 else 1.0,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})
            pt = usage.get("prompt_tokens", len(user_content) // 4)
            ct = usage.get("completion_tokens", len(text) // 4)

            prices = API_PRICE_1M_TOKENS.get(model_key, {"input": 0, "output": 0})
            cost = PoolModelCost(
                model_key=model_key,
                prompt_tokens=pt,
                completion_tokens=ct,
                input_cost=pt * prices["input"] / 1_000_000,
                output_cost=ct * prices["output"] / 1_000_000,
            )
            return PoolCallResult(model_key=model_key, response=text, cost=cost)

        except requests.exceptions.Timeout:
            logger.warning(f"[{model_key}] timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return PoolCallResult(model_key=model_key, response="Request timed out",
                                      cost=PoolModelCost(model_key=model_key),
                                      success=False, error="timeout")

        except requests.exceptions.ConnectionError:
            logger.error(f"[{model_key}] connection error on port {cfg['port']}")
            return PoolCallResult(model_key=model_key, response="API Request Error",
                                  cost=PoolModelCost(model_key=model_key),
                                  success=False, error="connection_error")

        except Exception as e:
            logger.error(f"[{model_key}] error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return PoolCallResult(model_key=model_key, response=f"Error: {e}",
                                      cost=PoolModelCost(model_key=model_key),
                                      success=False, error=str(e))

    return PoolCallResult(model_key=model_key, response="API Request Error",
                          cost=PoolModelCost(model_key=model_key),
                          success=False, error="max_retries_exceeded")


def call_pool_models_parallel(
    model_keys: List[str],
    query: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    max_workers: int = 15,
    prompt_template: str = POOL_PROMPT,
) -> Dict[str, PoolCallResult]:
    """Call multiple pool models in parallel on the same query."""
    results: Dict[str, PoolCallResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_pool_model, mk, query,
                max_tokens=max_tokens, temperature=temperature,
                seed=seed, prompt_template=prompt_template,
            ): mk
            for mk in model_keys
        }
        for fut in as_completed(futures):
            mk = futures[fut]
            try:
                results[mk] = fut.result()
            except Exception as e:
                results[mk] = PoolCallResult(
                    model_key=mk, response=f"Error: {e}",
                    cost=PoolModelCost(model_key=mk),
                    success=False, error=str(e),
                )
    return results


# ---------------------------------------------------------------------------
# Router model calling (raw /generate for the router, which is fine for Qwen)
# ---------------------------------------------------------------------------

def call_router(
    prompt: str,
    model_key: str = "qwen2.5-3b-instruct",
    *,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
    timeout: int = 120,
) -> Tuple[str, int, int]:
    """Call the router model via raw /generate (Qwen handles raw text fine).

    Returns (response_text, prompt_tokens, completion_tokens).
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = MODEL_CONFIGS[model_key]
    host = cfg.get("ip_addr", DEFAULT_HOST)
    url = f"http://{host}:{cfg['port']}/generate"

    sampling_params: Dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9 if temperature > 0 else 1.0,
    }
    if seed is not None:
        sampling_params["sampling_seed"] = seed
    if stop:
        sampling_params["stop"] = stop

    try:
        resp = requests.post(url, json={"text": prompt, "sampling_params": sampling_params},
                             timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("text", "")
        meta = data.get("meta_info", {})
        pt = meta.get("prompt_tokens", len(prompt) // 4)
        ct = meta.get("completion_tokens", len(text) // 4)

        if stop:
            for s in stop:
                if s.lstrip("<") in text and s not in text:
                    text = text + s

        return text, pt, ct
    except Exception as e:
        logger.error(f"[router:{model_key}] error: {e}")
        return "", 0, 0


def calculate_cost(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = API_PRICE_1M_TOKENS.get(model_key, {"input": 0, "output": 0})
    return prompt_tokens * prices["input"] / 1_000_000 + completion_tokens * prices["output"] / 1_000_000
