"""
LLM_CALL module for eval_frames.py compatibility.

Provides get_llm_response used by eval_frames.py.
Supports:
  - Official OpenAI API (api.openai.com) when OPENAI_API_KEY is set, no gateway
  - Salesforce Gateway (GPT-5) when OPENAI_GATEWAY_KEY or OPENAI_BASE_URL points to gateway
  - vLLM/SGLang for local models

Loads .env from repository root so OPENAI_API_KEY / OPENAI_GATEWAY_KEY work when used standalone.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from repository root (same as eval_frames, pipeline)
try:
    from dotenv import load_dotenv
    _llm_call_root = Path(__file__).resolve().parent.parent
    load_dotenv(_llm_call_root / ".env")
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Provider selection
SALESFORCE_GATEWAY_URL = "https://gateway.salesforceresearch.ai/openai/process/v1/"


def _resolve_api_provider() -> str:
    """Resolve which API provider to use: openai, salesforce, or custom."""
    provider = os.environ.get("LLM_PROVIDER")
    base_url = os.environ.get("OPENAI_BASE_URL")
    gateway_key = os.environ.get("OPENAI_GATEWAY_KEY")
    api_key = os.environ.get("OPENAI_API_KEY")

    if provider:
        return provider
    if base_url:
        return "custom" if "salesforce" not in base_url.lower() else "salesforce"
    if gateway_key:
        return "salesforce"
    return "openai"


def _get_openai_official_client(timeout: Optional[float] = None) -> "OpenAI":
    """Get OpenAI client for official API (api.openai.com). Requires OPENAI_API_KEY."""
    if not OpenAI:
        raise ImportError("openai library required: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for official OpenAI API")
    t = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "60"))
    return OpenAI(api_key=api_key, timeout=t)


def _get_salesforce_client(timeout: Optional[float] = None) -> "OpenAI":
    """Get OpenAI client for Salesforce Gateway (GPT-5).
    When OPENAI_GATEWAY_KEY/OPENAI_API_KEY is not set, uses no auth (internal network).
    """
    if not OpenAI:
        raise ImportError("openai library required: pip install openai")
    api_key = os.environ.get("OPENAI_GATEWAY_KEY") or os.environ.get("OPENAI_API_KEY")
    t = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "60"))
    base_url = os.environ.get("OPENAI_BASE_URL") or SALESFORCE_GATEWAY_URL
    if api_key:
        return OpenAI(
            base_url=base_url,
            api_key="dummy",
            default_headers={"X-Api-Key": api_key},
            timeout=t,
        )
    return OpenAI(base_url=base_url, api_key="dummy", timeout=t)


def _get_custom_client(timeout: Optional[float] = None) -> "OpenAI":
    """Get OpenAI client for custom OpenAI-compatible endpoint (OPENAI_BASE_URL)."""
    if not OpenAI:
        raise ImportError("openai library required: pip install openai")
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL required for custom provider")
    t = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "60"))
    return OpenAI(base_url=base_url, api_key=api_key or "dummy", timeout=t)


_LLM_PROVIDER_LOGGED = False

def _get_api_client(timeout: Optional[float] = None) -> "OpenAI":
    """Get the appropriate API client based on env (openai, salesforce, custom)."""
    global _LLM_PROVIDER_LOGGED
    provider = _resolve_api_provider()
    if not _LLM_PROVIDER_LOGGED:
        base_set = "set" if os.environ.get("OPENAI_BASE_URL") else "default"
        t = timeout if timeout is not None else os.environ.get("OPENAI_TIMEOUT", "60")
        print(f"[LLM_CALL] API provider={provider}, OPENAI_BASE_URL={base_set}, timeout={t}s")
        _LLM_PROVIDER_LOGGED = True
    if provider == "openai":
        return _get_openai_official_client(timeout=timeout)
    if provider == "salesforce":
        return _get_salesforce_client(timeout=timeout)
    return _get_custom_client(timeout=timeout)


class ContextLengthExceeded(Exception):
    """Raised when vLLM returns 400 due to input exceeding max context length. Caller (eval_frames) catches and retries with truncated context."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


def _get_vllm_client(ip_addr: str, port: int) -> "OpenAI":
    """Get OpenAI client for vLLM/SGLang server."""
    if not OpenAI:
        raise ImportError("openai library required: pip install openai")
    return OpenAI(
        api_key="EMPTY",
        base_url=f"http://{ip_addr}:{port}/v1",
    )


def get_llm_response(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 1.0,
    return_raw_response: bool = False,
    tools: Optional[List[Dict]] = None,
    show_messages: bool = False,
    model_type: Optional[str] = None,
    max_length: int = 1024,
    model_config: Optional[List[Dict]] = None,
    model_config_idx: int = 0,
    model_config_path: Optional[str] = None,
    payload: Optional[Dict] = None,
    **kwargs,
):
    """
    Unified LLM response for eval_frames compatibility.

    Supports:
      - gpt-5, gpt-5-mini: Salesforce Gateway
      - Qwen, Llama, etc.: vLLM via model_config
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # API models (GPT-5, GPT-4o, etc.)
    if model in ("gpt-5", "gpt-5-mini", "o3", "o3-mini", "gpt-4o", "gpt-4o-mini"):
        if max_length == 1024:
            max_length = 40000
        # Official OpenAI gpt-4o/gpt-4o-mini support max 16384 completion tokens
        if _resolve_api_provider() == "openai" and model in ("gpt-4o", "gpt-4o-mini"):
            max_length = min(max_length, 16384)
        base_timeout = float(os.environ.get("OPENAI_TIMEOUT", "60"))
        max_timeout = float(os.environ.get("OPENAI_TIMEOUT_MAX", "150"))
        answer = ""
        retries = 0
        max_retries = 10
        while answer == "" and retries < max_retries:
            retry_timeout = min(base_timeout + 30 * retries, max_timeout)
            client = _get_api_client(timeout=retry_timeout)
            try:
                kwargs_create = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_length,
                }
                if tools:
                    kwargs_create["tools"] = tools
                chat_completion = client.chat.completions.create(**kwargs_create)
                if return_raw_response:
                    return chat_completion
                return chat_completion.choices[0].message.content or ""
            except Exception as error:
                print(f"API error (retry {retries}): {error}")
                retries += 1
                time.sleep(min(60 * retries, 300))
        raise RuntimeError(f"Failed to get response from {model} after {max_retries} retries")

    # vLLM models (Qwen, Llama, etc.)
    if (
        model_config is not None
        or "qwen" in model.lower()
        or "llama" in model.lower()
        or model_type in ("vllm", "sglang")
    ):
        if model_config is None and model_config_path:
            try:
                with open(model_config_path) as f:
                    configs = json.load(f)
                model_config = configs.get(model)
            except Exception:
                pass
        if model_config is None:
            raise ValueError(
                f"model_config required for {model}. "
                "Pass model_config or model_config_path."
            )
        answer = ""
        retries = 0
        max_retries = 10
        while answer == "" and retries < max_retries:
            cfg = model_config[model_config_idx % len(model_config)]
            ip_addr = cfg.get("ip_addr", cfg.get("host", "localhost"))
            port = cfg.get("port", 8000)
            try:
                client = _get_vllm_client(ip_addr, port)
                kwargs_create = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_length,
                    "temperature": temperature,
                }
                if tools:
                    kwargs_create["tools"] = tools
                chat_completion = client.chat.completions.create(**kwargs_create)
                if return_raw_response:
                    return chat_completion
                return chat_completion.choices[0].message.content or ""
            except Exception as error:
                err_str = str(error)
                is_context_error = "exceeds" in err_str.lower() or "input length" in err_str.lower() or "maximum allowed length" in err_str.lower()
                if is_context_error:
                    raise ContextLengthExceeded(err_str, error) from error
                print(f"vLLM error ({ip_addr}:{port}, retry {retries}): {error}")
                retries += 1
                if model_config_path and os.path.isfile(model_config_path):
                    try:
                        with open(model_config_path) as f:
                            configs = json.load(f)
                        if model in configs:
                            model_config = configs[model]
                    except Exception:
                        pass
                time.sleep(min(60 * retries, 300))
        raise RuntimeError(f"Failed to get response from {model} after {max_retries} retries")

    raise ValueError(
        f"Unknown model: {model}. "
        "Supported: gpt-5, gpt-5-mini, or vLLM models with model_config."
    )
