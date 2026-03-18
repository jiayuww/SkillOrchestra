"""
LLM client abstraction for SkillOrchestra.

Supports multiple backends via a provider registry:
- openai: Official OpenAI API (api.openai.com)
- salesforce: Salesforce gateway (GPT-5)
- custom: Any OpenAI-compatible API (Anthropic, Azure, local, etc.)

Features:
- Raw text completion
- Structured output (JSON -> Pydantic model)
- Call tracking and usage statistics
- Retry with exponential backoff
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from pydantic import BaseModel
    _PYDANTIC_AVAILABLE = True
    T = TypeVar("T", bound=BaseModel)
except ImportError:
    _PYDANTIC_AVAILABLE = False
    T = TypeVar("T")


# ---------------------------------------------------------------------------
# Provider registry: extensible backend configuration
# ---------------------------------------------------------------------------

def _provider_openai(api_key: str, **kwargs: Any) -> tuple[str | None, str, Dict[str, str]]:
    """Official OpenAI API. Uses default api.openai.com/v1 when base_url is None."""
    return (None, api_key, {})


def _provider_salesforce(api_key: str, **kwargs: Any) -> tuple[str, str, Dict[str, str]]:
    """Salesforce gateway: X-Api-Key header, dummy bearer."""
    base_url = "https://gateway.salesforceresearch.ai/openai/process/v1/"
    return (base_url, "dummy", {"X-Api-Key": api_key})


def _provider_custom(
    api_key: str,
    base_url: str,
    header_name: Optional[str] = None,
    **kwargs: Any,
) -> tuple[str, str, Dict[str, str]]:
    """Custom OpenAI-compatible API. Optional custom auth header."""
    headers = {}
    if header_name:
        headers[header_name] = api_key
        return (base_url, "dummy", headers)
    return (base_url, api_key, {})


_PROVIDER_FACTORIES: Dict[str, Callable[..., tuple[str | None, str, Dict[str, str]]]] = {
    "openai": _provider_openai,
    "salesforce": _provider_salesforce,
    "custom": _provider_custom,
}


def register_provider(
    name: str,
    factory: Callable[..., tuple[str | None, str, Dict[str, str]]],
) -> None:
    """Register a custom LLM provider.

    The factory receives (api_key, **kwargs) and returns (base_url, api_key, headers).
    - base_url: None = use OpenAI SDK default (api.openai.com/v1)
    - api_key: The key to pass to OpenAI client
    - headers: Extra headers (e.g. X-Api-Key for gateway-style auth)

    Example (Anthropic-compatible via OpenAI SDK):
        def anthropic_factory(api_key, **kwargs):
            return ("https://api.anthropic.com/v1", api_key, {})
        register_provider("anthropic", anthropic_factory)
    """
    _PROVIDER_FACTORIES[name] = factory


def get_registered_providers() -> List[str]:
    """Return list of registered provider names."""
    return list(_PROVIDER_FACTORIES.keys())


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class CallRecord:
    """Debugging record for a single LLM call."""

    call_id: int = 0
    timestamp: str = ""
    role: str = ""
    prompt_preview: str = ""
    response_preview: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


class LLMClient:
    """LLM client with retry, structured output, and usage tracking.

    Supports multiple backends via the provider parameter:
    - "openai": Official OpenAI API (set OPENAI_API_KEY)
    - "salesforce": Salesforce gateway (set OPENAI_GATEWAY_KEY or OPENAI_API_KEY)
    - "custom": Any OpenAI-compatible API (set OPENAI_BASE_URL + OPENAI_API_KEY)

    Example (official OpenAI, no gateway):
        client = LLMClient(provider="openai", model="gpt-4o")
        resp = client.complete("What is 2+2?")

    Example (Salesforce gateway):
        client = LLMClient(provider="salesforce", model="gpt-5")

    Example (custom endpoint, e.g. Azure, local vLLM):
        client = LLMClient(provider="custom", base_url="https://your-api.com/v1")
    """

    def __init__(
        self,
        model: str = "gpt-5",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 8192,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **provider_kwargs: Any,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai library required: pip install openai")

        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        resolved_provider = provider or os.environ.get("LLM_PROVIDER")
        env_base_url = os.environ.get("OPENAI_BASE_URL")
        env_gateway_key = os.environ.get("OPENAI_GATEWAY_KEY")
        env_api_key = os.environ.get("OPENAI_API_KEY")

        if not resolved_provider:
            if env_base_url:
                resolved_provider = "custom"
            elif env_gateway_key:
                resolved_provider = "salesforce"
            else:
                resolved_provider = "openai"

        if resolved_provider not in _PROVIDER_FACTORIES:
            raise ValueError(
                f"Unknown provider '{resolved_provider}'. "
                f"Available: {get_registered_providers()}"
            )

        # Resolve API key per provider
        if resolved_provider == "openai":
            resolved_key = api_key or env_api_key
            if not resolved_key:
                raise ValueError("Set OPENAI_API_KEY for provider='openai'")
        elif resolved_provider == "salesforce":
            resolved_key = api_key or env_gateway_key or env_api_key
            if not resolved_key:
                raise ValueError(
                    "Set OPENAI_GATEWAY_KEY or OPENAI_API_KEY for provider='salesforce'"
                )
        else:  # custom
            resolved_key = api_key or env_api_key
            resolved_base = base_url or env_base_url
            if not resolved_base:
                raise ValueError(
                    "Set OPENAI_BASE_URL or pass base_url= for provider='custom'"
                )
            provider_kwargs["base_url"] = resolved_base

        factory = _PROVIDER_FACTORIES[resolved_provider]
        url, key, headers = factory(resolved_key, **provider_kwargs)

        client_kwargs: Dict[str, Any] = {"api_key": key}
        if url is not None:
            client_kwargs["base_url"] = url
        if headers:
            client_kwargs["default_headers"] = headers

        self.client = OpenAI(**client_kwargs)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.call_history: List[CallRecord] = []
        self._current_role: str = "unknown"

    def set_role(self, role: str) -> None:
        """Tag subsequent calls with a role name (for debugging)."""
        self._current_role = role

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a text completion."""
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return self._call_with_retry(messages, max_tokens or self.max_tokens, **kwargs)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model."""
        if not _PYDANTIC_AVAILABLE:
            raise ImportError("pydantic required for structured output")

        json_prompt = prompt + "\n\nRespond with valid JSON only, no markdown."
        response = self.complete(json_prompt, system_message=system_message, **kwargs)

        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            data = json.loads(content)
            return response_model(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error(f"Structured parse failed: {exc}\nContent: {content[:500]}")
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    **kwargs,
                )
                latency_ms = (time.time() - start) * 1000

                usage = response.usage
                pt = usage.prompt_tokens if usage else 0
                ct = usage.completion_tokens if usage else 0
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                self.total_calls += 1

                content = response.choices[0].message.content or ""

                self.call_history.append(CallRecord(
                    call_id=self.total_calls,
                    timestamp=datetime.now().isoformat(),
                    role=self._current_role,
                    prompt_preview=messages[-1]["content"][:200],
                    response_preview=content[:200],
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    latency_ms=latency_ms,
                    success=True,
                ))

                return LLMResponse(
                    content=content,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    model=self.model,
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                last_error = exc
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        self.total_calls += 1
        self.call_history.append(CallRecord(
            call_id=self.total_calls,
            timestamp=datetime.now().isoformat(),
            role=self._current_role,
            success=False,
            error=str(last_error),
        ))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")
