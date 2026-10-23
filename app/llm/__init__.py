"""LLM provider factory."""

from __future__ import annotations

from ..config import Settings
from .base import (
    BudgetExceededError,
    InvalidResponseError,
    LLMClient,
    LLMError,
    LLMProvider,
    LLMRequest,
    ModelUnavailableError,
    PermanentLLMError,
    QuotaExhaustedError,
    RateLimitError,
    TransientLLMError,
)

__all__ = [
    "BudgetExceededError",
    "InvalidResponseError",
    "LLMClient",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "ModelUnavailableError",
    "PermanentLLMError",
    "QuotaExhaustedError",
    "RateLimitError",
    "TransientLLMError",
    "build_provider",
    "build_client",
]


def build_provider(settings: Settings, provider_name: str | None = None) -> LLMProvider:
    name = (provider_name or settings.llm.provider).lower()
    cfg = settings.llm
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.secrets.gemini_api_key,
            model=cfg.gemini_model,
            api_base=cfg.gemini_api_base,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            timeout=cfg.request_timeout_seconds,
        )
    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(
            model=cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            timeout=cfg.request_timeout_seconds,
        )
    if name == "mock":
        from .mock import MockProvider

        return MockProvider()
    raise PermanentLLMError(f"Unknown LLM provider {name!r} (expected gemini, ollama or mock)")


def build_client(settings: Settings, provider: LLMProvider, sleep=None) -> LLMClient:
    cfg = settings.llm
    kwargs = {} if sleep is None else {"sleep": sleep}
    no_spacing = provider.name in ("mock", "scripted")
    return LLMClient(
        provider,
        max_calls=cfg.max_calls_per_run,
        min_interval=0.0 if no_spacing else cfg.min_seconds_between_calls,
        max_retries=cfg.max_retries,
        backoff_base=cfg.backoff_base_seconds,
        backoff_max=cfg.backoff_max_seconds,
        max_retry_after=cfg.max_retry_after_seconds,
        **kwargs,
    )
