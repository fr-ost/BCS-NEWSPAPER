"""LLM provider factory."""

from __future__ import annotations

import logging

from ..config import Settings
from .base import (
    BudgetExceededError,
    FallbackClient,
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

log = logging.getLogger(__name__)

__all__ = [
    "BudgetExceededError",
    "FallbackClient",
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
    "build_llm",
]


def build_provider(settings: Settings, provider_name: str | None = None) -> LLMProvider:
    name = (provider_name or settings.llm.provider).lower()
    cfg = settings.llm
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.secrets.openai_api_key,
            model=cfg.openai_model,
            api_base=cfg.openai_api_base,
            max_output_tokens=cfg.max_output_tokens,
            timeout=cfg.request_timeout_seconds,
            reasoning_effort=cfg.openai_reasoning_effort,
        )
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
    raise PermanentLLMError(f"Unknown LLM provider {name!r} (expected openai, gemini, ollama or mock)")


def build_client(settings: Settings, provider: LLMProvider, sleep=None) -> LLMClient:
    cfg = settings.llm
    kwargs = {} if sleep is None else {"sleep": sleep}
    if provider.name in ("mock", "scripted"):
        interval = 0.0
    elif provider.name == "openai":
        interval = cfg.openai_min_seconds_between_calls
    else:
        interval = cfg.min_seconds_between_calls
    return LLMClient(
        provider,
        max_calls=cfg.max_calls_per_run,
        min_interval=interval,
        max_retries=cfg.max_retries,
        backoff_base=cfg.backoff_base_seconds,
        backoff_max=cfg.backoff_max_seconds,
        max_retry_after=cfg.max_retry_after_seconds,
        **kwargs,
    )


def build_llm(settings: Settings, provider_override: str | None = None, sleep=None):
    """The model client the pipeline uses.

    With an override (``--provider``), exactly that provider is used. Otherwise
    the configured primary is followed by ``fallback_providers``; a provider
    whose credentials are missing is skipped with a warning.
    """
    if provider_override:
        return build_client(settings, build_provider(settings, provider_override), sleep=sleep)
    names = list(dict.fromkeys([settings.llm.provider, *settings.llm.fallback_providers]))
    clients, problems = [], []
    for name in names:
        try:
            clients.append(build_client(settings, build_provider(settings, name), sleep=sleep))
        except PermanentLLMError as exc:
            problems.append(f"{name}: {exc}")
            log.warning("Skipping provider %s: %s", name, exc)
    if not clients:
        raise PermanentLLMError("No usable LLM provider: " + "; ".join(problems))
    if len(clients) == 1:
        return clients[0]
    return FallbackClient(clients)
