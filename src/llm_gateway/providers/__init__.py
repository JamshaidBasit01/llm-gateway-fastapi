"""Provider adapters and the factory that builds the enabled ones."""

from __future__ import annotations

import httpx

from ..config import Settings
from .anthropic import ANTHROPIC_URL, AnthropicProvider
from .base import CompletionRequest, CompletionResult, Provider, ProviderError
from .gemini import GEMINI_BASE_URL, GeminiProvider
from .mock import MockProvider
from .openai_compat import GROQ_URL, OPENAI_URL, OpenAICompatProvider

__all__ = [
    "CompletionRequest",
    "CompletionResult",
    "Provider",
    "ProviderError",
    "build_providers",
]


def build_providers(settings: Settings, client: httpx.AsyncClient) -> dict[str, Provider]:
    timeout = settings.request_timeout
    built: dict[str, Provider] = {}
    for name, cfg in settings.providers.items():
        if name == "openai":
            built[name] = OpenAICompatProvider(
                "openai",
                cfg.api_key or "",
                cfg.base_url or OPENAI_URL,
                client,
                timeout,
                max_tokens_field="max_completion_tokens",
                omit_temperature=cfg.omit_temperature,
            )
        elif name == "groq":
            built[name] = OpenAICompatProvider(
                "groq",
                cfg.api_key or "",
                cfg.base_url or GROQ_URL,
                client,
                timeout,
                omit_temperature=cfg.omit_temperature,
            )
        elif name == "anthropic":
            built[name] = AnthropicProvider(
                cfg.api_key or "",
                client,
                cfg.base_url or ANTHROPIC_URL,
                timeout,
                cfg.omit_temperature,
            )
        elif name == "gemini":
            built[name] = GeminiProvider(
                cfg.api_key or "",
                client,
                cfg.base_url or GEMINI_BASE_URL,
                timeout,
                cfg.omit_temperature,
            )
        elif name == "mock":
            built[name] = MockProvider()
    return built
