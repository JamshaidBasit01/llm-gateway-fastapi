"""A deterministic offline provider for demos and tests (enable with ``ENABLE_MOCK=true``).

It echoes the last user message. A message containing ``[fail]`` raises a retryable error, so
the retry and fallback behaviour can be tried without any API key.
"""

from __future__ import annotations

from .base import CompletionRequest, CompletionResult, Provider, ProviderError


class MockProvider(Provider):
    name = "mock"

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        if "[fail]" in last_user:
            raise ProviderError("mock provider failure requested", retryable=True, status_code=503)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        content = f"echo: {last_user}"
        return CompletionResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(content.split()),
            model=request.model,
        )
