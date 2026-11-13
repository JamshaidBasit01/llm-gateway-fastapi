"""Anthropic Messages API adapter."""

from __future__ import annotations

import httpx

from .base import CompletionRequest, CompletionResult, Provider, ProviderError, post_json

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        url: str = ANTHROPIC_URL,
        timeout: float = 30.0,
        omit_temperature: bool = False,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._url = url
        self._timeout = timeout
        self._omit_temperature = omit_temperature

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        payload: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,  # required by the Messages API
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system"
            ],
        }
        if system:
            payload["system"] = system
        if not self._omit_temperature:
            payload["temperature"] = request.temperature
        headers = {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}
        data = await post_json(self._client, self.name, self._url, headers, payload, self._timeout)
        try:
            text = "".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
            usage = data.get("usage") or {}
            return CompletionResult(
                content=text,
                prompt_tokens=int(usage.get("input_tokens", 0)),
                completion_tokens=int(usage.get("output_tokens", 0)),
                model=str(data.get("model") or request.model),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("anthropic returned a malformed response") from exc
