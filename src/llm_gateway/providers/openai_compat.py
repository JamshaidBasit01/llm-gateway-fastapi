"""OpenAI Chat Completions and any API that speaks the same protocol (e.g. Groq)."""

from __future__ import annotations

import httpx

from .base import CompletionRequest, CompletionResult, Provider, ProviderError, post_json

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        name: str,
        api_key: str,
        url: str,
        client: httpx.AsyncClient,
        timeout: float = 30.0,
        max_tokens_field: str = "max_tokens",
        omit_temperature: bool = False,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._url = url
        self._client = client
        self._timeout = timeout
        self._max_tokens_field = max_tokens_field
        self._omit_temperature = omit_temperature

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        payload: dict = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            self._max_tokens_field: request.max_tokens,
        }
        if not self._omit_temperature:
            payload["temperature"] = request.temperature
        headers = {"Authorization": f"Bearer {self._api_key}"}
        data = await post_json(self._client, self.name, self._url, headers, payload, self._timeout)
        try:
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            return CompletionResult(
                content=content,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                model=str(data.get("model") or request.model),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{self.name} returned a malformed response") from exc
