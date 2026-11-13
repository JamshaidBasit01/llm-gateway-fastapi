"""Google Gemini (Generative Language API) adapter."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from .base import CompletionRequest, CompletionResult, Provider, ProviderError, post_json

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        base_url: str = GEMINI_BASE_URL,
        timeout: float = 30.0,
        omit_temperature: bool = False,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._omit_temperature = omit_temperature

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        url = f"{self._base_url}/v1beta/models/{quote(request.model, safe='')}:generateContent"
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        generation_config: dict = {"maxOutputTokens": request.max_tokens}
        if not self._omit_temperature:
            generation_config["temperature"] = request.temperature
        payload: dict = {
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [{"text": m.content}],
                }
                for m in request.messages
                if m.role != "system"
            ],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"x-goog-api-key": self._api_key}
        data = await post_json(self._client, self.name, url, headers, payload, self._timeout)

        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError("gemini returned no candidates (the prompt may have been blocked)")
        try:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
            usage = data.get("usageMetadata") or {}
            return CompletionResult(
                content=text,
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                completion_tokens=int(usage.get("candidatesTokenCount", 0)),
                model=str(data.get("modelVersion") or request.model),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError("gemini returned a malformed response") from exc
