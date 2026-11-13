"""Provider interface shared by every adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from ..schemas import Message


@dataclass(frozen=True)
class CompletionRequest:
    messages: list[Message]
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class ProviderError(Exception):
    """A provider call failed. ``retryable`` means a retry on the same provider may succeed."""

    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one chat completion or raise :class:`ProviderError`."""


async def post_json(
    client: httpx.AsyncClient,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: float,
) -> dict:
    """POST JSON and return the decoded body, mapping failures to :class:`ProviderError`."""
    try:
        response = await client.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{provider} timed out", retryable=True) from exc
    except httpx.TransportError as exc:
        raise ProviderError(
            f"{provider} connection error: {type(exc).__name__}", retryable=True
        ) from exc

    if response.status_code >= 400:
        retryable = response.status_code == 429 or response.status_code >= 500
        raise ProviderError(
            f"{provider} returned HTTP {response.status_code}: {_error_message(response)}",
            retryable=retryable,
            status_code=response.status_code,
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(f"{provider} returned a non-JSON body") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"{provider} returned an unexpected JSON body")
    return data


def _error_message(response: httpx.Response) -> str:
    """Short, key-free description of an error response."""
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(error, dict):
            error = error.get("message") or error
        return str(error)[:200]
    except ValueError:
        return response.text[:200]
