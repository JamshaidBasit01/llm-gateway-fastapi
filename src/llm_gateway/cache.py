"""Response caches: in-memory (default) or Redis (set ``REDIS_URL``)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .schemas import Message


class Cache(Protocol):
    async def get(self, key: str) -> dict | None: ...

    async def set(self, key: str, value: dict, ttl: int) -> None: ...


def cache_key(
    client: str,
    targets: Sequence[tuple[str, str]],
    messages: Sequence[Message],
    temperature: float,
    max_tokens: int,
) -> str:
    """Stable hash of everything that determines the answer.

    The client label is part of the key, so one client never receives another client's
    cached answer.
    """
    payload = {
        "client": client,
        "targets": [list(t) for t in targets],
        "messages": [[m.role, m.content] for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class MemoryCache:
    """Small TTL + LRU cache. Process-local: use Redis to share across workers."""

    def __init__(self, max_items: int = 1000, clock: Callable[[], float] = time.monotonic):
        self._max_items = max_items
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    async def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= self._clock():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: dict, ttl: int) -> None:
        self._data[key] = (self._clock() + ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_items:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class RedisCache:
    """Redis-backed cache. ``client`` is a ``redis.asyncio.Redis`` (or compatible) instance."""

    def __init__(self, client: Any, prefix: str = "llmgw:") -> None:
        self._client = client
        self._prefix = prefix

    async def get(self, key: str) -> dict | None:
        raw = await self._client.get(self._prefix + key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    async def set(self, key: str, value: dict, ttl: int) -> None:
        await self._client.set(self._prefix + key, json.dumps(value), ex=ttl)


def build_cache(redis_url: str | None, max_items: int) -> Cache:
    if not redis_url:
        return MemoryCache(max_items)
    try:
        import redis.asyncio as redis_asyncio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "REDIS_URL is set but the redis package is missing: "
            "pip install 'llm-gateway-fastapi[redis]'"
        ) from exc
    return RedisCache(redis_asyncio.from_url(redis_url))
