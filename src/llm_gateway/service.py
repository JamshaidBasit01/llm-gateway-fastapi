"""Routing logic: cache lookup, retries with backoff, provider fallback, cost and usage logging."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from .cache import Cache, cache_key
from .config import Settings
from .pricing import compute_cost
from .providers import CompletionRequest, Provider, ProviderError
from .schemas import Attempt, ChatRequest, ChatResponse, Usage
from .usage import UsageStore


class RouteError(ValueError):
    """The request names a provider that is not enabled, or nothing is routable."""


class GatewayError(Exception):
    """Every provider in the route failed."""

    def __init__(self, attempts: list[Attempt]) -> None:
        super().__init__("all providers failed")
        self.attempts = attempts


class GatewayService:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, Provider],
        cache: Cache,
        usage: UsageStore,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._cache = cache
        self._usage = usage
        self._sleep = sleep

    # ----------------------------------------------------------------- routing
    def resolve_targets(self, request: ChatRequest) -> list[tuple[str, str]]:
        raw = request.targets or None
        resolved: list[tuple[str, str]] = []
        if raw is None:
            for name in self._settings.default_route:
                resolved.append((name, self._settings.providers[name].model))
        else:
            for target in raw:
                if target.provider not in self._providers:
                    raise RouteError(f"provider {target.provider!r} is not enabled")
                model = target.model or self._settings.providers[target.provider].model
                resolved.append((target.provider, model))
        if not resolved:
            raise RouteError("no providers are enabled for this route")
        return resolved

    # -------------------------------------------------------------------- chat
    async def chat(self, request: ChatRequest, client: str) -> ChatResponse:
        started = time.perf_counter()
        targets = self.resolve_targets(request)

        use_cache = request.cache and request.temperature == 0.0
        key = (
            cache_key(client, targets, request.messages, request.temperature, request.max_tokens)
            if use_cache
            else None
        )
        if key is not None:
            hit = await self._cache.get(key)
            if hit is not None:
                return await self._respond_from_cache(hit, client, started)

        attempts: list[Attempt] = []
        for provider_name, model in targets:
            result = await self._try_provider(provider_name, model, request, attempts)
            if result is None:
                continue
            cost = compute_cost(
                self._settings.pricing,
                provider_name,
                result.model,
                result.prompt_tokens,
                result.completion_tokens,
            )
            if cost is None and result.model != model:
                cost = compute_cost(
                    self._settings.pricing,
                    provider_name,
                    model,
                    result.prompt_tokens,
                    result.completion_tokens,
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            response = ChatResponse(
                content=result.content,
                provider=provider_name,
                model=result.model,
                usage=Usage(
                    prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens
                ),
                total_tokens=result.prompt_tokens + result.completion_tokens,
                cost_usd=cost,
                latency_ms=round(latency_ms, 2),
                cached=False,
                attempts=attempts,
            )
            await asyncio.to_thread(
                self._usage.record,
                client=client,
                provider=provider_name,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=cost,
                saved_usd=0.0,
                latency_ms=latency_ms,
                cached=False,
                status="ok",
            )
            if key is not None:
                stored = response.model_dump(mode="json", exclude={"attempts", "latency_ms"})
                await self._cache.set(key, stored, self._settings.cache_ttl_seconds)
            return response

        last_provider, last_model = targets[-1]
        await asyncio.to_thread(
            self._usage.record,
            client=client,
            provider=last_provider,
            model=last_model,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=None,
            saved_usd=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cached=False,
            status="error",
        )
        raise GatewayError(attempts)

    # ----------------------------------------------------------------- helpers
    async def _try_provider(
        self, provider_name: str, model: str, request: ChatRequest, attempts: list[Attempt]
    ):
        provider = self._providers[provider_name]
        completion = CompletionRequest(
            messages=request.messages,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        for retry in range(self._settings.max_retries + 1):
            tick = time.perf_counter()
            try:
                result = await provider.complete(completion)
            except ProviderError as exc:
                attempts.append(
                    Attempt(
                        provider=provider_name,
                        model=model,
                        ok=False,
                        status_code=exc.status_code,
                        error=str(exc),
                        latency_ms=round((time.perf_counter() - tick) * 1000.0, 2),
                    )
                )
                if exc.retryable and retry < self._settings.max_retries:
                    await self._sleep(self._settings.retry_backoff_seconds * (2**retry))
                    continue
                return None
            attempts.append(
                Attempt(
                    provider=provider_name,
                    model=model,
                    ok=True,
                    latency_ms=round((time.perf_counter() - tick) * 1000.0, 2),
                )
            )
            return result
        return None

    async def _respond_from_cache(self, hit: dict, client: str, started: float) -> ChatResponse:
        original_cost = hit.get("cost_usd")
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = ChatResponse(
            content=hit["content"],
            provider=hit["provider"],
            model=hit["model"],
            usage=Usage(**hit["usage"]),
            total_tokens=hit["total_tokens"],
            cost_usd=0.0,
            latency_ms=round(latency_ms, 2),
            cached=True,
            attempts=[],
        )
        await asyncio.to_thread(
            self._usage.record,
            client=client,
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=0.0,
            saved_usd=original_cost or 0.0,
            latency_ms=latency_ms,
            cached=True,
            status="ok",
        )
        return response
