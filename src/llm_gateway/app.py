"""FastAPI application factory.

Run with: ``uvicorn --factory llm_gateway.app:create_app``

Note: this module deliberately does not use ``from __future__ import annotations``. FastAPI must
resolve the ``Client`` dependency alias defined inside ``create_app``; with postponed annotations
it cannot, and the auth dependency would silently turn into a missing query parameter.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .cache import Cache, build_cache
from .config import Settings
from .providers import build_providers
from .schemas import ChatRequest, ChatResponse
from .security import Authenticator, FixedWindowLimiter
from .service import GatewayError, GatewayService, RouteError
from .usage import UsageStore


def create_app(
    settings: Settings | None = None,
    http_client: httpx.AsyncClient | None = None,
    cache: Cache | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> FastAPI:
    """Build the gateway. Arguments after ``settings`` exist so tests can inject fakes."""
    settings = settings or Settings.from_env()
    settings.validate()

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=settings.request_timeout)
    providers = build_providers(settings, client)
    usage = UsageStore(settings.db_path)
    service = GatewayService(
        settings,
        providers,
        cache or build_cache(settings.redis_url, settings.cache_max_items),
        usage,
        sleep or asyncio.sleep,
    )
    authenticator = Authenticator(settings.client_keys, settings.auth_disabled)
    limiter = FixedWindowLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_client:
            await client.aclose()
        usage.close()

    app = FastAPI(
        title="LLM Gateway",
        version="0.1.0",
        description="One API in front of OpenAI, Anthropic, Gemini and Groq.",
        lifespan=lifespan,
    )

    async def current_client(
        x_api_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> str:
        token = x_api_key
        if token is None and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        label = authenticator.authenticate(token)
        if label is None:
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        allowed, retry_after = limiter.allow(label)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        return label

    Client = Annotated[str, Depends(current_client)]

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "providers": sorted(settings.providers)}

    @app.get("/v1/providers", tags=["meta"])
    async def list_providers(_: Client) -> dict:
        return {
            "default_route": list(settings.default_route),
            "providers": [
                {"provider": name, "default_model": cfg.model}
                for name, cfg in settings.providers.items()
            ],
        }

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest, client_label: Client):
        try:
            return await service.chat(request, client_label)
        except RouteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GatewayError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "detail": "all providers failed",
                    "attempts": [a.model_dump(mode="json") for a in exc.attempts],
                },
            )

    @app.get("/v1/usage", tags=["usage"])
    async def usage_summary(client_label: Client) -> dict:
        return await asyncio.to_thread(usage.summary, client_label)

    return app
