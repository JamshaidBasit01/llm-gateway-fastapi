"""Settings, read from environment variables (see ``.env.example``)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from .pricing import load_pricing

REAL_PROVIDERS = ("openai", "anthropic", "gemini", "groq")
PROVIDER_ORDER = (*REAL_PROVIDERS, "mock")


class ConfigError(ValueError):
    """Raised when the environment describes an invalid or unsafe configuration."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str | None
    model: str
    base_url: str | None = None
    omit_temperature: bool = False


@dataclass(frozen=True)
class Settings:
    providers: dict[str, ProviderConfig]
    default_route: tuple[str, ...]
    client_keys: dict[str, str] = field(repr=False)  # label -> secret key
    auth_disabled: bool = False
    rate_limit_per_minute: int = 60
    cache_ttl_seconds: int = 3600
    cache_max_items: int = 1000
    redis_url: str | None = None
    request_timeout: float = 30.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.5
    db_path: str = "gateway_usage.db"
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        providers: dict[str, ProviderConfig] = {}
        for name in REAL_PROVIDERS:
            prefix = name.upper()
            key, model = env.get(f"{prefix}_API_KEY"), env.get(f"{prefix}_MODEL")
            if key and model:
                providers[name] = ProviderConfig(
                    name=name,
                    api_key=key,
                    model=model,
                    base_url=env.get(f"{prefix}_BASE_URL") or None,
                    omit_temperature=_truthy(env.get(f"{prefix}_OMIT_TEMPERATURE")),
                )
            elif key or model:
                raise ConfigError(f"set both {prefix}_API_KEY and {prefix}_MODEL, or neither")
        if _truthy(env.get("ENABLE_MOCK")):
            providers["mock"] = ProviderConfig("mock", None, "echo")

        enabled = [name for name in PROVIDER_ORDER if name in providers]
        raw_route = env.get("DEFAULT_ROUTE", "")
        route = tuple(p.strip() for p in raw_route.split(",") if p.strip()) or tuple(enabled)
        unknown = [p for p in route if p not in providers]
        if unknown:
            raise ConfigError(f"DEFAULT_ROUTE lists providers that are not enabled: {unknown}")

        return cls(
            providers=providers,
            default_route=route,
            client_keys=_parse_client_keys(env.get("GATEWAY_API_KEYS", "")),
            auth_disabled=_truthy(env.get("GATEWAY_AUTH_DISABLED")),
            rate_limit_per_minute=_int(env, "RATE_LIMIT_PER_MINUTE", 60, minimum=0),
            cache_ttl_seconds=_int(env, "CACHE_TTL_SECONDS", 3600, minimum=1),
            cache_max_items=_int(env, "CACHE_MAX_ITEMS", 1000, minimum=1),
            redis_url=env.get("REDIS_URL") or None,
            request_timeout=_float(env, "REQUEST_TIMEOUT", 30.0, minimum=0.1),
            max_retries=_int(env, "MAX_RETRIES", 1, minimum=0),
            retry_backoff_seconds=_float(env, "RETRY_BACKOFF_SECONDS", 0.5, minimum=0.0),
            db_path=env.get("DB_PATH", "gateway_usage.db"),
            pricing=load_pricing(env.get("PRICING_FILE") or None),
        )

    # --------------------------------------------------------------- validate
    def validate(self) -> None:
        """Refuse to start in an unsafe state (open gateway) or with nothing to route to."""
        if not self.auth_disabled and not self.client_keys:
            raise ConfigError(
                "no client keys configured: set GATEWAY_API_KEYS (e.g. 'alice:secret') "
                "or GATEWAY_AUTH_DISABLED=true for local development only"
            )
        if not self.providers:
            raise ConfigError(
                "no providers enabled: set <PROVIDER>_API_KEY and <PROVIDER>_MODEL, "
                "or ENABLE_MOCK=true to try the gateway without real keys"
            )


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, minimum: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _parse_client_keys(raw: str) -> dict[str, str]:
    """Parse ``label:key,label2:key2`` (a bare ``key`` gets the label ``client-N``)."""
    keys: dict[str, str] = {}
    for index, entry in enumerate(e.strip() for e in raw.split(",") if e.strip()):
        label, sep, secret = entry.partition(":")
        if not sep:
            label, secret = f"client-{index + 1}", entry
        if not label or not secret:
            raise ConfigError("GATEWAY_API_KEYS entries must look like 'label:key'")
        if label in keys:
            raise ConfigError(f"duplicate client label {label!r} in GATEWAY_API_KEYS")
        keys[label] = secret
    return keys
