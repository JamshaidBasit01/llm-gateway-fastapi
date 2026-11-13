import asyncio
import json

import pytest

from llm_gateway.cache import MemoryCache, cache_key
from llm_gateway.config import ConfigError, Settings
from llm_gateway.pricing import compute_cost, load_pricing
from llm_gateway.schemas import Message
from llm_gateway.security import Authenticator, FixedWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------- config
def test_startup_refuses_open_gateway_and_empty_provider_list():
    with pytest.raises(ConfigError, match="no client keys"):
        Settings.from_env({"ENABLE_MOCK": "true"}).validate()
    with pytest.raises(ConfigError, match="no providers"):
        Settings.from_env({"GATEWAY_API_KEYS": "a:b"}).validate()
    Settings.from_env({"ENABLE_MOCK": "true", "GATEWAY_AUTH_DISABLED": "true"}).validate()


def test_half_configured_provider_is_an_error():
    with pytest.raises(ConfigError, match="both OPENAI_API_KEY and OPENAI_MODEL"):
        Settings.from_env({"OPENAI_API_KEY": "k"})


def test_default_route_must_name_enabled_providers():
    with pytest.raises(ConfigError, match="not enabled"):
        Settings.from_env({"ENABLE_MOCK": "true", "DEFAULT_ROUTE": "openai"})


def test_numeric_settings_are_validated():
    with pytest.raises(ConfigError, match="integer"):
        Settings.from_env({"RATE_LIMIT_PER_MINUTE": "many"})
    with pytest.raises(ConfigError, match=">="):
        Settings.from_env({"CACHE_TTL_SECONDS": "0"})


def test_client_key_parsing_and_duplicates():
    s = Settings.from_env({"GATEWAY_API_KEYS": "alice:one, bob:two, plainkey"})
    assert s.client_keys == {"alice": "one", "bob": "two", "client-3": "plainkey"}
    with pytest.raises(ConfigError, match="duplicate"):
        Settings.from_env({"GATEWAY_API_KEYS": "a:1,a:2"})


def test_secrets_are_not_printed_in_settings_repr():
    s = Settings.from_env({"GATEWAY_API_KEYS": "alice:topsecret", "ENABLE_MOCK": "true"})
    assert "topsecret" not in repr(s)


# ------------------------------------------------------------------ pricing
def test_compute_cost_and_unknown_model():
    table = {"openai/m": {"input_per_1m": 1.0, "output_per_1m": 3.0}}
    assert compute_cost(table, "openai", "m", 500_000, 100_000) == pytest.approx(0.8)
    assert compute_cost(table, "openai", "other", 1, 1) is None


def test_load_pricing_validates_entries(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"_comment": "x", "a/b": {"input_per_1m": 1, "output_per_1m": 2}}))
    assert load_pricing(good) == {"a/b": {"input_per_1m": 1.0, "output_per_1m": 2.0}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a/b": {"input_per_1m": -1, "output_per_1m": 2}}))
    with pytest.raises(ValueError, match="negative"):
        load_pricing(bad)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"a/b": {"input_per_1m": 1}}))
    with pytest.raises(ValueError, match="numeric"):
        load_pricing(missing)


# -------------------------------------------------------------------- cache
def test_cache_key_depends_on_every_input():
    msgs = [Message(role="user", content="hi")]
    base = cache_key("alice", [("mock", "echo")], msgs, 0.0, 100)
    assert base == cache_key("alice", [("mock", "echo")], msgs, 0.0, 100)
    assert base != cache_key("bob", [("mock", "echo")], msgs, 0.0, 100)
    assert base != cache_key("alice", [("mock", "other")], msgs, 0.0, 100)
    assert base != cache_key("alice", [("mock", "echo")], msgs, 0.0, 200)
    other = [Message(role="user", content="hello")]
    assert base != cache_key("alice", [("mock", "echo")], other, 0.0, 100)


def test_memory_cache_expires_and_evicts_least_recently_used():
    clock = FakeClock()
    cache = MemoryCache(max_items=2, clock=clock)

    async def scenario():
        await cache.set("a", {"v": 1}, ttl=10)
        await cache.set("b", {"v": 2}, ttl=10)
        assert await cache.get("a") == {"v": 1}  # touch a, so b is now the oldest
        await cache.set("c", {"v": 3}, ttl=10)
        assert await cache.get("b") is None and await cache.get("a") is not None
        clock.now = 11
        assert await cache.get("a") is None  # expired

    asyncio.run(scenario())


# ----------------------------------------------------------------- security
def test_authenticator():
    auth = Authenticator({"alice": "secret"})
    assert auth.authenticate("secret") == "alice"
    assert auth.authenticate("wrong") is None and auth.authenticate(None) is None
    assert Authenticator({}, disabled=True).authenticate(None) == "anonymous"


def test_rate_limiter_window_resets():
    clock = FakeClock()
    limiter = FixedWindowLimiter(limit=2, window=60, clock=clock)
    assert limiter.allow("a") == (True, 0) and limiter.allow("a") == (True, 0)
    allowed, retry = limiter.allow("a")
    assert not allowed and retry == 60
    clock.now = 59
    assert limiter.allow("a") == (False, 1)
    clock.now = 60
    assert limiter.allow("a")[0] is True
    assert limiter.allow("b")[0] is True  # separate budget per client


def test_rate_limit_of_zero_disables_limiting():
    limiter = FixedWindowLimiter(limit=0)
    assert all(limiter.allow("a")[0] for _ in range(100))
