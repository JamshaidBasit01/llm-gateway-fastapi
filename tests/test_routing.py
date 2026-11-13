"""Retry, fallback and cost tracking, driven through a fake HTTP transport."""

import json

import httpx
from fastapi.testclient import TestClient

from .conftest import app_with_transport

HEADERS = {"x-api-key": "alice-secret"}
BODY = {"messages": [{"role": "user", "content": "hi"}]}
ENV = {
    "OPENAI_API_KEY": "sk",
    "OPENAI_MODEL": "gpt-test",
    "GROQ_API_KEY": "gk",
    "GROQ_MODEL": "llama-test",
    "DEFAULT_ROUTE": "openai,groq",
}


def _ok(text: str, prompt: int = 10, completion: int = 5) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


def test_falls_back_to_the_next_provider_when_the_first_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        if "openai.com" in str(request.url):
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return _ok("from groq")

    app = app_with_transport(handler, **ENV)
    with TestClient(app) as c:
        data = c.post("/v1/chat", json=BODY, headers=HEADERS).json()
    assert data["provider"] == "groq" and data["content"] == "from groq"
    openai_attempts = [a for a in data["attempts"] if a["provider"] == "openai"]
    assert len(openai_attempts) == 2 and not any(a["ok"] for a in openai_attempts)  # retried once


def test_retries_a_transient_error_on_the_same_provider():
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={}) if calls["n"] == 1 else _ok("recovered")

    app = app_with_transport(handler, OPENAI_API_KEY="sk", OPENAI_MODEL="m")
    with TestClient(app) as c:
        data = c.post("/v1/chat", json=BODY, headers=HEADERS).json()
    assert data["content"] == "recovered" and calls["n"] == 2


def test_client_errors_are_not_retried():
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    app = app_with_transport(
        handler, OPENAI_API_KEY="sk", OPENAI_MODEL="m", DEFAULT_ROUTE="openai"
    )
    with TestClient(app) as c:
        assert c.post("/v1/chat", json=BODY, headers=HEADERS).status_code == 502
    assert calls["n"] == 1


def test_explicit_targets_override_the_default_route():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return _ok("ok")

    app = app_with_transport(handler, **ENV)
    body = {**BODY, "targets": [{"provider": "groq", "model": "custom-model"}]}
    with TestClient(app) as c:
        data = c.post("/v1/chat", json=body, headers=HEADERS).json()
    assert data["provider"] == "groq" and seen == ["custom-model"]


def test_cost_is_computed_from_the_price_file_and_cache_hits_are_free(tmp_path):
    prices = tmp_path / "prices.json"
    prices.write_text(
        json.dumps({"openai/gpt-test": {"input_per_1m": 2.0, "output_per_1m": 10.0}})
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return _ok("ok", prompt=1_000_000, completion=500_000)

    app = app_with_transport(
        handler, OPENAI_API_KEY="sk", OPENAI_MODEL="gpt-test", PRICING_FILE=str(prices)
    )
    with TestClient(app) as c:
        first = c.post("/v1/chat", json=BODY, headers=HEADERS).json()
        second = c.post("/v1/chat", json=BODY, headers=HEADERS).json()
        usage = c.get("/v1/usage", headers=HEADERS).json()
    assert first["cost_usd"] == 2.0 + 5.0  # 1M in * $2 + 0.5M out * $10 per 1M
    assert second["cached"] is True and second["cost_usd"] == 0.0
    assert usage["cost_usd"] == 7.0 and usage["estimated_saved_usd"] == 7.0
    assert usage["unpriced_requests"] == 0
    assert usage["by_model"][0]["requests"] == 2


def test_unpriced_models_are_counted_not_guessed():
    app = app_with_transport(lambda _: _ok("ok"), OPENAI_API_KEY="sk", OPENAI_MODEL="m")
    with TestClient(app) as c:
        c.post("/v1/chat", json=BODY, headers=HEADERS)
        usage = c.get("/v1/usage", headers=HEADERS).json()
    assert usage["cost_usd"] == 0 and usage["unpriced_requests"] == 1
