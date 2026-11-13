"""Each adapter is tested against a fake HTTP transport: no network, no API keys."""

import json

import httpx
from fastapi.testclient import TestClient

from .conftest import app_with_transport

BODY = {
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Capital of France?"},
    ],
    "max_tokens": 50,
}
HEADERS = {"x-api-key": "alice-secret"}


def _call(handler, provider: str, **env: str):
    app = app_with_transport(handler, **env)
    with TestClient(app) as c:
        body = {**BODY, "targets": [{"provider": provider}]}
        return c.post("/v1/chat", json=body, headers=HEADERS)


def test_openai_request_shape_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-test-2025",
                "choices": [{"message": {"content": "Paris"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    r = _call(handler, "openai", OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    assert r.status_code == 200
    data = r.json()
    assert data["content"] == "Paris" and data["model"] == "gpt-test-2025"
    assert data["usage"] == {"prompt_tokens": 12, "completion_tokens": 3}
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["payload"]["max_completion_tokens"] == 50
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "Be brief."}


def test_openai_omit_temperature_flag():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    r = _call(
        handler, "openai", OPENAI_API_KEY="k", OPENAI_MODEL="m", OPENAI_OMIT_TEMPERATURE="true"
    )
    assert r.status_code == 200 and "temperature" not in seen["payload"]


def test_groq_uses_its_own_url_and_max_tokens_field():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["payload"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    r = _call(handler, "groq", GROQ_API_KEY="gk", GROQ_MODEL="llama-test")
    assert r.status_code == 200
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["payload"]["max_tokens"] == 50


def test_anthropic_moves_system_prompt_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"], seen["payload"] = request.headers, json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [{"type": "text", "text": "Paris"}, {"type": "text", "text": "."}],
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        )

    r = _call(handler, "anthropic", ANTHROPIC_API_KEY="ak", ANTHROPIC_MODEL="claude-test")
    assert r.status_code == 200
    data = r.json()
    assert data["content"] == "Paris." and data["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 4,
    }
    assert seen["headers"]["x-api-key"] == "ak"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["payload"]["system"] == "Be brief."
    assert all(m["role"] != "system" for m in seen["payload"]["messages"])
    assert seen["payload"]["max_tokens"] == 50


def test_gemini_maps_roles_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"] = str(request.url), request.headers
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Paris"}]}}],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2},
            },
        )

    r = _call(handler, "gemini", GEMINI_API_KEY="gk", GEMINI_MODEL="gemini-test")
    assert r.status_code == 200
    assert r.json()["usage"] == {"prompt_tokens": 9, "completion_tokens": 2}
    assert seen["url"].endswith("/v1beta/models/gemini-test:generateContent")
    assert "gk" not in seen["url"]  # key travels in a header, never in the URL
    assert seen["headers"]["x-goog-api-key"] == "gk"
    roles = [c["role"] for c in seen["payload"]["contents"]]
    assert roles == ["user", "model", "user"]
    assert seen["payload"]["systemInstruction"]["parts"][0]["text"] == "Be brief."


def test_gemini_blocked_prompt_is_reported():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    r = _call(handler, "gemini", GEMINI_API_KEY="gk", GEMINI_MODEL="g")
    assert r.status_code == 502
    assert "no candidates" in r.json()["attempts"][0]["error"]


def test_error_responses_do_not_leak_the_api_key():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    r = _call(handler, "openai", OPENAI_API_KEY="sk-supersecret", OPENAI_MODEL="m")
    assert r.status_code == 502
    assert "sk-supersecret" not in r.text
    assert r.json()["attempts"][0]["status_code"] == 401


def test_malformed_provider_body_is_a_clean_failure():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    r = _call(handler, "openai", OPENAI_API_KEY="k", OPENAI_MODEL="m")
    assert r.status_code == 502
    assert "malformed" in r.json()["attempts"][0]["error"]
