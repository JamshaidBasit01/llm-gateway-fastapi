from fastapi.testclient import TestClient

from llm_gateway import create_app

from .conftest import _no_sleep, make_settings

BODY = {"messages": [{"role": "user", "content": "hello gateway"}]}


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "providers": ["mock"]}


def test_chat_requires_a_valid_key(client):
    assert client.post("/v1/chat", json=BODY).status_code == 401
    assert client.post("/v1/chat", json=BODY, headers={"x-api-key": "nope"}).status_code == 401


def test_chat_accepts_bearer_token(client):
    r = client.post("/v1/chat", json=BODY, headers={"Authorization": "Bearer alice-secret"})
    assert r.status_code == 200


def test_chat_returns_answer_usage_and_attempts(client, alice):
    r = client.post("/v1/chat", json=BODY, headers=alice)
    assert r.status_code == 200
    data = r.json()
    assert data["content"] == "echo: hello gateway"
    assert data["provider"] == "mock" and data["cached"] is False
    usage = data["usage"]
    assert data["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert data["cost_usd"] is None  # no price configured: never guessed
    assert [a["ok"] for a in data["attempts"]] == [True]


def test_second_identical_request_is_served_from_cache(client, alice):
    first = client.post("/v1/chat", json=BODY, headers=alice).json()
    second = client.post("/v1/chat", json=BODY, headers=alice).json()
    assert first["cached"] is False and second["cached"] is True
    assert second["content"] == first["content"] and second["cost_usd"] == 0.0


def test_cache_is_not_shared_between_clients(client, alice, bob):
    client.post("/v1/chat", json=BODY, headers=alice)
    assert client.post("/v1/chat", json=BODY, headers=bob).json()["cached"] is False


def test_cache_can_be_bypassed_and_skips_nonzero_temperature(client, alice):
    client.post("/v1/chat", json=BODY, headers=alice)
    no_cache = {**BODY, "cache": False}
    assert client.post("/v1/chat", json=no_cache, headers=alice).json()["cached"] is False
    warm = {**BODY, "temperature": 0.7}
    client.post("/v1/chat", json=warm, headers=alice)
    assert client.post("/v1/chat", json=warm, headers=alice).json()["cached"] is False


def test_validation_errors(client, alice):
    assert client.post("/v1/chat", json={"messages": []}, headers=alice).status_code == 422
    only_system = {"messages": [{"role": "system", "content": "be brief"}]}
    assert client.post("/v1/chat", json=only_system, headers=alice).status_code == 422
    bad_temp = {**BODY, "temperature": 5}
    assert client.post("/v1/chat", json=bad_temp, headers=alice).status_code == 422


def test_unknown_provider_is_a_400(client, alice):
    body = {**BODY, "targets": [{"provider": "openai"}]}
    r = client.post("/v1/chat", json=body, headers=alice)
    assert r.status_code == 400 and "not enabled" in r.json()["detail"]


def test_all_providers_failing_returns_502_with_attempts(client, alice):
    body = {"messages": [{"role": "user", "content": "please [fail]"}]}
    r = client.post("/v1/chat", json=body, headers=alice)
    assert r.status_code == 502
    attempts = r.json()["attempts"]
    assert len(attempts) == 2  # 1 try + MAX_RETRIES (default 1)
    assert all(not a["ok"] and a["status_code"] == 503 for a in attempts)


def test_rate_limit_returns_429_with_retry_after():
    app = create_app(make_settings(RATE_LIMIT_PER_MINUTE="2"), sleep=_no_sleep)
    headers = {"x-api-key": "alice-secret"}
    with TestClient(app) as c:
        assert c.post("/v1/chat", json=BODY, headers=headers).status_code == 200
        assert c.post("/v1/chat", json=BODY, headers=headers).status_code == 200
        blocked = c.post("/v1/chat", json=BODY, headers=headers)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1
        # another client has its own budget
        assert c.post("/v1/chat", json=BODY, headers={"x-api-key": "bob-secret"}).status_code == 200


def test_usage_summary_counts_requests_cache_hits_and_errors(client, alice, bob):
    client.post("/v1/chat", json=BODY, headers=alice)
    client.post("/v1/chat", json=BODY, headers=alice)  # cache hit
    client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "[fail]"}]}, headers=alice
    )
    usage = client.get("/v1/usage", headers=alice).json()
    assert usage["client"] == "alice"
    assert usage["requests"] == 3 and usage["cache_hits"] == 1 and usage["errors"] == 1
    assert usage["cache_hit_rate"] == round(1 / 3, 4)
    assert client.get("/v1/usage", headers=bob).json()["requests"] == 0


def test_providers_endpoint_lists_routes(client, alice):
    data = client.get("/v1/providers", headers=alice).json()
    assert data["default_route"] == ["mock"]
    assert data["providers"] == [{"provider": "mock", "default_model": "echo"}]
