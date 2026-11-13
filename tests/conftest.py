import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway import Settings, create_app

KEYS = "alice:alice-secret,bob:bob-secret"


def make_settings(**overrides: str) -> Settings:
    env = {
        "ENABLE_MOCK": "true",
        "GATEWAY_API_KEYS": KEYS,
        "DB_PATH": ":memory:",
        "RETRY_BACKOFF_SECONDS": "0",
        **overrides,
    }
    return Settings.from_env(env)


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def client():
    app = create_app(make_settings(), sleep=_no_sleep)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def alice():
    return {"x-api-key": "alice-secret"}


@pytest.fixture
def bob():
    return {"x-api-key": "bob-secret"}


def app_with_transport(handler, **env: str):
    """Build an app whose provider HTTP calls are answered by ``handler`` (no network)."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = make_settings(**env)
    return create_app(settings, http_client=http, sleep=_no_sleep)
