"""Client authentication and per-client rate limiting."""

from __future__ import annotations

import hmac
import math
import time
from collections.abc import Callable


class Authenticator:
    """Maps an API key to a client label using constant-time comparisons."""

    def __init__(self, keys: dict[str, str], disabled: bool = False) -> None:
        self._keys = dict(keys)
        self._disabled = disabled

    def authenticate(self, token: str | None) -> str | None:
        if self._disabled:
            return "anonymous"
        if not token:
            return None
        found: str | None = None
        for label, secret in self._keys.items():  # no early exit: timing does not leak position
            if hmac.compare_digest(token.encode("utf-8"), secret.encode("utf-8")):
                found = label
        return found


class FixedWindowLimiter:
    """Allow at most ``limit`` requests per ``window`` seconds for each client."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._state: dict[str, tuple[float, int]] = {}

    def allow(self, client: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``. A limit of 0 disables limiting."""
        if self._limit <= 0:
            return True, 0
        now = self._clock()
        start, count = self._state.get(client, (now, 0))
        if now - start >= self._window:
            start, count = now, 0
        if count >= self._limit:
            return False, max(1, math.ceil(self._window - (now - start)))
        self._state[client] = (start, count + 1)
        return True, 0
