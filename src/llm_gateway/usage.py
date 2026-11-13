"""SQLite-backed request log and per-client usage summary."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    client TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL,
    saved_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    cached INTEGER NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_client ON requests (client, ts);
"""


class UsageStore:
    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record(
        self,
        *,
        client: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None,
        saved_usd: float,
        latency_ms: float,
        cached: bool,
        status: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests (ts, client, provider, model, prompt_tokens, "
                "completion_tokens, cost_usd, saved_usd, latency_ms, cached, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(), client, provider, model, prompt_tokens, completion_tokens,
                    cost_usd, saved_usd, latency_ms, int(cached), status,
                ),
            )
            self._conn.commit()

    def summary(self, client: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cached), 0), "
                "COALESCE(SUM(status = 'error'), 0), "
                "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
                "COALESCE(SUM(cost_usd), 0), COALESCE(SUM(saved_usd), 0), "
                "COALESCE(AVG(latency_ms), 0), "
                "COALESCE(SUM(cost_usd IS NULL AND status = 'ok' AND cached = 0), 0) "
                "FROM requests WHERE client = ?",
                (client,),
            ).fetchone()
            by_model = self._conn.execute(
                "SELECT provider, model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), "
                "COALESCE(SUM(cost_usd), 0) FROM requests "
                "WHERE client = ? AND status = 'ok' GROUP BY provider, model "
                "ORDER BY COUNT(*) DESC",
                (client,),
            ).fetchall()
        total, cached, errors, p_tok, c_tok, cost, saved, avg_latency, unpriced = row
        return {
            "client": client,
            "requests": total,
            "errors": errors,
            "cache_hits": cached,
            "cache_hit_rate": round(cached / total, 4) if total else 0.0,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "cost_usd": round(cost, 6),
            "estimated_saved_usd": round(saved, 6),
            "unpriced_requests": unpriced,
            "avg_latency_ms": round(avg_latency, 2),
            "by_model": [
                {
                    "provider": provider,
                    "model": model,
                    "requests": n,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cost_usd": round(c, 6),
                }
                for provider, model, n, pt, ct, c in by_model
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
