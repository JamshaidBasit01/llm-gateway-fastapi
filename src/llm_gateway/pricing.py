"""Cost calculation from a user-supplied price table.

Prices change often, so none are built in. Provide a JSON file (``PRICING_FILE``)::

    {"openai/your-model-id": {"input_per_1m": 1.0, "output_per_1m": 2.0}}

Unknown models have no cost (``None``) rather than a guessed one.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_pricing(path: str | Path | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, dict[str, float]] = {}
    for key, entry in data.items():
        if key.startswith("_"):  # allow "_comment" style keys
            continue
        try:
            inp, out = float(entry["input_per_1m"]), float(entry["output_per_1m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"pricing entry {key!r} needs numeric input_per_1m and output_per_1m"
            ) from exc
        if inp < 0 or out < 0:
            raise ValueError(f"pricing entry {key!r} has a negative price")
        table[key] = {"input_per_1m": inp, "output_per_1m": out}
    return table


def compute_cost(
    pricing: dict[str, dict[str, float]],
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """USD cost of one call, or ``None`` when the model has no price entry."""
    entry = pricing.get(f"{provider}/{model}")
    if entry is None:
        return None
    return (
        prompt_tokens * entry["input_per_1m"] + completion_tokens * entry["output_per_1m"]
    ) / 1_000_000
