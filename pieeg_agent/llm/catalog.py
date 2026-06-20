"""Read-side access to the committed OpenRouter model catalog.

The catalog is a snapshot of https://openrouter.ai/api/v1/models, trimmed by
``scripts/update_models.py`` and shipped as package data at
``pieeg_agent/data/openrouter_models.json``. Keeping a committed copy means the
model list is always available — no network call on the hot path — yet stays
current because the pre-commit hook refreshes it.

This module is intentionally tiny and dependency-free: it loads the JSON once,
caches it, and degrades gracefully to an empty catalog if the file is missing
or unreadable (the UI then falls back to free-text model entry).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "openrouter_models.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """Return the model catalog as ``{source, fetched_at, count, models}``.

    Never raises: a missing or corrupt snapshot yields an empty (but
    well-shaped) catalog so callers can rely on the keys existing.
    """
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"source": "", "fetched_at": None, "count": 0, "models": []}

    models = data.get("models")
    if not isinstance(models, list):
        models = []
    return {
        "source": data.get("source", ""),
        "fetched_at": data.get("fetched_at"),
        "count": len(models),
        "models": models,
    }


def models() -> list[dict]:
    """The trimmed model records (newest first)."""
    return load_catalog()["models"]
