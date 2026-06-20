#!/usr/bin/env python3
"""Refresh the committed OpenRouter model catalog.

OpenRouter (https://openrouter.ai/api/v1/models) tracks every model it serves
— always current, never deprecated under your feet. We snapshot a *trimmed*
copy into ``pieeg_agent/data/openrouter_models.json`` so the app ships with a
known-good catalog and never depends on the network at runtime.

The snapshot is refreshed automatically by the ``.githooks/pre-commit`` hook
(install once with ``scripts/install-hooks.sh`` / ``.cmd``) and can be run by
hand at any time::

    python scripts/update_models.py

Design notes:
  * Standard-library only (``urllib``) — no new dependencies, runs anywhere.
  * Offline-tolerant: a network failure leaves the existing snapshot intact and
    exits 0 so it never blocks a commit.
  * Deterministic output (sorted, fixed indentation) so diffs stay readable.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://openrouter.ai/api/v1/models"
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "pieeg_agent"
    / "data"
    / "openrouter_models.json"
)
TIMEOUT = 20.0


def _price(value: object) -> str | None:
    """Normalise a price string; drop unknown/negative (router) prices."""
    if value in (None, "", "-1"):
        return None
    try:
        if float(value) < 0:  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        return None
    return str(value)


def trim(model: dict) -> dict | None:
    """Keep only the fields the UI and agent need; skip non-text models."""
    mid = model.get("id")
    if not mid:
        return None

    arch = model.get("architecture") or {}
    out_mods = arch.get("output_modalities") or []
    # The agent consumes text; skip image/audio-only generators.
    if out_mods and "text" not in out_mods:
        return None

    pricing = model.get("pricing") or {}
    params = model.get("supported_parameters") or []

    return {
        "id": mid,
        "name": model.get("name") or mid,
        "context_length": model.get("context_length"),
        "prompt_price": _price(pricing.get("prompt")),
        "completion_price": _price(pricing.get("completion")),
        "supports_tools": "tools" in params,
        "supports_reasoning": "reasoning" in params,
        "created": model.get("created"),
    }


def fetch() -> list[dict]:
    req = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "pieeg-agent/update_models"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.load(resp)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("Unexpected response shape from OpenRouter /models")
    return data


def main() -> int:
    try:
        raw = fetch()
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        # Offline or API hiccup: keep whatever snapshot we already have.
        print(f"update_models: skipped (could not reach OpenRouter: {exc})")
        return 0

    models = [m for m in (trim(x) for x in raw) if m is not None]
    # Newest first so the picker leads with current models.
    models.sort(key=lambda m: (-(m.get("created") or 0), m["id"]))

    snapshot = {
        "source": SOURCE_URL,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(models),
        "models": models,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"update_models: wrote {len(models)} models → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
