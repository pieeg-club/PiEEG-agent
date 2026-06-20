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


def find_fallback(model_id: str, *, max_price_ratio: float = 0.5) -> str | None:
    """Find a cheaper fallback model from the same vendor.
    
    Given a model ID (e.g. "anthropic/claude-sonnet-4.6"), find a cheaper
    alternative from the same vendor that costs ≤ max_price_ratio of the
    original and supports the same capabilities (tools, reasoning).
    
    Returns the fallback model ID or None if no suitable alternative exists.
    This is used as a last resort when AUTO_FALLBACK_MODELS doesn't have an
    entry for the given model.
    
    Examples:
        >>> find_fallback("anthropic/claude-opus-4.8")
        "anthropic/claude-sonnet-4.6"
        >>> find_fallback("openai/gpt-5.5-pro")
        "openai/gpt-5.4-mini"
    """
    catalog = load_catalog()
    all_models = catalog.get("models", [])
    
    # Find the source model
    source = next((m for m in all_models if m["id"] == model_id), None)
    if not source:
        return None
    
    # Extract vendor (part before /)
    vendor = model_id.split("/")[0] if "/" in model_id else ""
    if not vendor:
        return None
    
    # Get source model's price (use completion price as the comparison metric)
    try:
        source_price = float(source.get("completion_price") or 0)
    except (TypeError, ValueError):
        return None
    
    if source_price == 0:  # Free models have no cheaper alternative
        return None
    
    max_price = source_price * max_price_ratio
    
    # Find candidates: same vendor, cheaper, supports same capabilities
    candidates = []
    for m in all_models:
        # Same vendor check
        if not m["id"].startswith(vendor + "/"):
            continue
        
        # Don't fallback to self
        if m["id"] == model_id:
            continue
        
        # Must support tools if source does
        if source.get("supports_tools") and not m.get("supports_tools"):
            continue
        
        # Check price
        try:
            price = float(m.get("completion_price") or 0)
        except (TypeError, ValueError):
            continue
        
        if price == 0 or price > max_price:
            continue
        
        candidates.append((m["id"], price))
    
    # Return cheapest candidate
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    return None


def get_recommended_model(vendor: str, tier: str = "balanced") -> str | None:
    """Get a recommended model ID from the catalog.
    
    This is the SINGLE SOURCE OF TRUTH for default models. All hardcoded
    model IDs should reference this instead.
    
    Args:
        vendor: "anthropic", "openai", "google", etc.
        tier: "flagship" (best/expensive), "balanced" (mid-tier), "budget" (cheapest)
    
    Returns:
        Model ID or None if vendor has no models in catalog.
        Prefers -latest aliases when available.
    """
    all_models = models()
    
    # Try -latest alias first (auto-tracking)
    if tier == "flagship":
        latest = f"~{vendor}/claude-sonnet-latest" if vendor == "anthropic" else f"~{vendor}/gpt-latest"
        if any(m["id"] == latest for m in all_models):
            return latest
    elif tier == "budget":
        latest = f"~{vendor}/claude-haiku-latest" if vendor == "anthropic" else f"~{vendor}/gpt-mini-latest"
        if any(m["id"] == latest for m in all_models):
            return latest
    
    # Fallback: find by price and vendor
    vendor_models = [m for m in all_models if m["id"].startswith(vendor + "/") and m.get("supports_tools")]
    if not vendor_models:
        return None
    
    # Sort by price (completion_price)
    with_price = []
    for m in vendor_models:
        try:
            price = float(m.get("completion_price") or 0)
            if price > 0:
                with_price.append((m["id"], price))
        except (TypeError, ValueError):
            continue
    
    if not with_price:
        return vendor_models[0]["id"]  # Fallback to first model
    
    with_price.sort(key=lambda x: x[1])
    
    if tier == "flagship":
        # Most expensive (top 10%)
        idx = max(0, len(with_price) - 1)
    elif tier == "budget":
        # Cheapest
        idx = 0
    else:  # balanced
        # Middle 40-60% range
        idx = len(with_price) // 2
    
    return with_price[idx][0]
