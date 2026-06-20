"""Committed model catalog loader + OpenRouter provider wiring (no network)."""

from pieeg_agent.config import PROVIDERS, AgentConfig
from pieeg_agent.llm import OpenAICompatProvider
from pieeg_agent.llm.catalog import load_catalog, models
from pieeg_agent.llm.factory import get_provider


def test_catalog_loads_with_expected_shape():
    cat = load_catalog()
    assert set(cat) >= {"source", "fetched_at", "count", "models"}
    assert isinstance(cat["models"], list)
    assert cat["count"] == len(cat["models"])


def test_catalog_models_are_trimmed_records():
    items = models()
    assert items, "snapshot should ship with models"
    sample = items[0]
    assert "id" in sample and "name" in sample
    assert "supports_tools" in sample


def test_openrouter_is_registered_as_openai_kind():
    spec = PROVIDERS["openrouter"]
    assert spec["kind"] == "openai"
    assert spec["base_url"] == "https://openrouter.ai/api/v1"
    assert spec["env_key"] == "OPENROUTER_API_KEY"


def test_openrouter_builds_openai_adapter():
    cfg = AgentConfig.from_env(provider="openrouter", api_key="sk-or-test")
    provider = get_provider(cfg)
    assert isinstance(provider, OpenAICompatProvider)
    assert "openrouter.ai" in provider._url


def test_catalog_ids_resolve_against_openrouter():
    # Catalog IDs are OpenRouter-native, so they map 1:1 onto this provider.
    items = models()
    cfg = AgentConfig.from_env(
        provider="openrouter", model=items[0]["id"], api_key="sk-or-test"
    )
    assert cfg.model == items[0]["id"]
