"""Provider factory selection and validation (no network)."""

import pytest

from pieeg_agent.config import AgentConfig
from pieeg_agent.llm import AnthropicProvider, OpenAICompatProvider
from pieeg_agent.llm.factory import ProviderError, get_provider


def test_anthropic_kind_builds_anthropic_adapter():
    cfg = AgentConfig.from_env(provider="anthropic", api_key="sk-test")
    provider = get_provider(cfg)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == cfg.model


def test_openai_kind_builds_openai_adapter():
    cfg = AgentConfig.from_env(provider="openai", api_key="sk-test")
    provider = get_provider(cfg)
    assert isinstance(provider, OpenAICompatProvider)


def test_groq_uses_openai_adapter_with_its_base_url():
    cfg = AgentConfig.from_env(provider="groq", api_key="gsk-test")
    provider = get_provider(cfg)
    assert isinstance(provider, OpenAICompatProvider)
    assert "groq.com" in provider._url


def test_local_provider_needs_no_key():
    cfg = AgentConfig.from_env(provider="ollama")  # env_key=""
    provider = get_provider(cfg)
    assert isinstance(provider, OpenAICompatProvider)


def test_missing_key_raises_provider_error():
    cfg = AgentConfig(provider="anthropic", api_key="")
    cfg._resolve_provider_defaults()
    with pytest.raises(ProviderError, match="API key"):
        get_provider(cfg)


def test_unknown_provider_raises_provider_error():
    cfg = AgentConfig(provider="does-not-exist")
    with pytest.raises(ProviderError, match="Unknown LLM provider"):
        get_provider(cfg)
