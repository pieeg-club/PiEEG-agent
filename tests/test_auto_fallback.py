"""Tests for automatic fallback model configuration."""

import os
import pytest
from pieeg_agent.config import AgentConfig, AUTO_FALLBACK_MODELS


# Env vars these tests read or mutate. Snapshot and restore them around every
# test so process-wide state doesn't leak between tests and cause
# order-dependent failures.
_FALLBACK_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "PIEEG_LLM_PROVIDER",
    "PIEEG_LLM_MODEL",
    "PIEEG_LLM_FALLBACK_PROVIDER",
    "PIEEG_LLM_FALLBACK_MODEL",
)


@pytest.fixture(autouse=True)
def _restore_fallback_env():
    saved = {k: os.environ.get(k) for k in _FALLBACK_ENV_KEYS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v



def test_auto_fallback_anthropic_sonnet():
    """Sonnet should auto-fallback to Haiku."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    cfg = AgentConfig.from_env(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
    )
    
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-3-5-sonnet-20241022"
    
    # Auto-fallback to Haiku
    assert cfg.fallback_provider == "anthropic"
    assert cfg.fallback_model == "claude-3-5-haiku-20241022"
    assert cfg.fallback_api_key == "test-key"


def test_auto_fallback_openai_gpt4o():
    """GPT-4o should auto-fallback to GPT-4o-mini."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    cfg = AgentConfig.from_env(
        provider="openai",
        model="gpt-4o",
    )
    
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"
    
    # Auto-fallback to mini
    assert cfg.fallback_provider == "openai"
    assert cfg.fallback_model == "gpt-4o-mini"
    assert cfg.fallback_api_key == "test-key"


def test_auto_fallback_groq_70b():
    """Groq 70B should auto-fallback to 8B."""
    os.environ["GROQ_API_KEY"] = "test-key"
    cfg = AgentConfig.from_env(
        provider="groq",
        model="llama-3.3-70b-versatile",
    )
    
    assert cfg.provider == "groq"
    assert cfg.model == "llama-3.3-70b-versatile"
    
    # Auto-fallback to 8B
    assert cfg.fallback_provider == "groq"
    assert cfg.fallback_model == "llama-3.1-8b-instant"
    assert cfg.fallback_api_key == "test-key"


def test_no_auto_fallback_for_unknown_model():
    """Models not in AUTO_FALLBACK_MODELS should not get automatic fallback."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    cfg = AgentConfig.from_env(
        provider="anthropic",
        model="claude-custom-model",  # Not in fallback map
    )
    
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-custom-model"
    
    # No auto-fallback
    assert cfg.fallback_provider == ""
    assert cfg.fallback_model == ""


def test_explicit_fallback_overrides_auto():
    """Explicit fallback config should override automatic fallback."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["GROQ_API_KEY"] = "groq-key"
    os.environ["PIEEG_LLM_FALLBACK_PROVIDER"] = "groq"
    
    cfg = AgentConfig.from_env(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
    )
    
    # Explicit fallback wins
    assert cfg.fallback_provider == "groq"
    assert cfg.fallback_model == "llama-3.3-70b-versatile"  # groq default
    assert cfg.fallback_api_key == "groq-key"


def test_all_anthropic_models_have_fallback():
    """All major Anthropic models should have fallback configured."""
    major_models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
    ]
    for model in major_models:
        assert model in AUTO_FALLBACK_MODELS["anthropic"], f"{model} missing fallback"


def test_all_openai_models_have_fallback():
    """All major OpenAI models should have fallback configured."""
    major_models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
    for model in major_models:
        assert model in AUTO_FALLBACK_MODELS["openai"], f"{model} missing fallback"


def test_fallback_shares_same_api_key():
    """Auto-fallback should use the same API key as primary."""
    os.environ["ANTHROPIC_API_KEY"] = "shared-key"
    cfg = AgentConfig.from_env(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
    )
    
    assert cfg.api_key == "shared-key"
    assert cfg.fallback_api_key == "shared-key"
    assert cfg.provider == cfg.fallback_provider
