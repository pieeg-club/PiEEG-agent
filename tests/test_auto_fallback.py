"""Tests for automatic fallback model configuration."""

import os
import pytest
from pieeg_agent.config import AgentConfig, AUTO_FALLBACK_MODELS


def test_auto_fallback_anthropic_sonnet():
    """Sonnet should auto-fallback to Haiku."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    cfg = AgentConfig.from_env(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )
    
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-20250514"
    
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
        model="claude-sonnet-4-20250514",
    )
    
    # Explicit fallback wins
    assert cfg.fallback_provider == "groq"
    assert cfg.fallback_model == "llama-3.3-70b-versatile"  # groq default
    assert cfg.fallback_api_key == "groq-key"
    
    os.environ.pop("PIEEG_LLM_FALLBACK_PROVIDER", None)


def test_all_anthropic_models_have_fallback():
    """All major Anthropic models should have fallback configured."""
    major_models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
    ]
    for model in major_models:
        assert model in AUTO_FALLBACK_MODELS, f"{model} missing fallback"


def test_all_openai_models_have_fallback():
    """All major OpenAI models should have fallback configured."""
    major_models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
    for model in major_models:
        assert model in AUTO_FALLBACK_MODELS, f"{model} missing fallback"


def test_fallback_shares_same_api_key():
    """Auto-fallback should use the same API key as primary."""
    os.environ["ANTHROPIC_API_KEY"] = "shared-key"
    cfg = AgentConfig.from_env(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )
    
    assert cfg.api_key == "shared-key"
    assert cfg.fallback_api_key == "shared-key"
    assert cfg.provider == cfg.fallback_provider
