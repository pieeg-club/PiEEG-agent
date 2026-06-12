"""Provider factory — turn an :class:`AgentConfig` into a live adapter.

The agent core never names a concrete backend; it asks here for "the provider
this config selects". Selection is by the registry's ``kind`` field:

  * ``"anthropic"`` → :class:`AnthropicProvider` (native Messages API)
  * ``"openai"``    → :class:`OpenAICompatProvider` (OpenAI, Groq, Together,
                       Ollama, LM Studio, … — one wire format, many base URLs)

Both adapters speak plain HTTP, so there is nothing optional to import; the
only failure modes are a missing key or an unknown ``kind``, both reported as
:class:`ProviderError`.
"""

from __future__ import annotations

from ..config import PROVIDERS, AgentConfig
from .anthropic import AnthropicProvider
from .echo import EchoProvider
from .openai_compat import OpenAICompatProvider
from .provider import LLMProvider


class ProviderError(RuntimeError):
    """The selected provider could not be constructed (bad config/key/kind)."""


def get_provider(config: AgentConfig, *, timeout: float = 60.0) -> LLMProvider:
    """Build the LLM provider that ``config`` selects.

    Validates only the LLM-relevant settings (an ingestion-only run may carry
    an unrelated config without a key). Raises :class:`ProviderError` with an
    actionable message if the provider is unknown or a required key is absent.
    """
    spec = config.provider_spec
    if not spec:
        known = ", ".join(sorted(PROVIDERS))
        raise ProviderError(
            f"Unknown LLM provider {config.provider!r}. Known: {known}."
        )

    kind = config.provider_kind
    if config.needs_api_key and not config.has_api_key:
        env_key = spec.get("env_key")
        raise ProviderError(
            f"Provider {config.provider!r} needs an API key — set ${env_key}."
        )

    if kind == "anthropic":
        return AnthropicProvider(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout=timeout,
        )
    if kind == "openai":
        return OpenAICompatProvider(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout=timeout,
        )
    if kind == "echo":
        return EchoProvider(
            api_key="",  # not needed
            model=config.model,
            base_url="",
            timeout=0.0,
        )
    raise ProviderError(
        f"Provider {config.provider!r} has unsupported kind {kind!r}."
    )


def get_fallback_provider(config: AgentConfig, *, timeout: float = 60.0) -> LLMProvider | None:
    """Build the fallback LLM provider if configured, otherwise returns None.
    
    The fallback provider is used for resilience when the primary provider hits
    rate limits or errors. It's optional and configured via PIEEG_LLM_FALLBACK_PROVIDER.
    """
    if not config.fallback_provider:
        return None
    
    spec = PROVIDERS.get(config.fallback_provider)
    if not spec:
        # Don't fail if fallback is misconfigured, just log and continue without it
        return None
    
    kind = spec.get("kind")
    
    # Check for API key if needed
    needs_key = bool(spec.get("env_key"))
    if needs_key and not config.fallback_api_key:
        # Don't fail, just skip fallback
        return None
    
    try:
        if kind == "anthropic":
            return AnthropicProvider(
                api_key=config.fallback_api_key,
                model=config.fallback_model,
                base_url=config.fallback_base_url,
                timeout=timeout,
            )
        if kind == "openai":
            return OpenAICompatProvider(
                api_key=config.fallback_api_key,
                model=config.fallback_model,
                base_url=config.fallback_base_url,
                timeout=timeout,
            )
        if kind == "echo":
            return EchoProvider(
                api_key="",
                model=config.fallback_model,
                base_url="",
                timeout=0.0,
            )
    except Exception:
        # Don't fail if fallback provider can't be created
        return None
    
    return None
