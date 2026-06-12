"""Provider-agnostic LLM layer.

The interface lives in :mod:`pieeg_agent.llm.provider`; concrete wire adapters
(Anthropic Messages, OpenAI-compatible chat completions) are selected from an
:class:`~pieeg_agent.config.AgentConfig` via :func:`get_provider`. Adapters
speak plain HTTP — no vendor SDKs — so importing this package is cheap and
side-effect free.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .factory import ProviderError, get_provider, get_fallback_provider
from .openai_compat import OpenAICompatProvider
from .provider import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "StreamEvent",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "AnthropicProvider",
    "OpenAICompatProvider",
    "get_provider",
    "get_fallback_provider",
    "ProviderError",
]
