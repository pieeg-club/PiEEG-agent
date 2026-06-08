"""Provider-agnostic LLM layer.

Phase 0 ships the interface only (:mod:`pieeg_agent.llm.provider`). Concrete
adapters (Anthropic, OpenAI-compatible) land in later phases and will be
exposed through a ``get_provider(config)`` factory.
"""

from __future__ import annotations

from .provider import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
