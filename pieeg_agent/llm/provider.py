"""Provider-agnostic LLM interface.

This is the *contract* every backend adapter implements. It deliberately
hides vendor specifics (Anthropic blocks vs OpenAI messages, tool-call
encodings, streaming details) behind a small normalized surface so the agent
core never imports a vendor SDK directly.

Concrete adapters arrive in later phases:
  * ``anthropic.py``      → native Anthropic Messages API   (kind="anthropic")
  * ``openai_compat.py``  → OpenAI-compatible /chat endpoint (kind="openai")

The normalized tool-use loop is:
  1. agent calls :meth:`LLMProvider.complete` with the conversation + tools
  2. provider returns an :class:`LLMResponse` with ``text`` and/or ``tool_calls``
  3. if ``tool_calls`` is non-empty, the agent executes them and appends a
     ``tool`` :class:`Message` per call, then calls ``complete`` again
  4. loop ends when the provider returns text and no tool calls
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolSpec:
    """A tool the model may call. ``input_schema`` is JSON Schema."""

    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """One conversation turn in the normalized representation.

    * ``user`` / ``assistant`` use ``content`` (and ``tool_calls`` for an
      assistant turn that requested tools).
    * ``tool`` carries a tool *result*: ``tool_call_id`` references the
      originating :class:`ToolCall` and ``content`` is the result payload
      (already serialized to text).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Usage:
    """Token accounting for cost/observability (best-effort per provider)."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    """Normalized result of one :meth:`LLMProvider.complete` call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    raw: Any = field(default=None, repr=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Abstract base every backend adapter implements.

    Implementations are constructed with the resolved model/credentials and
    expose a single blocking :meth:`complete`. Keeping it synchronous lets
    the agent run providers in a worker thread without forcing an async SDK;
    streaming can be layered on later without changing this contract.
    """

    #: Registry id, e.g. "anthropic" or "groq".
    name: str = ""
    #: Concrete model id, e.g. "claude-sonnet-4-20250514".
    model: str = ""

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Run one completion turn and return a normalized response."""
        raise NotImplementedError

    @classmethod
    def available(cls) -> bool:
        """Whether this adapter's dependencies are importable."""
        return True
