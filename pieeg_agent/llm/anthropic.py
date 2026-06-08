"""Native Anthropic Messages API adapter (``kind="anthropic"``).

Translates the provider-agnostic surface in :mod:`pieeg_agent.llm.provider`
to and from Anthropic's content-block wire format:

* ``system`` is a top-level string (not a message).
* Each turn's ``content`` is a list of typed blocks. We emit ``text`` and
  ``tool_use`` blocks for assistant turns and ``tool_result`` blocks (carried
  on a ``user`` turn) for tool outputs.
* Tools are declared with ``input_schema`` (already JSON Schema in
  :class:`ToolSpec`), so they pass straight through.

Only the small slice the agent loop needs is implemented; streaming and the
vendor SDK are intentionally avoided (see :mod:`pieeg_agent.llm._http`).
"""

from __future__ import annotations

from ._http import post_json
from .provider import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

_API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Talks to ``/v1/messages`` on api.anthropic.com (or a compatible base)."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("AnthropicProvider requires an API key.")
        self.model = model
        self._key = api_key
        self._url = base_url.rstrip("/") + "/v1/messages"
        self._timeout = timeout

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [_encode_message(m) for m in messages],
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_encode_tool(t) for t in tools]

        data = post_json(
            self._url,
            payload,
            headers={
                "x-api-key": self._key,
                "anthropic-version": _API_VERSION,
            },
            timeout=self._timeout,
        )
        return _decode_response(data)

    @classmethod
    def available(cls) -> bool:
        # Pure stdlib HTTP — no optional package to import.
        return True


# ── wire encoding ───────────────────────────────────────────────────────────


def _encode_tool(tool: ToolSpec) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _encode_message(msg: Message) -> dict:
    """Map one normalized :class:`Message` to an Anthropic message dict."""
    if msg.role == "tool":
        # Tool results ride on a ``user`` turn as tool_result blocks.
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
            ],
        }

    if msg.role == "assistant" and msg.tool_calls:
        blocks: list[dict] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for call in msg.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return {"role": "assistant", "content": blocks}

    # Plain user / assistant text turn.
    return {"role": msg.role, "content": msg.content}


def _decode_response(data: dict) -> LLMResponse:
    """Map an Anthropic response back to the normalized :class:`LLMResponse`."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}) or {},
                )
            )

    usage_raw = data.get("usage", {}) or {}
    usage = Usage(
        input_tokens=int(usage_raw.get("input_tokens", 0)),
        output_tokens=int(usage_raw.get("output_tokens", 0)),
    )
    return LLMResponse(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=data.get("stop_reason", "") or "",
        raw=data,
    )
