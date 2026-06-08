"""OpenAI-compatible /chat/completions adapter (``kind="openai"``).

One adapter serves every backend that speaks the OpenAI chat wire format:
OpenAI itself, Groq, Together, Ollama, LM Studio, … — they differ only by
``base_url`` and (optionally) an API key.

Translation notes versus the normalized surface:

* ``system`` becomes a leading ``{"role": "system"}`` message.
* An assistant turn that asked for tools carries a ``tool_calls`` array whose
  ``function.arguments`` is a **JSON string**; we serialise/parse accordingly.
* A tool result is a ``{"role": "tool", "tool_call_id": …}`` message.
* Local backends need no key, so an empty key is allowed here (unlike the
  Anthropic adapter) and simply omits the ``Authorization`` header.
"""

from __future__ import annotations

import json

from ._http import post_json
from .provider import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


class OpenAICompatProvider(LLMProvider):
    """Talks to ``{base_url}/chat/completions`` (OpenAI-style)."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.model = model
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
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
        wire: list[dict] = []
        if system:
            wire.append({"role": "system", "content": system})
        wire.extend(_encode_message(m) for m in messages)

        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": wire,
        }
        if tools:
            payload["tools"] = [_encode_tool(t) for t in tools]

        headers = {}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"

        data = post_json(self._url, payload, headers=headers, timeout=self._timeout)
        return _decode_response(data)

    @classmethod
    def available(cls) -> bool:
        return True


# ── wire encoding ───────────────────────────────────────────────────────────


def _encode_tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _encode_message(msg: Message) -> dict:
    if msg.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }

    if msg.role == "assistant" and msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in msg.tool_calls
            ],
        }

    return {"role": msg.role, "content": msg.content}


def _decode_response(data: dict) -> LLMResponse:
    choices = data.get("choices") or [{}]
    message = choices[0].get("message", {}) or {}

    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function", {}) or {}
        tool_calls.append(
            ToolCall(
                id=raw.get("id", ""),
                name=fn.get("name", ""),
                arguments=_parse_arguments(fn.get("arguments")),
            )
        )

    usage_raw = data.get("usage", {}) or {}
    usage = Usage(
        input_tokens=int(usage_raw.get("prompt_tokens", 0)),
        output_tokens=int(usage_raw.get("completion_tokens", 0)),
    )
    return LLMResponse(
        text=(message.get("content") or "").strip(),
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=choices[0].get("finish_reason", "") or "",
        raw=data,
    )


def _parse_arguments(arguments) -> dict:
    """Tool arguments arrive as a JSON string; be forgiving if empty/odd."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
