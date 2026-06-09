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

import json
from typing import Iterator

from ._http import post_json, post_sse
from .provider import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamEvent,
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

    def stream_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [_encode_message(m) for m in messages],
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_encode_tool(t) for t in tools]

        events = post_sse(
            self._url,
            payload,
            headers={
                "x-api-key": self._key,
                "anthropic-version": _API_VERSION,
            },
            timeout=self._timeout,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        blocks: dict[int, dict] = {}  # content-block index → assembly state
        usage = Usage()
        stop_reason = ""

        for ev in events:
            etype = ev.get("type")
            if etype == "message_start":
                u = (ev.get("message", {}) or {}).get("usage", {}) or {}
                usage.input_tokens = int(u.get("input_tokens", 0))
            elif etype == "content_block_start":
                idx = int(ev.get("index", 0))
                cb = ev.get("content_block", {}) or {}
                blocks[idx] = {
                    "type": cb.get("type"),
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "json": [],
                }
            elif etype == "content_block_delta":
                idx = int(ev.get("index", 0))
                delta = ev.get("delta", {}) or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    chunk = delta.get("text", "")
                    if chunk:
                        text_parts.append(chunk)
                        yield StreamEvent(type="text", text=chunk)
                elif dtype == "input_json_delta":
                    blk = blocks.setdefault(idx, {"type": "tool_use", "json": []})
                    blk["json"].append(delta.get("partial_json", ""))
            elif etype == "content_block_stop":
                idx = int(ev.get("index", 0))
                blk = blocks.get(idx)
                if blk and blk.get("type") == "tool_use":
                    call = ToolCall(
                        id=blk.get("id", ""),
                        name=blk.get("name", ""),
                        arguments=_parse_json("".join(blk.get("json", []))),
                    )
                    tool_calls.append(call)
                    yield StreamEvent(type="tool_call", tool_call=call)
            elif etype == "message_delta":
                delta = ev.get("delta", {}) or {}
                stop_reason = delta.get("stop_reason") or stop_reason
                u = ev.get("usage", {}) or {}
                if "output_tokens" in u:
                    usage.output_tokens = int(u.get("output_tokens", 0))
            elif etype == "message_stop":
                break

        yield StreamEvent(
            type="final",
            response=LLMResponse(
                text="".join(text_parts).strip(),
                tool_calls=tool_calls,
                usage=usage,
                stop_reason=stop_reason,
            ),
        )

    @classmethod
    def available(cls) -> bool:
        # Pure stdlib HTTP — no optional package to import.
        return True


# ── wire encoding ───────────────────────────────────────────────────────────


def _parse_json(text: str) -> dict:
    """Parse accumulated ``input_json_delta`` text into a tool-args dict.

    Empty input (a tool called with no arguments) yields ``{}``; anything that
    is not a JSON object is treated as no arguments rather than raising.
    """
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
