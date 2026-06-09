"""Streaming surface: native SSE adapters, base fallback, copilot stream.

These pin the token-streaming contract without any network. Each adapter's
``stream_complete`` is exercised by monkeypatching ``post_sse`` to replay a
hand-built sequence of provider chunks, so we assert that:

* text deltas arrive in order as ``text`` events,
* incrementally-streamed tool-call arguments are reassembled into one
  ``tool_call`` event with parsed args,
* a terminal ``final`` event carries the full text, tool calls, usage and
  stop reason.

We also pin the base-class fallback (providers that only implement
``complete`` still stream) and :meth:`Copilot.ask_stream`'s event vocabulary.
"""

import pieeg_agent.llm.anthropic as anth
import pieeg_agent.llm.openai_compat as oai
from pieeg_agent.agent.copilot import Copilot
from pieeg_agent.agent.tools import NeuralTools
from pieeg_agent.llm.provider import (
    LLMProvider,
    LLMResponse,
    StreamEvent,
    ToolCall,
    Usage,
)
from tests.test_tools import FakeCascade, mk_state


def _collect(stream):
    return list(stream)


# ── Anthropic native SSE ─────────────────────────────────────────────────────


def test_anthropic_stream_reassembles_text_and_tool_call(monkeypatch):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "t9",
                           "name": "get_band_powers"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"per_'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": 'channel": true}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    monkeypatch.setattr(anth, "post_sse", lambda *a, **k: iter(events))

    provider = anth.AnthropicProvider(api_key="x", model="m")
    out = _collect(provider.stream_complete(system="", messages=[]))

    texts = [e.text for e in out if e.type == "text"]
    assert texts == ["Hel", "lo"]

    calls = [e.tool_call for e in out if e.type == "tool_call"]
    assert len(calls) == 1
    assert calls[0].name == "get_band_powers"
    assert calls[0].arguments == {"per_channel": True}

    final = out[-1]
    assert final.type == "final"
    assert final.response.text == "Hello"
    assert final.response.stop_reason == "tool_use"
    assert final.response.usage.input_tokens == 12
    assert final.response.usage.output_tokens == 5
    assert final.response.tool_calls[0].name == "get_band_powers"


def test_anthropic_stream_text_only(monkeypatch):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "calm"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    monkeypatch.setattr(anth, "post_sse", lambda *a, **k: iter(events))

    provider = anth.AnthropicProvider(api_key="x", model="m")
    out = _collect(provider.stream_complete(system="", messages=[]))
    assert [e.type for e in out] == ["text", "final"]
    assert out[-1].response.text == "calm"
    assert out[-1].response.tool_calls == []


# ── OpenAI-compatible native SSE ─────────────────────────────────────────────


def test_openai_stream_reassembles_text_and_tool_call(monkeypatch):
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "t7", "type": "function",
             "function": {"name": "get_recent_events", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"limit": '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "5}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"usage": {"prompt_tokens": 9, "completion_tokens": 3}, "choices": []},
    ]
    monkeypatch.setattr(oai, "post_sse", lambda *a, **k: iter(chunks))

    provider = oai.OpenAICompatProvider(api_key="", model="m")
    out = _collect(provider.stream_complete(system="", messages=[]))

    texts = [e.text for e in out if e.type == "text"]
    assert texts == ["Hel", "lo"]

    calls = [e.tool_call for e in out if e.type == "tool_call"]
    assert len(calls) == 1
    assert calls[0].id == "t7"
    assert calls[0].name == "get_recent_events"
    assert calls[0].arguments == {"limit": 5}

    final = out[-1]
    assert final.type == "final"
    assert final.response.text == "Hello"
    assert final.response.stop_reason == "tool_calls"
    assert final.response.usage.input_tokens == 9
    assert final.response.usage.output_tokens == 3


# ── base-class fallback ──────────────────────────────────────────────────────


class _BlockingOnly(LLMProvider):
    """Implements only ``complete`` — must still stream via the default."""

    name = "blocking"
    model = "b1"

    def __init__(self, response):
        self._response = response

    def complete(self, **_kw):
        return self._response


def test_base_stream_fallback_replays_complete():
    response = LLMResponse(
        text="hi",
        tool_calls=[ToolCall(id="t1", name="get_neural_state", arguments={})],
        usage=Usage(4, 2),
        stop_reason="tool_use",
    )
    provider = _BlockingOnly(response)
    out = _collect(provider.stream_complete(system="s", messages=[]))

    assert [e.type for e in out] == ["text", "tool_call", "final"]
    assert out[0].text == "hi"
    assert out[1].tool_call.name == "get_neural_state"
    assert out[-1].response is response


# ── copilot streaming surface ────────────────────────────────────────────────


class ScriptedStream(LLMProvider):
    """Replays queued StreamEvent sequences, one per ``stream_complete``."""

    name = "scripted-stream"
    model = "s1"

    def __init__(self, scripts):
        self._scripts = list(scripts)

    def complete(self, **_kw):  # pragma: no cover - unused in streaming path
        raise AssertionError("streaming path should not call complete")

    def stream_complete(self, **_kw):
        script = self._scripts.pop(0)
        yield from script


def _text_then(response):
    if response.text:
        yield StreamEvent(type="text", text=response.text)
    for call in response.tool_calls:
        yield StreamEvent(type="tool_call", tool_call=call)
    yield StreamEvent(type="final", response=response)


def test_copilot_ask_stream_emits_tokens_tool_and_done():
    tool_turn = LLMResponse(
        tool_calls=[ToolCall(id="t1", name="get_neural_state", arguments={})],
        usage=Usage(10, 2),
    )
    answer_turn = LLMResponse(text="Your focus is 0.42.", usage=Usage(8, 4))
    provider = ScriptedStream([_text_then(tool_turn), _text_then(answer_turn)])
    copilot = Copilot(provider, NeuralTools(FakeCascade(state=mk_state())))

    events = _collect(copilot.ask_stream("what is my focus?"))
    types = [e.type for e in events]
    assert types == ["tool_start", "tool_result", "token", "done"]

    assert events[0].name == "get_neural_state"
    assert "focus" in events[1].result  # tool result payload
    assert events[2].text == "Your focus is 0.42."

    done = events[-1]
    assert done.text == "Your focus is 0.42."
    assert done.tool_calls == ["get_neural_state"]
    assert done.usage.input_tokens == 18
    assert done.usage.output_tokens == 6
    assert done.iterations == 2

    # History matches the blocking loop's shape.
    roles = [m.role for m in copilot.history]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_copilot_ask_matches_stream_done():
    answer = LLMResponse(text="You look calm.", usage=Usage(5, 1))
    provider = ScriptedStream([_text_then(answer)])
    copilot = Copilot(provider, NeuralTools(FakeCascade(state=mk_state())))
    result = copilot.ask("how am I?")
    assert result.text == "You look calm."
    assert result.tool_calls == []
    assert result.iterations == 1
