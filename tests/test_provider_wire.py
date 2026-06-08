"""Wire-format round-trips for the LLM adapters (no network).

The adapters' value is correct translation between the normalized surface
(:class:`Message`, :class:`ToolSpec`, :class:`ToolCall`) and each provider's
JSON. These tests exercise the pure encode/decode helpers directly so they
run offline and pin the shapes both providers expect.
"""

import pieeg_agent.llm.anthropic as anth
import pieeg_agent.llm.openai_compat as oai
from pieeg_agent.llm.provider import Message, ToolCall, ToolSpec

TOOL = ToolSpec(
    name="get_neural_state",
    description="current state",
    input_schema={"type": "object", "properties": {}, "required": []},
)


# ── Anthropic ───────────────────────────────────────────────────────────────


def test_anthropic_encodes_tool_result_on_user_turn():
    msg = Message(role="tool", tool_call_id="abc", content='{"focus": 0.6}')
    wire = anth._encode_message(msg)
    assert wire["role"] == "user"
    block = wire["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "abc"
    assert block["content"] == '{"focus": 0.6}'


def test_anthropic_encodes_assistant_tool_use_blocks():
    msg = Message(
        role="assistant",
        content="let me check",
        tool_calls=[ToolCall(id="t1", name="get_neural_state", arguments={})],
    )
    wire = anth._encode_message(msg)
    assert wire["role"] == "assistant"
    types = [b["type"] for b in wire["content"]]
    assert types == ["text", "tool_use"]
    assert wire["content"][1]["id"] == "t1"


def test_anthropic_decodes_text_and_tool_use():
    data = {
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t9", "name": "get_band_powers",
             "input": {"per_channel": True}},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 5},
        "stop_reason": "tool_use",
    }
    resp = anth._decode_response(data)
    assert resp.text == "checking"
    assert resp.wants_tools
    call = resp.tool_calls[0]
    assert call.id == "t9"
    assert call.name == "get_band_powers"
    assert call.arguments == {"per_channel": True}
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 5


def test_anthropic_encodes_tool_spec():
    wire = anth._encode_tool(TOOL)
    assert wire["name"] == "get_neural_state"
    assert "input_schema" in wire


# ── OpenAI-compatible ───────────────────────────────────────────────────────


def test_openai_encodes_tool_message():
    msg = Message(role="tool", tool_call_id="abc", content='{"focus": 0.6}')
    wire = oai._encode_message(msg)
    assert wire == {
        "role": "tool",
        "tool_call_id": "abc",
        "content": '{"focus": 0.6}',
    }


def test_openai_encodes_assistant_tool_calls_with_json_string_args():
    msg = Message(
        role="assistant",
        content=None or "",
        tool_calls=[ToolCall(id="t1", name="get_band_powers",
                             arguments={"per_channel": True})],
    )
    wire = oai._encode_message(msg)
    assert wire["tool_calls"][0]["function"]["name"] == "get_band_powers"
    # Arguments must be a JSON *string*, not a dict, on the OpenAI wire.
    args = wire["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert '"per_channel": true' in args


def test_openai_decodes_tool_calls_parsing_string_arguments():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t7",
                            "type": "function",
                            "function": {
                                "name": "get_recent_events",
                                "arguments": '{"limit": 5}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3},
    }
    resp = oai._decode_response(data)
    assert resp.wants_tools
    call = resp.tool_calls[0]
    assert call.name == "get_recent_events"
    assert call.arguments == {"limit": 5}  # parsed from the JSON string
    assert resp.usage.input_tokens == 9


def test_openai_encodes_tool_spec_as_function():
    wire = oai._encode_tool(TOOL)
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "get_neural_state"
    assert "parameters" in wire["function"]


def test_openai_parse_arguments_is_forgiving():
    assert oai._parse_arguments(None) == {}
    assert oai._parse_arguments("") == {}
    assert oai._parse_arguments("not json") == {}
    assert oai._parse_arguments('{"a": 1}') == {"a": 1}
    assert oai._parse_arguments({"a": 1}) == {"a": 1}
