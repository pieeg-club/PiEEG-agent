"""MCP mode advertises the existing tool schemas and dispatches calls.

No LSL stream and no LLM: a fake toolset stands in for the live cascade.
The test skips cleanly when the ``mcp`` extra is not installed.
"""

import asyncio

import pytest

from pieeg_agent.llm.provider import ToolSpec


class FakeTools:
    def __init__(self):
        self.calls = []

    def specs(self):
        return [
            ToolSpec(
                name="echo_state",
                description="Echo the arguments and a fixed focus score.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Free-text note from the host.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many samples.",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["note"],
                    "additionalProperties": False,
                },
            )
        ]

    def names(self):
        return ["echo_state"]

    def call(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        if name != "echo_state":
            return {"error": f"unknown tool {name!r}"}
        return {"focus": 0.42, "arguments": arguments or {}}


def test_mcp_server_lists_schema_and_calls_tool():
    pytest.importorskip("mcp")
    from pieeg_agent.mcp_mode import build_mcp_server

    tools = FakeTools()
    server = build_mcp_server(tools, instructions="test session")

    listed = asyncio.run(server.list_tools())
    tools_page = listed.tools if hasattr(listed, "tools") else listed
    advertised = next(tool for tool in tools_page if tool.name == "echo_state")
    assert "focus" in advertised.description or "Echo" in advertised.description
    schema = getattr(advertised, "parameters", None) or advertised.input_schema
    assert schema["type"] == "object"
    assert schema["properties"]["note"]["type"] == "string"
    assert "note" in schema.get("required", [])
    limit = schema["properties"]["limit"]
    limit_type = limit.get("type")
    if limit_type is None:
        limit_type = next(
            branch["type"] for branch in limit["anyOf"] if branch["type"] != "null"
        )
    assert limit_type == "integer"

    result = asyncio.run(server.call_tool("echo_state", {"note": "hi", "limit": 2}))
    assert tools.calls == [("echo_state", {"note": "hi", "limit": 2})]
    payload = getattr(result, "structured_content", None) or {}
    if not payload:
        text = result.content[0].text
        assert "0.42" in text
    else:
        assert payload["focus"] == 0.42
        assert payload["arguments"]["note"] == "hi"


def test_endpoint_and_client_config():
    from pieeg_agent.mcp_mode import client_config, endpoint_for

    assert endpoint_for("streamable-http", "127.0.0.1", 8765) == "http://127.0.0.1:8765/mcp"
    assert endpoint_for("sse", "127.0.0.1", 8765) == "http://127.0.0.1:8765/sse"
    assert endpoint_for("stdio", "127.0.0.1", 8765) is None
    snippet = client_config("streamable-http", "127.0.0.1", 8765)
    assert "http://127.0.0.1:8765/mcp" in snippet
