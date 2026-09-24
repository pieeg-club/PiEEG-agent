"""MCP mode — let any MCP host drive the live PiEEG tools.

The built-in copilot is one agent. This module is the other door: the same
toolset (neural state, decode, docs, utility, and gated actuators when asked)
is advertised over the official MCP Python SDK so an external host can call
it. Claude, ChatGPT, Codex, Cursor, Grok, OpenClaw, Hermes, and Muse all
speak this protocol; they bring their own model. This process does not.

The public server class is ``MCPServer`` (SDK v2). Transport settings are
passed to :meth:`MCPServer.run`, not the constructor, as the v2 server docs
require. Tool input schemas are the JSON Schemas the copilot already
advertises, expressed as the function signatures ``MCPServer`` turns into
``tools/list`` schemas.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from . import __version__
from .llm.provider import ToolSpec

INSTRUCTIONS = """\
You are connected to a live PiEEG EEG session. Read the brain through the \
tools — do not invent focus, band power, quality, or pattern scores. \
Neural tools are read-only. Pattern tools record and score the live signal. \
Device tools, if present, are gated: they may preview an action instead of \
performing it. Prefer a tool call over a guess.\
"""

_TRANSPORTS = ("streamable-http", "sse", "stdio")


def missing_sdk_message() -> str:
    """What to print when the MCP extra is not installed."""
    return (
        "MCP mode needs the official MCP Python SDK v2.\n"
        "  pip install -e \".[mcp]\""
    )


def build_mcp_server(toolset, *, instructions: str = INSTRUCTIONS, version: str | None = None):
    """Register every tool in ``toolset`` on a v2 ``MCPServer``.

    ``toolset`` is anything with ``specs()`` and ``call(name, arguments)`` —
    a single tool collection or a :class:`~pieeg_agent.agent.tools.CombinedToolset`.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise ImportError(missing_sdk_message()) from exc

    server = MCPServer(
        "pieeg-agent",
        title="PiEEG Agent",
        instructions=instructions,
        version=version or __version__,
    )
    for spec in toolset.specs():
        server.add_tool(
            _handler(toolset, spec),
            name=spec.name,
            description=spec.description,
        )
    return server


def serve_toolset(
    toolset,
    *,
    transport: str = "streamable-http",
    host: str = "127.0.0.1",
    port: int = 8765,
    instructions: str = INSTRUCTIONS,
    version: str | None = None,
) -> None:
    """Serve ``toolset`` until the process is interrupted.

    ``streamable-http`` (default) is the endpoint current hosts use
    (``http://<host>:<port>/mcp``). ``sse`` is the older HTTP transport
    (``http://<host>:<port>/sse``). ``stdio`` is for hosts that spawn a
    local process; status must already have been written to stderr.
    """
    if transport not in _TRANSPORTS:
        raise ValueError(f"unknown MCP transport {transport!r}")
    server = build_mcp_server(toolset, instructions=instructions, version=version)
    if transport == "stdio":
        server.run(transport="stdio")
        return
    if transport == "sse":
        server.run(transport="sse", host=host, port=port)
        return
    # Stateless JSON responses: each host call is its own session, which is
    # what ChatGPT, Cursor, and Codex expect from a tool server that does
    # not push elicitation or sampling back to the client.
    server.run(
        transport="streamable-http",
        host=host,
        port=port,
        json_response=True,
        stateless_http=True,
    )


def endpoint_for(transport: str, host: str, port: int) -> str | None:
    """URL a remote host should put in its MCP config, or None for stdio."""
    if transport == "streamable-http":
        return f"http://{host}:{port}/mcp"
    if transport == "sse":
        return f"http://{host}:{port}/sse"
    return None


def client_config(transport: str, host: str, port: int) -> str:
    """A short config snippet for Claude, Cursor, Codex, and ChatGPT."""
    if transport == "stdio":
        payload = {
            "mcpServers": {
                "pieeg": {
                    "command": "pieeg-agent",
                    "args": ["mcp", "--transport", "stdio"],
                }
            }
        }
    else:
        payload = {
            "mcpServers": {
                "pieeg": {"url": endpoint_for(transport, host, port)}
            }
        }
    return json.dumps(payload, indent=2)


def _handler(toolset, spec: ToolSpec) -> Callable[..., dict]:
    """A sync function whose signature mirrors ``spec.input_schema``."""
    schema = spec.input_schema or {}
    properties: dict[str, dict] = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    def call_tool(**arguments: Any) -> dict:
        provided = {key: value for key, value in arguments.items() if value is not None}
        return toolset.call(spec.name, provided)

    call_tool.__name__ = spec.name
    call_tool.__doc__ = spec.description
    _install_signature(call_tool, properties, required)
    return call_tool


def _install_signature(fn: Callable[..., dict], properties: dict[str, dict], required: list[str]) -> None:
    """Attach the annotations ``MCPServer`` reads when it builds a tool schema."""
    required_set = set(required)
    # Required parameters first so the signature is valid if a host calls positionally.
    names = [name for name in required if name in properties]
    names.extend(name for name in properties if name not in required_set)

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for name in names:
        annotation = _annotation(properties[name])
        default = inspect.Parameter.empty if name in required_set else None
        if name not in required_set:
            annotation = _optional(annotation)
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
        annotations[name] = annotation
    annotations["return"] = dict
    fn.__annotations__ = annotations
    fn.__signature__ = inspect.Signature(parameters, return_annotation=dict)  # type: ignore[attr-defined]


def _annotation(prop: dict):
    """Map one JSON Schema property onto a type the SDK can advertise."""
    from typing import Annotated, Literal

    from pydantic import Field

    base = _python_type(prop, Literal)
    field: dict[str, Any] = {}
    description = prop.get("description") or ""
    if "items" in prop:
        detail = json.dumps(prop["items"], ensure_ascii=False)
        description = f"{description}\nItems schema: {detail}".strip()
    if description:
        field["description"] = description
    if "minimum" in prop:
        field["ge"] = prop["minimum"]
    if "maximum" in prop:
        field["le"] = prop["maximum"]
    if not field:
        return base
    return Annotated[base, Field(**field)]


def _optional(annotation):
    """Make a generated annotation accept an omitted argument.

    Metadata stays on the outside (``Annotated[T | None, Field(...)]``). That
    form is built one argument at a time: a star inside ``[]`` is a syntax
    error on 3.10, and ``Annotated.__class_getitem__`` is hidden on 3.13.
    """
    from typing import Annotated, Union, get_args, get_origin

    if get_origin(annotation) is not Annotated:
        return Union[annotation, None]
    base, *meta = get_args(annotation)
    wrapped = Union[base, None]
    for item in meta:
        wrapped = Annotated[wrapped, item]
    return wrapped


def _python_type(prop: dict, literal_type):
    enum = prop.get("enum")
    if enum and all(isinstance(item, str) for item in enum):
        return literal_type[enum[0]] if len(enum) == 1 else literal_type[tuple(enum)]
    kind = prop.get("type")
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        return list
    if kind == "object":
        return dict
    if kind == "string" or kind is None:
        return str
    return Any
