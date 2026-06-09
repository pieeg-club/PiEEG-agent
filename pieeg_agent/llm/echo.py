"""Echo/debug provider (``kind="echo"``).

A no-op backend for testing without API keys or network calls. Returns
canned responses that acknowledge the user's input and echo tool schemas
without making real requests.

Useful for:
  * Frontend development (test chat UI without burning credits)
  * System integration tests (validate tool wiring without LLM variance)
  * Debugging perception/action pipelines in isolation

The echo provider never calls tools — it just reports what it *would* call
if it were a real LLM, based on simple keyword matching in the user message.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

from .provider import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)


class EchoProvider(LLMProvider):
    """A debug provider that echoes input and pretends to understand tools."""

    name = "echo"

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "echo-debug-v1",
        base_url: str = "",
        timeout: float = 0.0,
    ):
        # Accept args for signature compatibility but ignore them
        self.model = model
        self._simulate_tools = True  # Can be toggled for different test modes

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Echo the last user message and optionally simulate a tool call."""
        
        # Find the last user message
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content
                break

        # Simple keyword-based tool simulation
        tool_calls = []
        if tools and self._simulate_tools:
            tool_calls = self._maybe_simulate_tool(last_user, tools)

        # Generate response text
        if tool_calls:
            tool_names = ", ".join(tc.name for tc in tool_calls)
            text = (
                f"[ECHO DEBUG MODE] Matched keywords in your query, would call: {tool_names}\n\n"
                f"💡 This is the echo provider — tool calls are simulated based on keywords. "
                f"The agent will execute them for real and show you actual EEG data.\n\n"
                f"Available tools: {len(tools)} ({', '.join(t.name for t in tools[:5])}{'...' if len(tools) > 5 else ''})"
            )
        else:
            text = self._generate_echo_response(last_user, messages, tools)

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(input_tokens=len(last_user.split()), output_tokens=len(text.split())),
            stop_reason="end_turn",
        )

    def stream_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        """Simulate streaming by yielding the response in chunks."""
        response = self.complete(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Emit text in chunks for more realistic streaming
        if response.text:
            words = response.text.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield StreamEvent(type="text", text=chunk)
        
        for call in response.tool_calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        
        yield StreamEvent(type="final", response=response)

    def _generate_echo_response(
        self, last_user: str, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> str:
        """Generate a debug response that acknowledges the input."""
        if not last_user:
            return "[ECHO DEBUG MODE] No user message to respond to."
        
        # Count turns
        turn_count = sum(1 for m in messages if m.role == "user")
        tool_count = len(tools) if tools else 0
        
        # Check for tool results in conversation (indicates system is working)
        has_tool_results = any(m.role == "tool" for m in messages)
        
        # Extract keywords
        keywords = re.findall(r'\b\w{4,}\b', last_user.lower())[:3]
        
        # Build informative response
        parts = [
            f"[ECHO DEBUG MODE — Turn {turn_count}]",
            f"📨 Your query: \"{last_user[:60]}{'...' if len(last_user) > 60 else ''}\"",
            f"🔧 Available tools: {tool_count}",
        ]
        
        if has_tool_results:
            parts.append("✅ Perception system is connected and returning data")
        else:
            parts.append("⚠️  No tool results yet — system may still be warming up")
        
        if keywords:
            parts.append(f"🔑 Detected keywords: {', '.join(keywords)}")
            parts.append(f"💡 Try queries with: 'state', 'focus', 'quality', 'channel', 'event', 'pattern'")
        
        parts.append("")
        parts.append(
            "ℹ️  Echo provider simulates LLM responses without API keys. "
            "Tool calls are real — you'll see actual EEG data if a stream is connected. "
            "For natural language understanding, use a real provider (anthropic, ollama, etc.)"
        )
        
        return "\n".join(parts)

    def _maybe_simulate_tool(
        self, user_message: str, tools: list[ToolSpec]
    ) -> list[ToolCall]:
        """Simulate tool calls based on keyword matching."""
        user_lower = user_message.lower()
        calls = []
        
        # Map common keywords to likely tool names
        tool_triggers = {
            "state": ["get_neural_state"],
            "focus": ["get_neural_state"],
            "relax": ["get_neural_state"],
            "quality": ["get_channel_quality"],
            "signal": ["get_channel_quality"],
            "channel": ["get_channel_quality", "get_band_powers"],
            "event": ["get_recent_events"],
            "band": ["get_band_powers"],
            "power": ["get_band_powers"],
            "pattern": ["list_patterns", "explain_pattern"],
            "train": ["train_pattern"],
            "connect": ["connectivity"],
            "session": ["list_sessions", "compare_sessions"],
            "record": ["record_segment", "record_session"],
        }
        
        # Find matching tools
        available_names = {t.name for t in tools}
        triggered = set()
        
        for keyword, tool_names in tool_triggers.items():
            if keyword in user_lower:
                for tname in tool_names:
                    if tname in available_names and tname not in triggered:
                        triggered.add(tname)
                        break  # Only trigger one per keyword
        
        # Generate fake tool calls
        for idx, tool_name in enumerate(sorted(triggered)):
            # Find the tool spec to generate appropriate fake args
            tool_spec = next((t for t in tools if t.name == tool_name), None)
            if not tool_spec:
                continue
            
            args = self._fake_args_for_tool(tool_spec)
            calls.append(
                ToolCall(
                    id=f"echo_call_{idx}",
                    name=tool_name,
                    arguments=args,
                )
            )
        
        return calls[:2]  # Limit to 2 tool calls max to avoid spam

    def _fake_args_for_tool(self, tool_spec: ToolSpec) -> dict:
        """Generate plausible fake arguments for a tool based on its schema."""
        schema = tool_spec.input_schema
        props = schema.get("properties", {})
        required = schema.get("required", [])
        
        args = {}
        for prop_name, prop_schema in props.items():
            # Only include required args for cleaner output
            if prop_name not in required:
                continue
            
            prop_type = prop_schema.get("type", "string")
            if prop_type == "boolean":
                args[prop_name] = False
            elif prop_type == "integer":
                args[prop_name] = 10
            elif prop_type == "number":
                args[prop_name] = 1.0
            elif prop_type == "array":
                args[prop_name] = []
            else:  # string
                args[prop_name] = "example"
        
        return args
