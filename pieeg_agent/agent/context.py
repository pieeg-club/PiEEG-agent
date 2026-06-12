"""Context window management for the reasoning loop.

This module implements sliding window compression with summarization to prevent
conversation history from saturating the context window during long EEG sessions.

The strategy:
  1. **Token counting** — rough estimate based on character count (~4 chars/token)
  2. **Sliding window** — when a threshold is reached, compress the oldest turns
     into a persistent state summary
  3. **Message compression** — remove detailed tool payloads while preserving
     semantic meaning

The compressor preserves:
  - Recent conversation history (last N turns, configurable)
  - A compressed summary of older turns
  - Tool call metadata (which tools were used, when, and for what purpose)

Heavy payloads like session data are replaced with semantic references that
can be retrieved via RAG if needed in future turns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.provider import Message

logger = logging.getLogger("pieeg.agent.context")

# Rough token estimation: ~4 characters per token (conservative for JSON)
CHARS_PER_TOKEN = 4

# Default thresholds
DEFAULT_MAX_TOKENS = 32000  # Trigger compression before hitting context limits
DEFAULT_KEEP_RECENT = 10    # Keep last N messages uncompressed


@dataclass
class CompressionStats:
    """Statistics from a compression operation."""
    
    original_tokens: int
    compressed_tokens: int
    messages_compressed: int
    messages_retained: int
    compression_ratio: float = field(init=False)
    
    def __post_init__(self):
        if self.original_tokens > 0:
            self.compression_ratio = self.compressed_tokens / self.original_tokens
        else:
            self.compression_ratio = 1.0


class ContextManager:
    """Manages conversation history with sliding window compression."""
    
    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ):
        """
        Args:
            max_tokens: Trigger compression when estimated tokens exceed this
            keep_recent: Number of recent messages to keep uncompressed
        """
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self._summary: str = ""
    
    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages.
        
        This is a rough approximation based on character count. It errs on the
        conservative side (overestimates) to trigger compression before hitting
        actual context limits.
        """
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.content)
            # Account for tool calls in assistant messages
            for tool_call in msg.tool_calls:
                total_chars += len(tool_call.name)
                total_chars += len(json.dumps(tool_call.arguments))
        
        # Add summary if it exists
        if self._summary:
            total_chars += len(self._summary)
        
        return total_chars // CHARS_PER_TOKEN
    
    def should_compress(self, messages: list[Message]) -> bool:
        """Check if compression should be triggered."""
        estimated = self.estimate_tokens(messages)
        return estimated > self.max_tokens
    
    def compress(self, messages: list[Message]) -> tuple[list[Message], CompressionStats]:
        """Compress conversation history using sliding window + summarization.
        
        Returns:
            Tuple of (compressed_messages, stats)
            
        The compressed history consists of:
          1. A synthetic "assistant" message containing the summary (if any older
             turns were compressed)
          2. The most recent `keep_recent` messages uncompressed
        """
        if len(messages) <= self.keep_recent:
            # Not enough messages to compress
            return messages, CompressionStats(
                original_tokens=self.estimate_tokens(messages),
                compressed_tokens=self.estimate_tokens(messages),
                messages_compressed=0,
                messages_retained=len(messages),
            )
        
        original_tokens = self.estimate_tokens(messages)
        
        # Split into old (to compress) and recent (to keep)
        old_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]
        
        # Compress old messages into a summary
        new_summary = self._compress_messages(old_messages)
        
        # Append to existing summary if we have one
        if self._summary:
            self._summary = self._merge_summaries(self._summary, new_summary)
        else:
            self._summary = new_summary
        
        # Build compressed history: summary + recent messages
        compressed = []
        if self._summary:
            compressed.append(
                Message(
                    role="assistant",
                    content=f"[Conversation summary up to this point]\n{self._summary}",
                )
            )
        compressed.extend(recent_messages)
        
        compressed_tokens = self.estimate_tokens(compressed)
        
        stats = CompressionStats(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            messages_compressed=len(old_messages),
            messages_retained=len(recent_messages),
        )
        
        logger.info(
            "Compressed %d messages (%.1f%% of original tokens)",
            stats.messages_compressed,
            stats.compression_ratio * 100,
        )
        
        return compressed, stats
    
    def _compress_messages(self, messages: list[Message]) -> str:
        """Compress a sequence of messages into a textual summary.
        
        The summary preserves:
          - Topics discussed
          - Tool calls made (but not full payloads)
          - Key insights or state changes
        """
        lines = []
        
        # Group messages by conversation turns (user → assistant → tools → assistant)
        current_turn: dict[str, Any] = {}
        
        for msg in messages:
            if msg.role == "user":
                # Start a new turn
                if current_turn:
                    lines.append(self._format_turn(current_turn))
                current_turn = {"user": msg.content, "tools": [], "response": ""}
            
            elif msg.role == "assistant":
                # Extract tool calls if present
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        current_turn.setdefault("tools", []).append({
                            "name": tc.name,
                            "args": self._compress_arguments(tc.arguments),
                        })
                # Save response text
                if msg.content:
                    current_turn["response"] = self._summarize_text(msg.content)
            
            elif msg.role == "tool":
                # Compress tool results
                try:
                    result = json.loads(msg.content)
                    compressed_result = self._compress_tool_result(result)
                    if current_turn.get("tools"):
                        current_turn["tools"][-1]["result"] = compressed_result
                except (json.JSONDecodeError, IndexError):
                    pass
        
        # Don't forget the last turn
        if current_turn:
            lines.append(self._format_turn(current_turn))
        
        return "\n".join(lines)
    
    def _format_turn(self, turn: dict) -> str:
        """Format a conversation turn for the summary."""
        parts = [f"User: {turn.get('user', '')[:100]}"]
        
        for tool in turn.get("tools", []):
            parts.append(
                f"  → {tool['name']}({tool.get('args', '')})"
                + (f" → {tool.get('result', '')}" if tool.get("result") else "")
            )
        
        if turn.get("response"):
            parts.append(f"Assistant: {turn['response']}")
        
        return "\n".join(parts)
    
    def _compress_arguments(self, arguments: dict) -> str:
        """Compress tool arguments for the summary."""
        # Keep only the most important keys
        important_keys = ["label", "name", "seconds", "a", "b", "band", "source"]
        compressed = {k: v for k, v in arguments.items() if k in important_keys}
        
        if not compressed:
            # If nothing important, just show key count
            return f"{len(arguments)} args"
        
        return ", ".join(f"{k}={v}" for k, v in compressed.items())
    
    def _compress_tool_result(self, result: Any) -> str:
        """Compress a tool result for the summary."""
        if isinstance(result, dict):
            # Extract key indicators
            if "error" in result:
                return f"error: {result['error'][:50]}"
            
            # For session data, just note what was retrieved
            if "band_powers" in result and "connectivity" in result:
                return "session_data"
            
            # For pattern results, summarize detection
            if "patterns" in result:
                active = [p["name"] for p in result["patterns"] if p.get("active")]
                return f"detected: {', '.join(active)}" if active else "none_active"
            
            # For lists, show count
            if "sessions" in result:
                return f"{result.get('count', 0)} sessions"
            
            # Generic: show status or key fields
            status = result.get("status", "")
            return status if status else "ok"
        
        return str(result)[:50]
    
    def _summarize_text(self, text: str, max_len: int = 100) -> str:
        """Summarize assistant response text."""
        if len(text) <= max_len:
            return text
        
        # Try to break at sentence boundary
        truncated = text[:max_len]
        last_period = truncated.rfind(".")
        if last_period > max_len // 2:
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _merge_summaries(self, old: str, new: str) -> str:
        """Merge an old summary with a new summary."""
        # For now, just concatenate with a separator
        # In a more sophisticated version, this could use an LLM to condense
        return f"{old}\n\n--- Additional turns ---\n{new}"
    
    def reset(self):
        """Clear the accumulated summary."""
        self._summary = ""


def compress_session_payload(session_data: dict) -> dict:
    """Compress a session data payload for storage in history.
    
    This is called before appending session data to tool result messages.
    Instead of the full payload, we return a compact reference that preserves
    semantic meaning but reduces token count.
    
    Args:
        session_data: Full session data from analyze_session
        
    Returns:
        Compressed payload with key metrics and a reference ID
    """
    if "error" in session_data:
        return session_data  # Don't compress errors
    
    # Extract only the high-level summary
    compressed = {
        "label": session_data.get("label", ""),
        "duration_s": session_data.get("duration_s"),
        "dominant_band": session_data.get("dominant_band"),
        "signal_quality": session_data.get("signal_quality", {}).get("mean"),
    }
    
    # Include indices (focus, relax, engagement) if present
    if "indices" in session_data:
        compressed["indices"] = session_data["indices"]
    
    # Artifact counts
    if "artifacts" in session_data:
        total_artifacts = sum(session_data["artifacts"].values())
        compressed["total_artifacts"] = total_artifacts
    
    # Note that full data is available if needed
    compressed["_note"] = (
        "Full session data available. Use list_sessions to see all recordings."
    )
    
    return compressed


def should_compress_tool_result(tool_name: str, result: Any) -> bool:
    """Determine if a tool result should be compressed before storage.
    
    Returns True for tools that return large payloads (sessions, spectra).
    """
    # Tools that return large payloads
    compress_these = {
        "analyze_session",
        "record_session",
        "analyze_spectrum",
    }
    
    return tool_name in compress_these and isinstance(result, dict)
