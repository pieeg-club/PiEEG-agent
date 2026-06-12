"""Tests for context window management and session RAG.

These tests verify that:
  1. Token counting is conservative (overestimates to stay safe)
  2. Compression triggers at the correct threshold
  3. Message compression preserves semantic meaning
  4. Session RAG enables efficient retrieval
"""

import json
from pathlib import Path

import pytest

from pieeg_agent.agent.context import (
    ContextManager,
    CompressionStats,
    compress_session_payload,
    should_compress_tool_result,
)
from pieeg_agent.agent.session_rag import SessionRAG, SessionIndex
from pieeg_agent.llm.provider import Message, ToolCall


class TestContextManager:
    """Test suite for ContextManager."""
    
    def test_estimate_tokens_simple(self):
        """Token estimation should be conservative."""
        ctx = ContextManager()
        messages = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
        ]
        
        # Estimate should be roughly chars / 4
        estimated = ctx.estimate_tokens(messages)
        total_chars = len("Hello, how are you?") + len("I'm doing well, thank you!")
        expected = total_chars // 4
        
        assert estimated == expected
    
    def test_estimate_tokens_with_tool_calls(self):
        """Tool calls should be included in token count."""
        ctx = ContextManager()
        messages = [
            Message(role="user", content="Analyze spectrum"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="analyze_spectrum",
                        arguments={"band": "alpha"},
                    )
                ],
            ),
        ]
        
        estimated = ctx.estimate_tokens(messages)
        
        # Should count user message + tool name + arguments
        assert estimated > 0
    
    def test_should_compress_threshold(self):
        """Compression should trigger when threshold is exceeded."""
        ctx = ContextManager(max_tokens=100)
        
        # Create messages that exceed threshold
        large_content = "x" * 500  # ~125 tokens
        messages = [Message(role="user", content=large_content)]
        
        assert ctx.should_compress(messages)
        
        # Small messages should not trigger
        small_messages = [Message(role="user", content="hi")]
        assert not ctx.should_compress(small_messages)
    
    def test_compress_preserves_recent(self):
        """Compression should keep recent messages intact."""
        ctx = ContextManager(keep_recent=3)
        
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(10)
        ]
        
        compressed, stats = ctx.compress(messages)
        
        # Should have summary + 3 recent messages
        assert len(compressed) == 4  # 1 summary + 3 recent
        assert compressed[0].role == "assistant"  # Summary
        assert compressed[0].content.startswith("[Conversation summary")
        
        # Recent messages should be unchanged
        assert compressed[-1].content == "Message 9"
        assert compressed[-2].content == "Message 8"
        assert compressed[-3].content == "Message 7"
    
    def test_compress_stats(self):
        """Compression should provide accurate statistics."""
        ctx = ContextManager(keep_recent=2)
        
        # Create realistic messages with large tool payloads
        large_payload = json.dumps({
            "band_powers": {f"channel_{i}": {"mean": i * 0.1, "std": 0.05} for i in range(50)},
            "connectivity": [[0.5] * 50 for _ in range(50)],
            "extra_data": "x" * 500,
        })
        
        messages = [
            Message(role="user", content="Analyze session"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="analyze_session", arguments={"label": "rest"})
            ]),
            Message(role="tool", tool_call_id="1", content=large_payload),
            Message(role="assistant", content="Session analyzed."),
            Message(role="user", content="What's my current state?"),
        ]
        
        compressed, stats = ctx.compress(messages)
        
        assert stats.messages_compressed == 3  # First 3 messages
        assert stats.messages_retained == 2    # Last 2 messages
        assert stats.original_tokens > stats.compressed_tokens
        assert 0 < stats.compression_ratio < 1
    
    def test_compress_tool_results(self):
        """Tool result compression should preserve key info."""
        ctx = ContextManager(keep_recent=1)
        
        # Create a conversation with a tool call
        messages = [
            Message(role="user", content="Record a session"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="record_session",
                        arguments={"label": "rest", "seconds": 20},
                    )
                ],
            ),
            Message(
                role="tool",
                tool_call_id="1",
                content=json.dumps({
                    "status": "recorded",
                    "label": "rest",
                    "n_frames": 5000,
                    "duration_s": 20.0,
                    "dominant_band": "alpha",
                }),
            ),
            Message(role="assistant", content="Session recorded successfully."),
            Message(role="user", content="What's my current state?"),
        ]
        
        compressed, stats = ctx.compress(messages)
        
        # Should compress first 4 messages, keep last 1
        assert len(compressed) == 2  # Summary + recent user message
        
        # Summary should mention the tool call
        summary = compressed[0].content
        assert "record_session" in summary.lower()
    
    def test_reset(self):
        """Reset should clear accumulated summary."""
        ctx = ContextManager(keep_recent=1)
        
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(5)
        ]
        
        # Compress to build summary
        ctx.compress(messages)
        assert ctx._summary != ""
        
        # Reset should clear it
        ctx.reset()
        assert ctx._summary == ""


class TestToolResultCompression:
    """Test suite for tool result compression helpers."""
    
    def test_should_compress_session_tools(self):
        """Session-related tools should be marked for compression."""
        assert should_compress_tool_result("analyze_session", {"data": "..."})
        assert should_compress_tool_result("record_session", {"data": "..."})
        assert should_compress_tool_result("analyze_spectrum", {"data": "..."})
    
    def test_should_not_compress_small_tools(self):
        """Small tools should not be compressed."""
        assert not should_compress_tool_result("get_neural_state", {})
        assert not should_compress_tool_result("list_sessions", {})
    
    def test_compress_session_payload(self):
        """Session payload compression should preserve key metrics."""
        full_payload = {
            "label": "rest",
            "duration_s": 20.0,
            "dominant_band": "alpha",
            "timestamp": 1234567890,
            "signal_quality": {"mean": 0.85, "std": 0.1},
            "band_powers": {
                "delta": {"mean": 2.3, "std": 0.5},
                "theta": {"mean": 1.8, "std": 0.4},
                "alpha": {"mean": 3.2, "std": 0.6},
                "beta": {"mean": 1.5, "std": 0.3},
                "gamma": {"mean": 0.8, "std": 0.2},
            },
            "connectivity": {
                "alpha": [[1.0, 0.5], [0.5, 1.0]],
            },
            "indices": {
                "focus": 0.65,
                "relax": 0.72,
                "engagement": 0.58,
            },
            "artifacts": {
                "blink": 5,
                "jaw_clench": 2,
            },
        }
        
        compressed = compress_session_payload(full_payload)
        
        # Should preserve key fields
        assert compressed["label"] == "rest"
        assert compressed["duration_s"] == 20.0
        assert compressed["dominant_band"] == "alpha"
        assert compressed["signal_quality"] == 0.85
        
        # Should preserve indices
        assert compressed["indices"] == full_payload["indices"]
        
        # Should compress artifacts to count
        assert compressed["total_artifacts"] == 7
        
        # Should not include full band_powers or connectivity
        assert "band_powers" not in compressed
        assert "connectivity" not in compressed
        
        # Should include retrieval hint
        assert "_note" in compressed
    
    def test_compress_session_payload_preserves_errors(self):
        """Error payloads should not be compressed."""
        error_payload = {
            "error": "Session not found",
            "known": ["rest", "focus"],
        }
        
        compressed = compress_session_payload(error_payload)
        
        # Should be unchanged
        assert compressed == error_payload


class TestSessionRAG:
    """Test suite for SessionRAG."""
    
    def test_index_session(self):
        """Indexing should build searchable metadata."""
        rag = SessionRAG()
        
        session_data = {
            "label": "rest",
            "timestamp": 1234567890,
            "duration_s": 20.0,
            "dominant_band": "alpha",
            "signal_quality": {"mean": 0.85},
            "artifacts": {"blink": 3},
            "indices": {
                "focus": 0.45,
                "relax": 0.82,
                "engagement": 0.50,
            },
        }
        
        rag.index_session("rest", session_data)
        
        assert "rest" in rag._index
        index_entry = rag._index["rest"]
        assert index_entry.label == "rest"
        assert index_entry.dominant_band == "alpha"
        assert index_entry.mean_quality == 0.85
        assert index_entry.total_artifacts == 3
        assert "relax: high" in index_entry.description
    
    def test_get_compressed_summary(self):
        """Compressed summary should be much smaller than full data."""
        rag = SessionRAG()
        
        # Large session data with many channels and full connectivity matrix
        session_data = {
            "label": "focus",
            "timestamp": 1234567890,
            "duration_s": 30.0,
            "dominant_band": "beta",
            "signal_quality": {
                "mean": 0.75,
                "std": 0.1,
                "per_channel": {f"ch{i}": 0.7 + i * 0.01 for i in range(32)},
            },
            "band_powers": {
                band: {
                    "mean": 2.0,
                    "std": 0.5,
                    "per_channel": {f"ch{i}": 2.0 + i * 0.1 for i in range(32)},
                }
                for band in ["delta", "theta", "alpha", "beta", "gamma"]
            },
            "connectivity": {
                "alpha": [[0.5 + (i + j) * 0.01 for j in range(32)] for i in range(32)],
                "beta": [[0.4 + (i + j) * 0.01 for j in range(32)] for i in range(32)],
            },
            "indices": {
                "focus": 0.78,
                "relax": 0.35,
                "engagement": 0.82,
            },
            "artifacts": {
                "blink": 8,
                "jaw_clench": 3,
            },
        }
        
        summary = rag.get_compressed_summary("focus", session_data)
        
        # Summary should be much smaller (removes per_channel data and matrices)
        full_size = len(json.dumps(session_data))
        summary_size = len(json.dumps(summary))
        assert summary_size < full_size / 3
        
        # But should preserve key info
        assert summary["label"] == "focus"
        assert summary["dominant_band"] == "beta"
        assert summary["indices"]["focus"] == 0.78
        assert "_retrieval_hint" in summary
        
        # Should not include full matrices
        assert "band_powers" not in summary or "per_channel" not in json.dumps(summary)
        assert "connectivity" not in summary
    
    def test_search_keyword_matching(self):
        """Search should find sessions by keywords."""
        rag = SessionRAG()
        
        # Index multiple sessions
        rag.index_session("rest", {
            "label": "rest",
            "timestamp": 100,
            "duration_s": 20,
            "dominant_band": "alpha",
            "signal_quality": {"mean": 0.8},
            "artifacts": {},
            "indices": {"focus": 0.3, "relax": 0.9, "engagement": 0.4},
        })
        
        rag.index_session("focus", {
            "label": "focus",
            "timestamp": 200,
            "duration_s": 30,
            "dominant_band": "beta",
            "signal_quality": {"mean": 0.7},
            "artifacts": {},
            "indices": {"focus": 0.85, "relax": 0.2, "engagement": 0.9},
        })
        
        # Search by keyword
        results = rag.search("high relax")
        assert "rest" in results
        
        results = rag.search("focus")
        assert "focus" in results
    
    def test_list_all(self):
        """List should return all indexed sessions."""
        rag = SessionRAG()
        
        rag.index_session("s1", {
            "label": "s1",
            "timestamp": 100,
            "duration_s": 10,
            "dominant_band": "alpha",
            "signal_quality": {"mean": 0.8},
            "artifacts": {},
        })
        
        rag.index_session("s2", {
            "label": "s2",
            "timestamp": 200,
            "duration_s": 20,
            "dominant_band": "beta",
            "signal_quality": {"mean": 0.7},
            "artifacts": {},
        })
        
        all_sessions = rag.list_all()
        assert len(all_sessions) == 2
        labels = [s["label"] for s in all_sessions]
        assert "s1" in labels
        assert "s2" in labels
    
    def test_remove(self):
        """Remove should delete a session from the index."""
        rag = SessionRAG()
        
        rag.index_session("temp", {
            "label": "temp",
            "timestamp": 100,
            "duration_s": 10,
            "dominant_band": "alpha",
            "signal_quality": {"mean": 0.8},
            "artifacts": {},
        })
        
        assert "temp" in rag._index
        
        result = rag.remove("temp")
        assert result is True
        assert "temp" not in rag._index
        
        # Removing again should return False
        result = rag.remove("temp")
        assert result is False
    
    def test_clear(self):
        """Clear should remove all indexed sessions."""
        rag = SessionRAG()
        
        for i in range(5):
            rag.index_session(f"s{i}", {
                "label": f"s{i}",
                "timestamp": i * 100,
                "duration_s": 10,
                "dominant_band": "alpha",
                "signal_quality": {"mean": 0.8},
                "artifacts": {},
            })
        
        assert len(rag._index) == 5
        
        rag.clear()
        assert len(rag._index) == 0


class TestIntegration:
    """Integration tests combining context management with session RAG."""
    
    def test_end_to_end_compression_with_sessions(self):
        """Test a realistic scenario with multiple session queries."""
        ctx = ContextManager(max_tokens=500, keep_recent=2)
        
        # Simulate a conversation with heavy session payloads
        messages = []
        
        # User asks to record a session
        messages.append(Message(role="user", content="Record a rest session"))
        messages.append(Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="record_session",
                    arguments={"label": "rest", "seconds": 20},
                )
            ],
        ))
        
        # Large session result - make it realistically large
        large_session = {
            "status": "recorded",
            "label": "rest",
            "n_frames": 5000,
            "duration_s": 20.0,
            "dominant_band": "alpha",
            "indices": {"focus": 0.4, "relax": 0.8, "engagement": 0.5},
            "signal_quality": {
                "mean": 0.85,
                "std": 0.1,
                "per_channel": {f"ch{i}": 0.8 + i * 0.01 for i in range(32)},
            },
            "band_powers": {
                band: {
                    "mean": 2.0 + i * 0.1,
                    "std": 0.5,
                    "per_channel": {f"ch{j}": 2.0 + j * 0.1 for j in range(32)},
                }
                for i, band in enumerate(["delta", "theta", "alpha", "beta", "gamma"])
            },
            "connectivity": {
                "alpha": [[0.5 + (i + j) * 0.01 for j in range(32)] for i in range(32)],
            },
            "extra_metadata": "x" * 1000,  # Add bulk to push over threshold
        }
        messages.append(Message(
            role="tool",
            tool_call_id="1",
            content=json.dumps(large_session),
        ))
        
        messages.append(Message(role="assistant", content="Session recorded."))
        
        # Add more turns to trigger compression
        for i in range(5):
            messages.append(Message(role="user", content=f"Question {i}"))
            messages.append(Message(role="assistant", content=f"Answer {i}"))
        
        # Should trigger compression
        assert ctx.should_compress(messages)
        
        compressed, stats = ctx.compress(messages)
        
        # Verify compression worked
        assert stats.messages_compressed > 0
        assert stats.compression_ratio < 1.0
        
        # Recent messages should be intact
        assert compressed[-1].content == "Answer 4"
        assert compressed[-2].content == "Question 4"
