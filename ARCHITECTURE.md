# PiEEG-agent Architecture — Resilience & Design

## Overview

PiEEG-agent is a brain-computer interface (BCI) with an AI copilot. The copilot can observe EEG state through read-only tools and answer natural-language questions about the user's brain activity.

## Core Design Philosophy

**Simple. Resilient. Honest.**

1. **Simple**: No unnecessary complexity. Direct wire protocols, standard library HTTP, minimal dependencies.
2. **Resilient**: Graceful degradation with automatic fallbacks, timeout detection, and clear error recovery.
3. **Honest**: Transparent metrics, clear limitations, no magic claims.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  Web UI (React + TypeScript)                            │
│  - Chat interface with streaming responses              │
│  - Notebook viewer for Jupyter notebooks                │
│  - Real-time EEG visualization (live/training)          │
└─────────────────────────────────────────────────────────┘
                        ↓ WebSocket
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend (Python)                               │
│  - 3 WebSocket channels: /ws/live, /ws/chat, /ws/train  │
│  - REST endpoints for notebook access                   │
│  - WebEngine façade between HTTP/WS and brain logic     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Copilot (Tool-Using LLM Loop)                          │
│  - Multi-turn conversation with bounded tool iterations │
│  - Automatic fallback on failures/timeouts              │
│  - Context compression for long conversations           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Tool Ecosystem                                          │
│  - NeuralTools: Read brain state (band powers, quality) │
│  - DecodeTools: Pattern training, session recording     │
│  - DocumentationTools: Search docs, web, fetch URLs     │
│  - UtilityTools: File I/O, notebooks, introspection     │
│  - ActuatorTools: Control actions (gated)               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  EEG Processing Pipeline                                 │
│  - LSL stream ingestion with ring buffers               │
│  - Spectral analysis (Welch PSD, band powers)           │
│  - Artifact detection (blinks, jaw, motion)             │
│  - Connectivity analysis (amplitude coupling)           │
│  - Pattern classifiers (shrinkage-LDA)                  │
└─────────────────────────────────────────────────────────┘
```

## Resilience Mechanisms

### 1. **Tool Iteration Limit (max_tool_iters = 10)**

**Why 10 iterations?**

The copilot operates in a loop:
1. User asks a question
2. LLM generates response or requests tools
3. Tools execute and return results
4. Go back to step 2 until LLM produces final text

**10 iterations allows ~5 rounds of tool calls** (ask → tool → ask → tool → ... → answer), which handles complex multi-step tasks like:
- "Compare my rest session to my focus session" → needs get_session, get_session, compare_sessions
- "Train a new relaxation pattern" → needs multiple record_segment calls with user confirmation between each

**It prevents infinite loops** where a misbehaving LLM keeps calling tools without converging to an answer.

When the limit is reached, the copilot makes one final attempt to answer without tools, ensuring the user always gets a response.

### 2. **Multi-Level Timeout Protection**

#### HTTP Request Timeout (60s default)
- Protects against completely stalled connections
- Applies to initial connection establishment
- Configurable via `timeout` parameter in `_http.py`

#### Streaming Heartbeat Timeout (30s default)
- **NEW** — Detects when streaming has started but then stalls
- Monitors time between chunks during streaming responses
- Triggers fallback if no chunks arrive for 30s
- Prevents the "tool calls complete but then nothing happens" issue

#### Per-Iteration Timeout (120s default)
- **NEW** — Total time allowed for one copilot iteration
- Includes tool execution + LLM response time
- Provides safety net for slow LLM providers
- Configurable via `per_iteration_timeout` parameter

### 3. **Automatic Fallback Provider**

The copilot supports a fallback LLM provider that automatically activates when the primary fails.

**Triggers fallback on:**
- HTTP 429 (rate limit)
- HTTP 503 (service unavailable)
- TimeoutError (stream stalled or iteration timeout)
- ConnectionError (network failure)

**Fallback behavior:**
1. Switches to fallback provider
2. Emits `model_switch` event to notify user
3. Retries the same request with fallback
4. Applies same heartbeat protection to fallback
5. After successful requests, attempts to switch back to primary

**User feedback:**
```
⚠️ Switched to gpt-4o-mini
Primary model (claude-3-7-sonnet-20250219) timeout after 45s
```

### 4. **Retry Logic with Exponential Backoff**

Rate limits (429) and service errors (503) are automatically retried:
- 5 retries max
- Exponential backoff: 2s, 4s, 8s, 16s, 32s
- Total retry window: ~62s
- Only for transient failures; other errors fail immediately

### 5. **Context Compression**

Long conversations are automatically compressed to stay under token limits:
- Removes old tool calls while preserving conversation flow
- Keeps recent context intact
- Logs compression stats for debugging
- Prevents context overflow errors mid-conversation

### 6. **Recording Limits (Pattern Training)**

Special protection for `record_segment` tool:
- **Max 1 recording per copilot turn**
- Prevents the model from chaining recordings (which would freeze UI and capture wrong mental state)
- User must confirm readiness between each recording
- System caps training at 8 reps total

This enforces the turn-based pattern training protocol:
```
User: "Train a relaxation pattern"
Copilot: "Please relax and say 'ready' when settled"
User: "ready"
Copilot: [records rest] "Now focus on relaxing deeply, say 'ready'"
User: "ready"
Copilot: [records active] "Great! Please return to normal state..."
```

## Configuration Parameters

All resilience parameters are configurable in `Copilot.__init__()`:

```python
copilot = Copilot(
    provider=primary_provider,
    tools=toolset,
    
    # Tool loop limits
    max_tool_iters=10,              # Max iterations before forcing answer
    
    # Timeout protection
    per_iteration_timeout=120.0,    # Max time per iteration (s)
    stream_heartbeat_timeout=30.0,  # Max time between chunks (s)
    
    # Provider failover
    fallback_provider=backup,       # Automatic fallback on errors
    min_request_interval=0.5,       # Rate limiting between requests
)
```

## Error Handling Flow

```
┌──────────────────────┐
│  Start LLM Request   │
└──────────┬───────────┘
           │
           ▼
    ┌──────────────┐         Success          ┌─────────────┐
    │   Streaming  │────────────────────────▶│   Complete  │
    │   Response   │                          └─────────────┘
    └──────┬───────┘
           │
           │ Failure (429, 503, Timeout, Connection)
           ▼
    ┌──────────────────────┐
    │ Fallback Available?  │
    └──────┬───────────────┘
           │
    Yes    │              No
           ▼               │
    ┌────────────────┐     │
    │ Switch to      │     │
    │ Fallback       │     │
    └──────┬─────────┘     │
           │               │
           ▼               ▼
    ┌────────────────┐  ┌──────────────┐
    │ Retry Request  │  │ Raise Error  │
    └──────┬─────────┘  └──────────────┘
           │
           ▼
    (Back to Streaming)
```

## Why This Design?

### Simplicity
- **No message queues**: Direct WebSocket streaming keeps architecture simple
- **Standard library HTTP**: No vendor SDKs reduces dependency surface
- **Tool-based**: LLM calls functions; clean separation of concerns

### Resilience
- **Multiple timeout layers**: HTTP, heartbeat, and per-iteration timeouts catch all failure modes
- **Automatic fallback**: Graceful degradation without user intervention
- **Bounded loops**: Iteration limits prevent runaway costs and hangs
- **Retry with backoff**: Transient failures recover automatically

### Transparency
- **Visible tool calls**: User sees what the copilot is doing
- **Honest metrics**: "focus/relax are relative to YOUR session, not clinical"
- **Error messages**: Clear, actionable feedback when things fail
- **Model switches**: User notified when fallback activates

## Testing Resilience

### Test Timeout Detection
```bash
# Simulate slow provider by adding delay in provider code
python -m pieeg_agent chat --allow-actions

# Ask complex question that requires multiple tool calls
# Watch for heartbeat timeout detection and fallback activation
```

### Test Fallback
```bash
# Configure both primary and fallback in config
export PIEEG_PRIMARY_MODEL="claude-3-7-sonnet-20250219"
export PIEEG_FALLBACK_MODEL="gpt-4o-mini"

# Use primary until it fails, then watch automatic switch
```

### Test Iteration Limit
```bash
# Ask question that might cause infinite loop
# System should stop at 10 iterations and provide best-effort answer
```

## Future Improvements

Potential enhancements while maintaining simplicity:

1. **Progress indicators**: Emit iteration count to frontend for better UX
2. **Adaptive timeouts**: Learn typical response times and adjust dynamically
3. **Circuit breaker**: Temporarily disable failing providers after repeated errors
4. **Streaming checkpoints**: Save partial responses to recover from crashes
5. **Tool result caching**: Cache expensive tool calls (e.g., search_web) within a session

## See Also

- `pieeg_agent/agent/copilot.py` — Copilot implementation
- `pieeg_agent/llm/_http.py` — HTTP client with retry logic
- `pieeg_agent/web/engine.py` — WebSocket event mapping
- `tests/test_copilot.py` — Copilot tests including fallback scenarios
