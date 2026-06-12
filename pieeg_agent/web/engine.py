"""The web engine — one façade over the live brain for the HTTP/WS layer.

This is the bridge between the graphical front-end and the *same* copilot,
perception cascade and pattern engine the CLI uses. It owns no new brain logic:
it reuses the already-tested tool serializers (:class:`NeuralTools`,
:class:`DecodeTools`) so every fact the browser sees is the exact JSON the LLM
would see, and it streams the copilot through :meth:`Copilot.ask_stream`.

Keeping this layer thin and dependency-light (no FastAPI import here) means the
engine is unit-testable without a server, and the transport in
:mod:`pieeg_agent.web.app` stays a dumb pipe.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator, Protocol

from ..agent.copilot import CopilotEvent


class _Toolset(Protocol):
    def call(self, name: str, arguments: dict | None = None) -> dict: ...


class _Copilot(Protocol):
    def ask_stream(self, question: str) -> Iterator[CopilotEvent]: ...
    def reset(self) -> None: ...


def event_to_dict(ev: CopilotEvent) -> dict:
    """Serialize one :class:`CopilotEvent` to a JSON-friendly dict for the wire."""
    out: dict[str, Any] = {"type": ev.type}
    if ev.type == "token":
        out["text"] = ev.text
    elif ev.type == "tool_start":
        out["name"] = ev.name
        out["arguments"] = ev.arguments
    elif ev.type == "tool_result":
        out["name"] = ev.name
        out["result"] = ev.result
    elif ev.type == "model_switch":
        out["text"] = ev.text
        out["reason"] = ev.reason
    elif ev.type == "done":
        out["text"] = ev.text
        out["tool_calls"] = ev.tool_calls
        out["usage"] = {
            "input_tokens": ev.usage.input_tokens,
            "output_tokens": ev.usage.output_tokens,
        }
        out["iterations"] = ev.iterations
    return out


class WebEngine:
    """Façade the HTTP/WS app calls — chat, live snapshots, pattern training.

    The engine holds the shared, stateful copilot, so chat turns are
    serialized behind a lock: two browser tabs can both be connected, but their
    messages interleave one full turn at a time rather than corrupting the
    single conversation history.
    """

    def __init__(
        self,
        *,
        copilot: _Copilot,
        senses: _Toolset,
        decode: _Toolset,
        info: dict | None = None,
        actions: Any | None = None,
    ):
        self._copilot = copilot
        self._senses = senses
        self._decode = decode
        self._info = dict(info or {})
        self._actions = actions
        self._chat_lock = threading.Lock()

    # ── metadata ─────────────────────────────────────────────────────────
    def info(self) -> dict:
        """Static session facts: stream name, channels, rate, provider/model."""
        return dict(self._info)

    # ── live state ───────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """A full read of the current brain, assembled from the tested tools.

        Every field is the same JSON the copilot reads, so the UI and the model
        never disagree. Safe to call at the UI's refresh rate; each call is a
        cheap read of the latest cascade outputs.
        """
        return {
            "state": self._senses.call("get_neural_state", {}),
            "bands": self._senses.call("get_band_powers", {"per_channel": True}),
            "quality": self._senses.call("get_channel_quality", {}),
            "events": self._senses.call("get_recent_events", {"limit": 8}),
            "artifacts": self._decode.call("find_artifacts", {"limit": 8}),
            "patterns": self._decode.call("detect_patterns", {}),
            "connectivity": self._decode.call("connectivity", {"seconds": 8.0}),
            "cascade": self._senses.call("get_cascade_stats", {}),
        }

    # ── chat ─────────────────────────────────────────────────────────────
    def chat_stream(self, message: str) -> Iterator[dict]:
        """Stream one copilot turn as JSON-friendly event dicts.

        Serialized behind a lock so concurrent connections can't corrupt the
        shared conversation history.
        """
        with self._chat_lock:
            for ev in self._copilot.ask_stream(message):
                yield event_to_dict(ev)

    def reset_chat(self) -> dict:
        """Forget the conversation so far."""
        with self._chat_lock:
            self._copilot.reset()
        return {"status": "reset"}

    # ── patterns (inspection) ────────────────────────────────────────────
    def list_patterns(self) -> dict:
        return self._decode.call("list_patterns", {})

    def detect_patterns(self) -> dict:
        return self._decode.call("detect_patterns", {})

    def explain_pattern(self, name: str) -> dict:
        return self._decode.call("explain_pattern", {"name": name})

    def forget_pattern(self, name: str) -> dict:
        return self._decode.call("forget_pattern", {"name": name})

    # ── pattern training (guided) ────────────────────────────────────────
    def train_begin(self, name: str) -> dict:
        return self._decode.call("start_pattern_training", {"name": name})

    def train_record(self, label: str, seconds: float) -> dict:
        """Capture one labelled segment. Blocks ``seconds`` while frames stream;
        the transport runs this off the event loop."""
        return self._decode.call(
            "record_segment", {"label": label, "seconds": seconds}
        )

    def train_finish(self, threshold: float = 0.6) -> dict:
        return self._decode.call(
            "finish_pattern_training", {"threshold": threshold}
        )

    def train_cancel(self) -> dict:
        return self._decode.call("cancel_pattern_training", {})

    # ── LSL streams discovery ────────────────────────────────────────────
    def list_streams(self, wait: float = 2.0) -> dict:
        """Discover LSL streams on the network (threadsafe)."""
        try:
            from pylsl import resolve_streams
        except Exception as exc:
            return {"error": f"pylsl unavailable: {exc}"}
        try:
            streams = resolve_streams(wait)
            return {
                "streams": [
                    {
                        "name": s.name(),
                        "type": s.type(),
                        "channels": s.channel_count(),
                        "rate": s.nominal_srate(),
                        "source_id": s.source_id(),
                    }
                    for s in streams
                ],
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ── server control (if available) ────────────────────────────────────
    def server_status(self) -> dict:
        """Get PiEEG-server status if actions are available."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled (use --allow-actions)"}
        try:
            return self._actions.server_info()
        except Exception as exc:
            return {"error": str(exc)}

    def server_filter(self, enabled: bool, lowcut: float, highcut: float) -> dict:
        """Set server-side band-pass filter."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled"}
        try:
            return self._actions.set_filter(
                enabled=enabled, lowcut=lowcut, highcut=highcut
            )
        except Exception as exc:
            return {"error": str(exc)}

    def server_record(self, action: str) -> dict:
        """Start or stop server-side recording."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled"}
        try:
            if action == "start":
                return self._actions.start_recording()
            elif action == "stop":
                return self._actions.stop_recording()
            else:
                return {"error": f"unknown action {action!r}"}
        except Exception as exc:
            return {"error": str(exc)}

    def server_osc(self, action: str, config: dict | None = None) -> dict:
        """Start or stop OSC output."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled"}
        try:
            if action == "start":
                return self._actions.start_osc(config)
            elif action == "stop":
                return self._actions.stop_osc()
            else:
                return {"error": f"unknown action {action!r}"}
        except Exception as exc:
            return {"error": str(exc)}

    def server_lsl(self, action: str, config: dict | None = None) -> dict:
        """Start or stop LSL output."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled"}
        try:
            if action == "start":
                return self._actions.start_lsl(config)
            elif action == "stop":
                return self._actions.stop_lsl()
            else:
                return {"error": f"unknown action {action!r}"}
        except Exception as exc:
            return {"error": str(exc)}

    def server_register_preset(self, preset: str) -> dict:
        """Apply an ADS1299 register preset."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled"}
        try:
            return self._actions.apply_register_preset(preset)
        except Exception as exc:
            return {"error": str(exc)}

    def server_webhooks(self) -> dict:
        """List server webhook rules."""
        if not hasattr(self, "_actions") or self._actions is None:
            return {"error": "Server control not enabled"}
        try:
            return self._actions.list_webhooks()
        except Exception as exc:
            return {"error": str(exc)}
