"""HTTP/WebSocket surface for the web UI (no network, no real cascade).

These drive the real :class:`WebEngine` and FastAPI app through Starlette's
``TestClient``, with fake toolsets and a scripted copilot standing in for the
live cascade. They pin the contract the front-end depends on: REST reads, the
``/ws/live`` telemetry push, streaming chat events, and the guided-training
command/response loop.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from pieeg_agent.agent.copilot import CopilotEvent  # noqa: E402
from pieeg_agent.llm.provider import Usage  # noqa: E402
from pieeg_agent.web import WebEngine, create_app  # noqa: E402


class FakeSenses:
    def call(self, name, arguments=None):
        return {
            "get_neural_state": {"focus": 0.4, "relax": 0.6, "warming_up": False},
            "get_band_powers": {"relative": {"Alpha": 0.3}, "dominant": "Alpha"},
            "get_channel_quality": {"overall": 0.9, "channels": []},
            "get_recent_events": {"count": 0, "events": []},
        }[name]


class FakeDecode:
    def __init__(self):
        self.calls = []
        self._patterns = {"eyes-closed": {"name": "eyes-closed",
                                          "cross_validation": {"balanced_accuracy": 0.9}}}

    def call(self, name, arguments=None):
        args = arguments or {}
        self.calls.append((name, args))
        if name == "find_artifacts":
            return {"count": 0, "artifacts": []}
        if name == "detect_patterns":
            return {"patterns": [{"name": "eyes-closed", "probability": 0.2,
                                  "active": False}], "active": []}
        if name == "connectivity":
            return {"band": "Alpha", "n_channels": 2, "n_frames": 64,
                    "mean_connectivity": 0.15,
                    "strongest_pairs": [{"a": "C0", "b": "C1", "r": 0.15}],
                    "per_channel": [{"channel": "C0", "mean_abs_r": 0.15, "flat": False},
                                    {"channel": "C1", "mean_abs_r": 0.15, "flat": False}],
                    "most_connected": "C0", "least_connected": "C1",
                    "method": "log band-power amplitude correlation (Pearson r, within-session)",
                    "caveat": "Amplitude coupling, not phase coherence; descriptive and within-session only."}
        if name == "list_patterns":
            return {"count": 1, "patterns": [{"name": "eyes-closed",
                                              "balanced_accuracy": 0.9, "loaded": True}]}
        if name == "explain_pattern":
            nm = args.get("name")
            return self._patterns.get(nm, {"error": f"unknown pattern {nm!r}"})
        if name == "forget_pattern":
            nm = args.get("name")
            if nm in self._patterns:
                del self._patterns[nm]
                return {"status": "forgotten", "name": nm}
            return {"error": f"unknown pattern {nm!r}"}
        if name == "start_pattern_training":
            return {"status": "training_started", "name": args.get("name")}
        if name == "record_segment":
            return {"status": "segment_recorded", "label": args.get("label"),
                    "captured_frames": 10}
        if name == "finish_pattern_training":
            return {"status": "trained", "name": "eyes-closed",
                    "balanced_accuracy": 0.9}
        if name == "cancel_pattern_training":
            return {"status": "cancelled"}
        return {"error": f"unknown tool {name!r}"}


class FakeCopilot:
    def __init__(self):
        self.reset_called = False
        self.questions = []

    def ask_stream(self, question):
        self.questions.append(question)
        yield CopilotEvent(type="tool_start", name="get_neural_state", arguments={})
        yield CopilotEvent(type="tool_result", name="get_neural_state",
                           result={"focus": 0.4})
        yield CopilotEvent(type="token", text="You ")
        yield CopilotEvent(type="token", text="look calm.")
        yield CopilotEvent(type="done", text="You look calm.",
                           tool_calls=["get_neural_state"], usage=Usage(5, 3),
                           iterations=2)

    def reset(self):
        self.reset_called = True


def _make(engine_kwargs=None):
    senses, decode, copilot = FakeSenses(), FakeDecode(), FakeCopilot()
    engine = WebEngine(
        copilot=copilot,
        senses=senses,
        decode=decode,
        info={"stream": "PiEEG", "channels": 8, "rate": 250,
              "provider": "anthropic", "model": "test"},
        **(engine_kwargs or {}),
    )
    app = create_app(engine, live_interval=0.02)
    return engine, copilot, decode, TestClient(app)


# ── REST ─────────────────────────────────────────────────────────────────────


def test_health_and_info():
    _, _, _, client = _make()
    assert client.get("/api/health").json() == {"ok": True}
    info = client.get("/api/info").json()
    assert info["channels"] == 8
    assert info["provider"] == "anthropic"


def test_state_snapshot_has_all_panels():
    _, _, _, client = _make()
    snap = client.get("/api/state").json()
    assert set(snap) >= {"state", "bands", "quality", "events",
                         "artifacts", "patterns"}
    assert snap["state"]["focus"] == 0.4
    assert snap["patterns"]["patterns"][0]["name"] == "eyes-closed"


def test_patterns_list_explain_and_forget():
    _, _, _, client = _make()
    listing = client.get("/api/patterns").json()
    assert listing["count"] == 1

    ok = client.get("/api/patterns/eyes-closed")
    assert ok.status_code == 200
    assert ok.json()["cross_validation"]["balanced_accuracy"] == 0.9

    missing = client.get("/api/patterns/nope")
    assert missing.status_code == 404
    assert "error" in missing.json()

    gone = client.delete("/api/patterns/eyes-closed")
    assert gone.status_code == 200
    assert gone.json()["status"] == "forgotten"
    # now unknown
    assert client.get("/api/patterns/eyes-closed").status_code == 404


# ── WebSocket: live telemetry ────────────────────────────────────────────────


def test_ws_live_pushes_snapshots():
    _, _, _, client = _make()
    with client.websocket_connect("/ws/live") as ws:
        snap = ws.receive_json()
        assert set(snap) >= {"state", "bands", "quality", "events",
                             "artifacts", "patterns"}
        # a second frame proves it's a push loop, not a one-shot
        snap2 = ws.receive_json()
        assert "state" in snap2


# ── WebSocket: streaming chat ────────────────────────────────────────────────


def test_ws_chat_streams_tokens_and_done():
    _, copilot, _, client = _make()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "how am I?"})
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] == "done":
                break
    types = [e["type"] for e in events]
    assert types == ["tool_start", "tool_result", "token", "token", "done"]
    assert events[0]["name"] == "get_neural_state"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "You look calm."
    done = events[-1]
    assert done["text"] == "You look calm."
    assert done["tool_calls"] == ["get_neural_state"]
    assert done["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert copilot.questions == ["how am I?"]


def test_ws_chat_reset_and_empty():
    _, copilot, _, client = _make()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"reset": True})
        assert ws.receive_json() == {"type": "reset"}
        assert copilot.reset_called
        ws.send_json({"message": "   "})
        assert ws.receive_json() == {"type": "error", "detail": "empty message"}


# ── WebSocket: guided training ───────────────────────────────────────────────


def test_ws_train_begin_record_finish():
    _, _, decode, client = _make()
    with client.websocket_connect("/ws/train") as ws:
        ws.send_json({"action": "begin", "name": "eyes-closed"})
        r = ws.receive_json()
        assert r["action"] == "begin"
        assert r["result"]["status"] == "training_started"

        ws.send_json({"action": "record", "label": "rest", "seconds": 1})
        r = ws.receive_json()
        assert r["result"]["captured_frames"] == 10
        assert r["result"]["label"] == "rest"

        ws.send_json({"action": "finish", "threshold": 0.6})
        r = ws.receive_json()
        assert r["result"]["status"] == "trained"

    # the engine delegated to the decode tools in order
    names = [c[0] for c in decode.calls]
    assert names == ["start_pattern_training", "record_segment",
                     "finish_pattern_training"]


def test_ws_train_unknown_action():
    _, _, _, client = _make()
    with client.websocket_connect("/ws/train") as ws:
        ws.send_json({"action": "bogus"})
        r = ws.receive_json()
        assert "error" in r["result"]
