"""Read-only neural tools against a fake cascade.

The tools must pull real fields from the cascade, return JSON-friendly dicts,
degrade gracefully before any state exists, and never raise into the agent
loop. A tiny duck-typed ``FakeCascade`` stands in for the real perception
thread so these run without LSL or threads.
"""

import numpy as np

from pieeg_agent.agent.tools import NeuralTools
from pieeg_agent.perceive.events import NeuralEvent
from pieeg_agent.perceive.features import BAND_NAMES, BandPowers
from pieeg_agent.perceive.quality import ChannelQuality, SignalQuality
from pieeg_agent.perceive.state import NeuralState

BANDS = {"Delta": 0.1, "Theta": 0.1, "Alpha": 0.6, "Beta": 0.1, "Gamma": 0.1}


def mk_state() -> NeuralState:
    return NeuralState(
        timestamp=1000.0,
        rel_bands=dict(BANDS),
        dominant_band="Alpha",
        focus=0.42,
        relax=0.71,
        engagement=0.55,
        signal_quality=0.95,
        n_channels=4,
        bad_channels=(),
        warming_up=False,
    )


def mk_bp() -> BandPowers:
    per = np.tile(np.array([BANDS[b] for b in BAND_NAMES], float), (4, 1))
    return BandPowers(
        timestamp=1000.0,
        bands=dict(BANDS),
        per_channel=per,
        psd=np.zeros((4, 5)),
        freqs=np.zeros(5),
        n_samples=512,
        n_channels=4,
    )


def mk_quality() -> SignalQuality:
    chans = [
        ChannelQuality(0, "Ch0", 20.0, 1.0, 0.0, "good", 1.0),
        ChannelQuality(1, "Ch1", 0.5, 1.0, 0.0, "flat", 0.2),
    ]
    return SignalQuality(timestamp=1000.0, channels=chans, overall=0.6)


class FakeCascade:
    def __init__(self, *, state=None, bp=None, quality=None, events=None):
        self._state = state
        self._bp = bp
        self._quality = quality
        self._events = events or []

    def latest_state(self):
        return self._state

    def latest_band_powers(self):
        return self._bp

    def latest_quality(self):
        return self._quality

    def recent_events(self, n=20):
        return self._events[-n:]

    def channel_labels(self):
        return ["Ch0", "Ch1", "Ch2", "Ch3"]

    def stats(self):
        summary = self._state.summary() if self._state else ""
        return {
            "ticks": 1234,
            "features": 153,
            "states": 19,
            "events": len(self._events),
            "last_summary": summary,
        }


def test_specs_cover_all_tools():
    tools = NeuralTools(FakeCascade())
    names = {s.name for s in tools.specs()}
    assert names == {
        "get_neural_state",
        "get_band_powers",
        "get_recent_events",
        "get_channel_quality",
        "summarize_last",
        "get_cascade_stats",
    }
    # Every spec advertises an object schema (providers require it).
    for spec in tools.specs():
        assert spec.input_schema["type"] == "object"


def test_get_neural_state_returns_state_dict():
    tools = NeuralTools(FakeCascade(state=mk_state()))
    out = tools.call("get_neural_state")
    assert out["focus"] == 0.42
    assert out["dominant_band"] == "Alpha"
    assert out["warming_up"] is False


def test_tools_report_no_data_before_first_state():
    tools = NeuralTools(FakeCascade())
    for name in ("get_neural_state", "get_band_powers",
                 "get_channel_quality", "summarize_last"):
        out = tools.call(name)
        assert out["status"] == "no_data"


def test_get_band_powers_per_channel_breakdown():
    tools = NeuralTools(FakeCascade(bp=mk_bp()))
    flat = tools.call("get_band_powers")
    assert "per_channel" not in flat
    assert flat["dominant"] == "Alpha"

    detailed = tools.call("get_band_powers", {"per_channel": True})
    rows = detailed["per_channel"]
    assert len(rows) == 4
    assert rows[0]["channel"] == "Ch0"
    # Relative bands within a channel sum to ~1.
    keys = [b[0].lower() for b in BAND_NAMES]
    assert abs(sum(rows[0][k] for k in keys) - 1.0) < 1e-6


def test_get_recent_events_respects_limit():
    events = [
        NeuralEvent(timestamp=float(i), type="focus_high", value=0.8,
                    detail="focus rose", severity="info")
        for i in range(20)
    ]
    tools = NeuralTools(FakeCascade(events=events))
    out = tools.call("get_recent_events", {"limit": 5})
    assert out["count"] == 5
    assert out["events"][-1]["type"] == "focus_high"


def test_get_channel_quality_lists_worst_channel():
    tools = NeuralTools(FakeCascade(quality=mk_quality()))
    out = tools.call("get_channel_quality")
    assert out["worst"] == "Ch1"
    statuses = {c["label"]: c["status"] for c in out["channels"]}
    assert statuses["Ch1"] == "flat"


def test_unknown_tool_returns_error_not_exception():
    tools = NeuralTools(FakeCascade())
    out = tools.call("nonexistent")
    assert "error" in out
    assert "available" in out


def test_summarize_last_returns_one_liner():
    tools = NeuralTools(FakeCascade(state=mk_state()))
    out = tools.call("summarize_last")
    assert "focus" in out["summary"]
    assert out["warming_up"] is False


def test_get_cascade_stats_returns_processing_counts():
    events = [
        NeuralEvent(timestamp=float(i), type="focus_high", value=0.8,
                    detail="focus rose", severity="info")
        for i in range(3)
    ]
    tools = NeuralTools(FakeCascade(state=mk_state(), events=events))
    out = tools.call("get_cascade_stats")
    assert out["ticks"] == 1234
    assert out["features"] == 153
    assert out["states"] == 19
    assert out["events"] == 3
    assert "focus" in out["last_summary"]
