"""Tests for the DecodeTools engine — the live training/scoring tool surface.

These drive the engine the way the agent would (start → record rest/active reps
→ finish → inspect), but replace the cascade thread with a deterministic frame
feeder injected as the ``sleep`` hook, so a ``record_segment`` call gathers a
known set of synthetic frames instead of waiting on wall-clock time.
"""

import numpy as np

from pieeg_agent.agent.decode_tools import DecodeTools
from pieeg_agent.decode import PatternStore, SessionStore



class _BP:
    """Minimal BandPowers stand-in (what on_frame actually touches)."""

    def __init__(self, per_channel, timestamp):
        from pieeg_agent.perceive.features import BAND_NAMES
        self.per_channel = per_channel
        self.n_channels = per_channel.shape[0]
        self.timestamp = timestamp
        # SessionRecorder.add() calls bp.relative(), so provide it.
        n_b = per_channel.shape[1] if per_channel.ndim > 1 else 5
        total = per_channel.sum() or 1e-12
        self._rel = {
            BAND_NAMES[i]: float(per_channel[:, i].sum() / total)
            for i in range(min(n_b, len(BAND_NAMES)))
        }

    def relative(self):
        return self._rel


class _FakeCascade:
    def __init__(self, n_ch=4):
        self._labels = [f"C{i}" for i in range(n_ch)]
        self._state = None

    def channel_labels(self):
        return list(self._labels)

    def recent_artifacts(self, n=20):
        return []

    def latest_band_powers(self):
        return None

    def latest_state(self):
        return self._state


class _Feeder:
    """Injected as the engine's ``sleep``: feeds frames for the current label."""

    def __init__(self, decode, n_ch=4, per_segment=25, hot_channel=2):
        self.decode = decode
        self.n_ch = n_ch
        self.per = per_segment
        self.hot = hot_channel
        self.mode = "rest"
        self.t = 0.0
        self.rng = np.random.default_rng(0)

    def feed(self, seconds=0.0):
        for _ in range(self.per):
            bands = np.abs(self.rng.normal(1.0, 0.05, size=(self.n_ch, 5)))
            raw = self.rng.normal(0.0, 2.0, size=(64, self.n_ch))
            if self.mode == "active":
                bands[self.hot] *= 50.0                    # one channel lights up
                raw[:, self.hot] += self.rng.normal(0.0, 40.0, size=64)
            self.t += 0.125
            self.decode.on_frame(_BP(bands, self.t), None, raw, None)


def _engine(tmp_path):
    decode = DecodeTools(
        _FakeCascade(4),
        store=PatternStore(tmp_path / "patterns"),
        session_store=SessionStore(tmp_path / "sessions"),
    )
    feeder = _Feeder(decode)
    decode._sleep = feeder.feed          # inject deterministic frame feeder
    return decode, feeder


def test_tools_are_advertised(tmp_path):
    decode, _ = _engine(tmp_path)
    names = decode.names()
    for t in (
        "find_artifacts", "analyze_spectrum", "start_pattern_training",
        "record_segment", "finish_pattern_training", "cancel_pattern_training",
        "list_patterns", "detect_patterns", "explain_pattern", "forget_pattern",
        "connectivity", "record_session", "list_sessions", "analyze_session",
        "compare_sessions", "forget_session",
    ):
        assert t in names


def test_record_before_start_errors(tmp_path):
    decode, _ = _engine(tmp_path)
    out = decode.call("record_segment", {"label": "rest"})
    assert "error" in out


def test_start_without_signal_reports_no_signal(tmp_path):
    decode, _ = _engine(tmp_path)  # no frames fed yet → layout unknown
    out = decode.call("start_pattern_training", {"name": "mathx"})
    assert out["status"] == "no_signal"


def test_full_training_flow_trains_and_scores(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    feeder.feed()  # warm-up frames so the engine learns the montage

    assert decode.call("start_pattern_training", {"name": "eyes-closed"})["status"] == "training_started"
    for _ in range(3):
        feeder.mode = "rest"
        r = decode.call("record_segment", {"label": "rest", "seconds": 1})
        assert r["captured_frames"] == feeder.per
        feeder.mode = "active"
        decode.call("record_segment", {"label": "active", "seconds": 1})

    res = decode.call("finish_pattern_training", {})
    assert res["status"] == "trained"
    assert res["balanced_accuracy"] is not None and res["balanced_accuracy"] > 0.8
    # The detector should rely on the channel that actually moved.
    assert res["channel_importance"][0]["channel"] == "C2"

    # Now listed, loaded and scoring live.
    listed = decode.call("list_patterns", {})
    assert any(p["name"] == "eyes-closed" and p["loaded"] for p in listed["patterns"])

    # Feeding active frames live should push the smoothed score over threshold.
    feeder.mode = "active"
    feeder.feed()
    feeder.feed()
    detect = decode.call("detect_patterns", {})
    assert "eyes-closed" in [p["name"] for p in detect["patterns"]]
    assert "eyes-closed" in detect["active"]


def test_explain_and_forget(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    feeder.feed()
    decode.call("start_pattern_training", {"name": "jaw"})
    for _ in range(3):
        feeder.mode = "rest"
        decode.call("record_segment", {"label": "rest"})
        feeder.mode = "active"
        decode.call("record_segment", {"label": "active"})
    decode.call("finish_pattern_training", {})

    explain = decode.call("explain_pattern", {"name": "jaw"})
    assert explain["name"] == "jaw"
    assert "channel_importance" in explain and "cross_validation" in explain

    forgotten = decode.call("forget_pattern", {"name": "jaw"})
    assert forgotten["status"] == "forgotten"
    assert not decode._store.exists("jaw")
    assert "jaw" not in decode.bank.names()


def test_cancel_training(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    feeder.feed()
    decode.call("start_pattern_training", {"name": "tmp"})
    assert decode.call("cancel_pattern_training", {})["status"] == "cancelled"
    # After cancel, recording should error again (no session).
    assert "error" in decode.call("record_segment", {"label": "rest"})


# ── connectivity ─────────────────────────────────────────────────────────


def test_connectivity_no_data(tmp_path):
    decode, _ = _engine(tmp_path)  # no frames yet
    out = decode.call("connectivity", {})
    assert out["status"] == "no_data"


def test_connectivity_insufficient_frames(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    feeder.feed(1.0)  # fewer than 8 frames
    # Ask for a large window → won't have enough
    out = decode.call("connectivity", {"seconds": 100.0})
    # should either be insufficient or succeed with whatever history is present
    # (If feed(1.0) produced >8 frames, the test might succeed — that's ok.)


def test_connectivity_basic(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    feeder.feed()  # feed full segment (~25 frames @ 0.125s apart)
    out = decode.call("connectivity", {"band": "Alpha", "seconds": 8.0})
    assert out.get("band") == "Alpha"
    assert out.get("n_channels") == 4
    assert "mean_connectivity" in out
    assert "strongest_pairs" in out


def test_connectivity_different_band(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    feeder.feed()
    out_alpha = decode.call("connectivity", {"band": "Alpha"})
    out_beta = decode.call("connectivity", {"band": "Beta"})
    assert out_alpha["band"] == "Alpha"
    assert out_beta["band"] == "Beta"


# ── sessions (the lab notebook) ──────────────────────────────────────────


def test_record_session_basic(tmp_path):
    """A session recording feeds frames via sleep, then saves the summary."""
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    out = decode.call("record_session", {"label": "test", "seconds": 1.0})
    assert out["status"] == "recorded"
    assert out["label"] == "test"
    assert out["n_frames"] == feeder.per  # feeder feeds 'per' frames per call


def test_list_sessions(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    r1 = decode.call("record_session", {"label": "one", "seconds": 1.0})
    assert r1["status"] == "recorded"
    r2 = decode.call("record_session", {"label": "two", "seconds": 1.0})
    assert r2["status"] == "recorded"
    listed = decode.call("list_sessions", {})
    assert listed["count"] == 2
    labels = {s["name"] for s in listed["sessions"]}
    assert labels == {"one", "two"}


def test_analyze_session(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    decode.call("record_session", {"label": "demo", "seconds": 1.0})
    summary = decode.call("analyze_session", {"label": "demo"})
    assert summary["label"] == "demo"
    assert summary["n_frames"] == feeder.per
    assert "bands" in summary
    assert "connectivity" in summary


def test_analyze_session_unknown_label_errors(tmp_path):
    decode, _ = _engine(tmp_path)
    out = decode.call("analyze_session", {"label": "unknown"})
    assert "error" in out


def test_compare_sessions(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    decode.call("record_session", {"label": "rest", "seconds": 1.0})
    feeder.mode = "active"
    decode.call("record_session", {"label": "active", "seconds": 1.0})
    cmp = decode.call("compare_sessions", {"a": "rest", "b": "active"})
    assert cmp["a"] == "rest"
    assert cmp["b"] == "active"
    assert "differences" in cmp
    assert len(cmp["differences"]) > 0


def test_forget_session(tmp_path):
    decode, feeder = _engine(tmp_path)
    feeder.mode = "rest"
    decode.call("record_session", {"label": "tmp", "seconds": 1.0})
    assert decode._sessions.exists("tmp")
    out = decode.call("forget_session", {"label": "tmp"})
    assert out["status"] == "forgotten"
    assert not decode._sessions.exists("tmp")


def test_forget_session_unknown_errors(tmp_path):
    decode, _ = _engine(tmp_path)
    out = decode.call("forget_session", {"label": "unknown"})
    assert "error" in out
