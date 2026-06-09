"""Tests for decode/session.py — session recording, summaries and comparisons."""

import time
from unittest.mock import Mock

import numpy as np

from pieeg_agent.decode.session import (
    SessionRecorder,
    SessionRecording,
    compare_summaries,
)
from pieeg_agent.perceive.features import BandPowers, BAND_NAMES


def _fake_bp(ts: float, per_ch: np.ndarray) -> Mock:
    """Build a minimal BandPowers stand-in."""
    bp = Mock(spec=BandPowers)
    bp.timestamp = ts
    bp.per_channel = per_ch
    n_ch, n_b = per_ch.shape
    bp.n_channels = n_ch
    bp.bands = {b: float(per_ch[:, i].mean()) for i, b in enumerate(BAND_NAMES)}
    bp.relative = Mock(return_value={
        b: float(per_ch[:, i].sum() / (per_ch.sum() or 1e-12))
        for i, b in enumerate(BAND_NAMES)
    })
    return bp


def _fake_quality(score: float) -> Mock:
    q = Mock()
    q.overall = score
    return q


def _fake_state(f: float, r: float, e: float) -> Mock:
    s = Mock()
    s.focus = f
    s.relax = r
    s.engagement = e
    return s


def test_recorder_open_close():
    """A recorder can open, accumulate frames, then close to produce a recording."""
    rec = SessionRecorder()
    assert not rec.is_open
    rec.open("test", t0=100.0, channel_labels=["A", "B"])
    assert rec.is_open
    assert rec.label == "test"
    assert rec.started_at == 100.0
    assert rec.n_frames == 0

    # Add a frame
    pc = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5, 5.5]])
    rec.add(_fake_bp(100.5, pc), quality=_fake_quality(0.8), state=_fake_state(0.6, 0.4, 0.5))
    assert rec.n_frames == 1

    recording = rec.close()
    assert not rec.is_open
    assert recording.label == "test"
    assert recording.n_frames == 1
    assert recording.channel_labels == ["A", "B"]


def test_recorder_no_frames():
    """Close with zero frames still produces a valid (empty) recording."""
    rec = SessionRecorder()
    rec.open("empty")
    recording = rec.close()
    assert recording.n_frames == 0
    summary = recording.summary()
    assert summary["label"] == "empty"
    assert summary["n_frames"] == 0
    assert summary["duration_s"] == 0.0


def test_recording_summary():
    """A recording condenses into a small, honest summary."""
    rec = SessionRecorder()
    rec.open("demo", t0=100.0, channel_labels=["Ch0", "Ch1"])
    rng = np.random.default_rng(42)
    for i in range(10):
        ts = 100.0 + i * 0.125
        pc = rng.random((2, 5))
        rec.add(_fake_bp(ts, pc), quality=_fake_quality(0.8 + i * 0.01), state=_fake_state(0.5, 0.4, 0.6))
    recording = rec.close()
    assert recording.n_frames == 10
    assert recording.duration_s > 1.0  # 9 * 0.125 ≈ 1.125 s

    summary = recording.summary()
    assert summary["label"] == "demo"
    assert summary["n_frames"] == 10
    assert summary["duration_s"] > 1.0
    assert "bands" in summary
    assert all(b in summary["bands"] for b in BAND_NAMES)
    assert "indices" in summary
    assert summary["indices"]["focus"]["n"] == 10
    assert summary["signal_quality"]["n"] == 10
    assert summary["per_channel_bands"]["labels"] == ["Ch0", "Ch1"]
    # With 10 frames, connectivity should compute (>= 8)
    conn = summary["connectivity"]
    # success → no 'status' key (only failures have status)
    assert "status" not in conn or conn["status"] != "insufficient"


def test_recording_insufficient_connectivity():
    """Fewer than 8 frames produces status=insufficient connectivity."""
    rec = SessionRecorder()
    rec.open("short", channel_labels=["A", "B"])
    for i in range(5):
        pc = np.random.random((2, 5))
        rec.add(_fake_bp(float(i), pc))
    recording = rec.close()
    summary = recording.summary()
    assert summary["connectivity"]["status"] == "insufficient"


def test_compare_empty():
    """compare_summaries with no matching features returns empty differences."""
    a = {"label": "a", "n_frames": 0, "bands": {}, "indices": {}}
    b = {"label": "b", "n_frames": 0, "bands": {}, "indices": {}}
    result = compare_summaries(a, b)
    assert result["a"] == "a"
    assert result["b"] == "b"
    assert result["differences"] == []


def test_compare_bands():
    """compare_summaries computes Cohen's d for each band."""
    a = {
        "label": "rest",
        "n_frames": 10,
        "bands": {
            "Alpha": {"mean": 10.0, "std": 2.0},
            "Beta": {"mean": 5.0, "std": 1.0},
        },
        "indices": {},
        "band_names": ["Alpha", "Beta"],
    }
    b = {
        "label": "active",
        "n_frames": 10,
        "bands": {
            "Alpha": {"mean": 8.0, "std": 2.0},
            "Beta": {"mean": 7.0, "std": 1.0},
        },
        "indices": {},
        "band_names": ["Alpha", "Beta"],
    }
    result = compare_summaries(a, b)
    diffs = result["differences"]
    assert len(diffs) == 2  # Alpha + Beta
    alpha_diff = [d for d in diffs if d["feature"] == "Alpha power"][0]
    beta_diff = [d for d in diffs if d["feature"] == "Beta power"][0]
    # Alpha: 8 - 10 = -2.0, d ≈ -2.0 / pooled_sd ≈ -1.0 (since both std=2.0 → pooled≈2.0)
    assert alpha_diff["delta"] == -2.0
    assert alpha_diff["cohens_d"] < 0  # negative
    # Beta: 7 - 5 = +2.0, d ≈ +2.0 (both std=1.0 → pooled≈1.0)
    assert beta_diff["delta"] == 2.0
    assert beta_diff["cohens_d"] > 0  # positive


def test_compare_indices():
    """compare_summaries includes focus/relax/engagement and signal_quality."""
    a = {
        "label": "a",
        "n_frames": 20,
        "bands": {},
        "indices": {
            "focus": {"mean": 0.5, "std": 0.1, "n": 20},
            "relax": {"mean": 0.4, "std": 0.1, "n": 20},
        },
        "signal_quality": {"mean": 0.8, "std": 0.05, "n": 20},
        "band_names": [],
    }
    b = {
        "label": "b",
        "n_frames": 20,
        "bands": {},
        "indices": {
            "focus": {"mean": 0.7, "std": 0.1, "n": 20},
            "relax": {"mean": 0.3, "std": 0.1, "n": 20},
        },
        "signal_quality": {"mean": 0.75, "std": 0.05, "n": 20},
        "band_names": [],
    }
    result = compare_summaries(a, b)
    diffs = result["differences"]
    feats = {d["feature"] for d in diffs}
    assert "focus" in feats
    assert "relax" in feats
    assert "signal quality" in feats
    focus_diff = [d for d in diffs if d["feature"] == "focus"][0]
    # focus: 0.7 - 0.5 = +0.2
    assert focus_diff["delta"] > 0.1


def test_compare_ranked():
    """differences should be ranked by absolute Cohen's d."""
    a = {
        "label": "a",
        "n_frames": 10,
        "bands": {
            "Alpha": {"mean": 10.0, "std": 1.0},  # big change → large d
            "Beta": {"mean": 5.0, "std": 1.0},    # small change → small d
        },
        "indices": {},
        "band_names": ["Alpha", "Beta"],
    }
    b = {
        "label": "b",
        "n_frames": 10,
        "bands": {
            "Alpha": {"mean": 15.0, "std": 1.0},  # Δ=5.0, d≈5.0
            "Beta": {"mean": 5.5, "std": 1.0},    # Δ=0.5, d≈0.5
        },
        "indices": {},
        "band_names": ["Alpha", "Beta"],
    }
    result = compare_summaries(a, b)
    diffs = result["differences"]
    # Alpha should be first (larger absolute d)
    assert diffs[0]["feature"] == "Alpha power"
    assert abs(diffs[0]["cohens_d"]) > abs(diffs[1]["cohens_d"])


def test_compare_headline():
    """headline should summarise the biggest change."""
    a = {
        "label": "rest",
        "n_frames": 10,
        "bands": {"Alpha": {"mean": 10.0, "std": 1.0}},
        "indices": {},
        "band_names": ["Alpha"],
    }
    b = {
        "label": "active",
        "n_frames": 10,
        "bands": {"Alpha": {"mean": 5.0, "std": 1.0}},
        "indices": {},
        "band_names": ["Alpha"],
    }
    result = compare_summaries(a, b)
    headline = result["headline"]
    assert "Alpha power" in headline
    assert "lower in B" in headline  # since b.mean < a.mean


def test_caveat():
    """compare_summaries should include the within-session caveat."""
    a = {"label": "a", "n_frames": 1, "bands": {}, "indices": {}, "band_names": []}
    b = {"label": "b", "n_frames": 1, "bands": {}, "indices": {}, "band_names": []}
    result = compare_summaries(a, b)
    assert "caveat" in result
    assert "within-session" in result["caveat"].lower()
