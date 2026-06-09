"""Tests for streaming artifact detection (blink / jaw / motion)."""

import numpy as np

from pieeg_agent.perceive.artifacts import ArtifactEvent, ArtifactMonitor

FS = 250.0


def _feed(monitor: ArtifactMonitor, signal: np.ndarray, *, chunk: int = 32):
    """Stream a (N, n_ch) signal through the monitor in small timestamped chunks."""
    events: list[ArtifactEvent] = []
    n = signal.shape[0]
    for start in range(0, n, chunk):
        block = signal[start : start + chunk].astype(np.float32)
        ts = (np.arange(start, start + block.shape[0]) / FS).astype(np.float64)
        events.extend(monitor.update(block, ts))
    return events


def _baseline(seconds: float, n_ch: int = 8, std: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    return rng.normal(0.0, std, size=(n, n_ch))


def _gaussian_bump(n: int, center: int, sigma_s: float, amp: float):
    t = np.arange(n) / FS
    return amp * np.exp(-(((t - center / FS) / sigma_s) ** 2))


def test_clean_baseline_has_no_artifacts():
    mon = ArtifactMonitor(FS, [f"Ch{i}" for i in range(8)])
    events = _feed(mon, _baseline(5.0))
    assert events == []


def test_blink_is_detected():
    sig = _baseline(4.0)
    # A ~120 ms frontal deflection on channels 0/1, well after warm-up.
    n = sig.shape[0]
    bump = _gaussian_bump(n, center=int(3.0 * FS), sigma_s=0.05, amp=220.0)
    sig[:, 0] += bump
    sig[:, 1] += bump
    mon = ArtifactMonitor(FS, [f"Ch{i}" for i in range(8)])
    events = _feed(mon, sig)
    blinks = [e for e in events if e.type in ("blink", "blink_double")]
    assert blinks, "expected at least one blink"
    assert all(0.0 <= e.confidence <= 1.0 for e in events)
    assert blinks[0].channel == "frontal"


def test_double_blink_classified():
    sig = _baseline(5.0)
    n = sig.shape[0]
    for c in (3.0, 3.3):  # two blinks 300 ms apart
        bump = _gaussian_bump(n, center=int(c * FS), sigma_s=0.04, amp=220.0)
        sig[:, 0] += bump
        sig[:, 1] += bump
    mon = ArtifactMonitor(FS, [f"Ch{i}" for i in range(8)])
    events = _feed(mon, sig)
    assert any(e.type == "blink_double" for e in events)


def test_jaw_clench_detected():
    sig = _baseline(4.0)
    n = sig.shape[0]
    rng = np.random.default_rng(1)
    burst = np.zeros(n)
    lo, hi = int(3.0 * FS), int(3.3 * FS)  # 300 ms high-band burst
    burst[lo:hi] = rng.normal(0.0, 120.0, size=hi - lo)
    sig[:, 4] += burst
    mon = ArtifactMonitor(FS, [f"Ch{i}" for i in range(8)])
    events = _feed(mon, sig)
    assert any(e.type == "jaw_clench" for e in events)


def test_event_to_dict_is_json_friendly():
    ev = ArtifactEvent(1.0, "blink", "frontal", 120.0, 0.8, "eye blink")
    d = ev.to_dict()
    assert d["type"] == "blink" and d["duration_ms"] == 120.0
    assert set(d) >= {"timestamp", "type", "channel", "confidence"}


def test_timestamp_dedup_skips_seen_samples():
    mon = ArtifactMonitor(FS, [f"Ch{i}" for i in range(8)])
    block = _baseline(1.0).astype(np.float32)
    ts = (np.arange(block.shape[0]) / FS).astype(np.float64)
    mon.update(block, ts)
    # Re-feeding the same window (older timestamps) must process nothing new.
    assert mon.update(block, ts) == []
