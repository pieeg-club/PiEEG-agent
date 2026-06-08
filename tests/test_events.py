"""Tests for debounced event detection (Schmitt trigger + minimum dwell)."""

from pieeg_agent.perceive.events import EventDetector, _Schmitt
from pieeg_agent.perceive.state import NeuralState

_REL = {"Delta": 0.2, "Theta": 0.2, "Alpha": 0.2, "Beta": 0.2, "Gamma": 0.2}


def mk_state(t, focus=0.5, relax=0.5, engage=0.5, q=1.0, bad=()):
    return NeuralState(
        timestamp=t,
        rel_bands=dict(_REL),
        dominant_band="Alpha",
        focus=focus,
        relax=relax,
        engagement=engage,
        signal_quality=q,
        n_channels=4,
        bad_channels=bad,
    )


def test_schmitt_requires_dwell():
    s = _Schmitt(0.7, 0.3, 2.0, "hi", "lo", "x")
    assert s.update(0.8, 0.0) is None     # crossing registered
    assert s.update(0.8, 1.0) is None     # < 2 s dwell
    ev = s.update(0.8, 2.0)               # dwell satisfied
    assert ev is not None and ev.type == "hi"


def test_brief_spike_no_event():
    s = _Schmitt(0.7, 0.3, 2.0, "hi", "lo", "x")
    assert s.update(0.8, 0.0) is None
    assert s.update(0.2, 0.5) is None     # fell back before dwell
    assert s.state is False               # never latched


def test_focus_high_then_low_single_events():
    det = EventDetector(hi=0.7, lo=0.3, min_dwell=2.0)
    rising = []
    for t in range(0, 6):
        rising += det.update(mk_state(float(t), focus=0.9))
    types = [e.type for e in rising]
    assert types.count("focus_high") == 1

    falling = []
    for t in range(6, 12):
        falling += det.update(mk_state(float(t), focus=0.05))
    assert "focus_low" in [e.type for e in falling]


def test_quality_drop_is_warning():
    det = EventDetector(quality_floor=0.5, min_dwell=1.0)
    events = []
    for t in range(0, 5):
        events += det.update(mk_state(float(t), q=0.2, bad=("Ch2",)))
    drops = [e for e in events if e.type == "quality_drop"]
    assert drops and drops[0].severity == "warn"
    assert "Ch2" in drops[0].detail


def test_quality_recovers_to_info():
    det = EventDetector(quality_floor=0.5, min_dwell=1.0)
    for t in range(0, 5):
        det.update(mk_state(float(t), q=0.2))
    recover = []
    for t in range(5, 12):
        recover += det.update(mk_state(float(t), q=0.95))
    oks = [e for e in recover if e.type == "quality_ok"]
    assert oks and oks[0].severity == "info"
