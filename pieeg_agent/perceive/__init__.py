"""Perception cascade — the high-rate reduction from raw EEG to language.

This package turns the ring buffer's raw samples into progressively smaller,
more meaningful representations:

* :mod:`features` — sliding-window band powers (T1, a few Hz).
* :mod:`quality`  — per-channel signal-quality verdicts.
* :mod:`state`    — a smoothed, normalised :class:`NeuralState` (T2, ~1 Hz).
* :mod:`events`   — sparse, debounced :class:`NeuralEvent`s (T3).
* :mod:`cascade`  — the thread that wires them together over an inlet's ring.

Nothing here imports an LLM; this is pure perception.
"""

from __future__ import annotations

from .cascade import CascadeConfig, PerceptionCascade
from .events import EventDetector, NeuralEvent
from .features import BANDS, BAND_NAMES, BandPowerExtractor, BandPowers
from .quality import ChannelQuality, QualityMonitor, SignalQuality
from .state import NeuralState, StateEstimator

__all__ = [
    "BANDS",
    "BAND_NAMES",
    "BandPowerExtractor",
    "BandPowers",
    "QualityMonitor",
    "SignalQuality",
    "ChannelQuality",
    "StateEstimator",
    "NeuralState",
    "EventDetector",
    "NeuralEvent",
    "PerceptionCascade",
    "CascadeConfig",
]
