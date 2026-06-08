"""T2 NeuralState — a smoothed, language-sized snapshot at ~1 Hz.

Feature frames arrive several times a second and are noisy. The estimator
EMA-smooths the spectral shape, derives three interpretable indices, and
normalises them to 0…1 against a rolling within-session range so they are
readable without per-user calibration.

The indices follow the conventions used elsewhere in PiEEG:

* **focus**       (Beta+Gamma) / (Alpha+Theta+Delta) — fast-over-slow activity.
* **relax**       Alpha / (Alpha+Beta) — alpha dominance.
* **engagement**  Beta / (Alpha+Theta) — the classic Pope engagement index.

Honesty note: the 0…1 values are *within-session relative* positions, not
absolute or clinical measures. They say "high for you, right now", nothing
more.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import BAND_NAMES, BandPowers
from .quality import SignalQuality

_EPS = 1e-9


@dataclass
class NeuralState:
    """The agent's ~1 Hz view of the brain — small enough to hand to an LLM."""

    timestamp: float
    rel_bands: dict[str, float]          # smoothed relative band powers (sum≈1)
    dominant_band: str
    focus: float                         # 0…1, within-session relative
    relax: float                         # 0…1
    engagement: float                    # 0…1
    signal_quality: float                # 0…1
    n_channels: int
    bad_channels: tuple[str, ...] = ()
    warming_up: bool = False

    def summary(self) -> str:
        """One-line, language-sized description."""
        bands = " ".join(
            f"{b[0].lower()}{self.rel_bands.get(b, 0.0):.2f}" for b in BAND_NAMES
        )
        qnote = "clean" if not self.bad_channels else "check " + ",".join(self.bad_channels)
        return (
            f"focus {self.focus:.2f} relax {self.relax:.2f} "
            f"engage {self.engagement:.2f} | {bands} | dom {self.dominant_band} "
            f"| Q {self.signal_quality:.2f} ({qnote})"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "rel_bands": {k: round(v, 4) for k, v in self.rel_bands.items()},
            "dominant_band": self.dominant_band,
            "focus": round(self.focus, 3),
            "relax": round(self.relax, 3),
            "engagement": round(self.engagement, 3),
            "signal_quality": round(self.signal_quality, 3),
            "n_channels": self.n_channels,
            "bad_channels": list(self.bad_channels),
            "warming_up": self.warming_up,
        }


class _RollingNorm:
    """Within-session min/max normaliser over a sliding window.

    Returns 0.5 until it has seen enough spread to be meaningful, so indices
    don't swing wildly during the first few seconds.
    """

    def __init__(self, window: int = 600, warmup: int = 10):
        self._buf: deque[float] = deque(maxlen=window)
        self._warmup = warmup
        self._calibrated = False

    @property
    def calibrated(self) -> bool:
        """True once the window holds a non-degenerate range to normalise against.

        Until then ``norm`` returns 0.5 — a steady signal simply has no spread
        to rank a value within, and saying "middle" is the honest answer.
        """
        return self._calibrated

    def norm(self, x: float) -> float:
        self._buf.append(float(x))
        if len(self._buf) < self._warmup:
            return 0.5
        lo = min(self._buf)
        hi = max(self._buf)
        if hi - lo < _EPS:
            return 0.5
        self._calibrated = True
        return float(min(max((x - lo) / (hi - lo), 0.0), 1.0))


class StateEstimator:
    """Accumulates feature frames and emits a smoothed :class:`NeuralState`."""

    def __init__(self, *, ema_tau: float = 1.5, norm_window: int = 600):
        self._tau = float(ema_tau)
        self._ema: dict[str, float] | None = None
        self._quality = 1.0
        self._bad: tuple[str, ...] = ()
        self._n_channels = 0
        self._updates = 0
        self._n_focus = _RollingNorm(norm_window)
        self._n_relax = _RollingNorm(norm_window)
        self._n_engage = _RollingNorm(norm_window)

    def update(self, bp: BandPowers, q: SignalQuality, dt: float) -> None:
        """Fold one feature frame into the running estimate (feature rate)."""
        rel = bp.relative()
        alpha = 1.0 - np.exp(-dt / self._tau) if self._tau > 0 else 1.0
        if self._ema is None:
            self._ema = dict(rel)
        else:
            for b in BAND_NAMES:
                self._ema[b] += alpha * (rel[b] - self._ema[b])
        self._quality = q.overall
        self._bad = tuple(c.label for c in q.bad_channels())[:6]
        self._n_channels = bp.n_channels
        self._updates += 1

    def emit(self, timestamp: float) -> NeuralState | None:
        """Produce the current state (call at the state rate, ~1 Hz)."""
        if self._ema is None:
            return None
        rel = self._ema
        d = rel["Delta"]
        th = rel["Theta"]
        al = rel["Alpha"]
        be = rel["Beta"]
        ga = rel["Gamma"]

        focus_raw = (be + ga) / (al + th + d + _EPS)
        relax_raw = al / (al + be + _EPS)
        engage_raw = be / (al + th + _EPS)

        focus = self._n_focus.norm(focus_raw)
        relax = self._n_relax.norm(relax_raw)
        engage = self._n_engage.norm(engage_raw)
        calibrated = (
            self._n_focus.calibrated
            and self._n_relax.calibrated
            and self._n_engage.calibrated
        )

        return NeuralState(
            timestamp=timestamp,
            rel_bands=dict(rel),
            dominant_band=max(rel, key=rel.get),
            focus=focus,
            relax=relax,
            engagement=engage,
            signal_quality=self._quality,
            n_channels=self._n_channels,
            bad_channels=self._bad,
            warming_up=self._updates < 12 or not calibrated,
        )
