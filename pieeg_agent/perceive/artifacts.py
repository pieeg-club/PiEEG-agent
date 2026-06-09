"""Streaming artifact / transient detection — blinks, jaw clench, motion.

The band-power tiers (``features`` → ``state``) describe *sustained* spectral
shape. This module catches the other half of what an EEG sees: **time-domain
transients** — sub-second events the agent should notice ("you blinked twice",
"that was a jaw clench"). Each detector runs a streaming envelope follower with
a Schmitt onset/offset trigger over the newest raw samples, so detection is at
sample resolution without a second thread and without reprocessing the
overlapping FFT windows.

The approach is ported from two PiEEG-community experiences:

* **Blink Runner** — rectified EOG envelope, onset/offset hysteresis, and
  single/double classification by inter-blink timing.
* **Face Trainer** — high-band (EMG) envelope bursts for jaw clench / muscle
  tension.

Honesty note: these are descriptive transient detectors with adaptive
thresholds, not a calibrated EOG/EMG montage. They flag the failure/affordance
modes that actually occur (blinks, clenches, gross motion) and report a
confidence, but the agent should treat them as cues, not ground truth.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass
class ArtifactEvent:
    """A single detected transient."""

    timestamp: float        # wall-clock time of the event (offset instant)
    type: str               # "blink" | "blink_double" | "jaw_clench" | "motion"
    channel: str            # montage the event came from ("frontal"/"global"/label)
    duration_ms: float
    confidence: float       # 0..1, peak excursion above the adaptive threshold
    detail: str = ""
    severity: str = "info"  # "info" | "warn"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "channel": self.channel,
            "duration_ms": round(self.duration_ms, 1),
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
            "severity": self.severity,
        }


def _hp_coeff(cutoff_hz: float, dt: float) -> float:
    """One-pole high-pass coefficient for a cutoff in Hz at sample period ``dt``."""
    rc = 1.0 / (2.0 * math.pi * max(cutoff_hz, _EPS))
    return rc / (rc + dt)


def _lp_alpha(tau_s: float, dt: float) -> float:
    """EMA smoothing factor for an envelope with time constant ``tau_s``."""
    if tau_s <= 0:
        return 1.0
    return 1.0 - math.exp(-dt / tau_s)


class _EnvelopeBank:
    """Per-channel streaming envelope: high-pass → rectify → smooth.

    Holds one filter state per channel and processes a short chunk of new
    samples, returning the rectified, smoothed envelope (same shape). The
    recursion is per-sample but vectorised across channels, so the hot loop is
    a handful of NumPy ops over the (few-dozen-sample) chunk.
    """

    def __init__(
        self,
        n_channels: int,
        hp_cutoff: float,
        env_tau: float,
        dt: float,
        *,
        order: int = 1,
    ):
        self._a = _hp_coeff(hp_cutoff, dt)
        self._alpha = _lp_alpha(env_tau, dt)
        self._order = max(1, int(order))
        # One (x_prev, y_prev) state pair per cascaded high-pass stage. A 2nd
        # order cascade rolls off at 12 dB/oct, so a large low-frequency blink
        # cannot leak into a high-frequency (EMG) band.
        self._x_prev = [np.zeros(n_channels, dtype=np.float64) for _ in range(self._order)]
        self._y_prev = [np.zeros(n_channels, dtype=np.float64) for _ in range(self._order)]
        self._env = np.zeros(n_channels, dtype=np.float64)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a ``(m, n_channels)`` chunk → ``(m, n_channels)`` envelope."""
        m = chunk.shape[0]
        out = np.empty_like(chunk, dtype=np.float64)
        a, alpha = self._a, self._alpha
        xs, ys, env = self._x_prev, self._y_prev, self._env
        for i in range(m):
            s = chunk[i]
            for k in range(self._order):
                y = a * (ys[k] + s - xs[k])
                xs[k] = s
                ys[k] = y
                s = y
            env = env + alpha * (np.abs(s) - env)
            out[i] = env
        self._x_prev = [np.array(v, dtype=np.float64) for v in xs]
        self._y_prev = [np.array(v, dtype=np.float64) for v in ys]
        self._env = np.array(env, dtype=np.float64)
        return out


@dataclass
class _Excursion:
    """A completed above-threshold excursion of a scalar envelope."""

    t_onset: float
    t_offset: float
    peak: float
    confidence: float

    @property
    def duration_ms(self) -> float:
        return (self.t_offset - self.t_onset) * 1000.0


class _SchmittTransient:
    """Adaptive onset/offset trigger over a scalar envelope signal.

    Tracks a slow baseline and spread (EMA of the envelope and of its absolute
    deviation), and fires when the envelope rises ``k_on`` spreads above the
    baseline, releasing at ``k_off``. The baseline is frozen while an excursion
    is active so the event itself does not inflate it. Each completed
    excursion is returned with a duration and a 0..1 confidence.
    """

    def __init__(
        self,
        *,
        dt: float,
        k_on: float = 4.0,
        k_off: float = 2.0,
        floor: float = 1.0,
        base_tau: float = 4.0,
        spread_tau: float = 4.0,
        warmup_s: float = 1.0,
        min_ms: float = 30.0,
        max_ms: float = 3000.0,
    ):
        self._dt = dt
        self._k_on = k_on
        self._k_off = k_off
        self._floor = floor
        self._base_alpha = _lp_alpha(base_tau, dt)
        self._spread_alpha = _lp_alpha(spread_tau, dt)
        self._warmup_n = int(warmup_s / dt)
        self._min_ms = min_ms
        self._max_ms = max_ms

        self._baseline: float | None = None
        self._spread = 0.0
        self._active = False
        self._t_onset = 0.0
        self._peak = 0.0
        self._n = 0

    def step(self, value: float, t: float) -> _Excursion | None:
        """Feed one envelope sample; return a completed excursion or ``None``."""
        self._n += 1
        if self._baseline is None:
            self._baseline = value
        if not self._active:
            # Track baseline + spread only while idle.
            self._baseline += self._base_alpha * (value - self._baseline)
            dev = abs(value - self._baseline)
            self._spread += self._spread_alpha * (dev - self._spread)

        thr_on = self._baseline + self._k_on * self._spread + self._floor
        thr_off = self._baseline + self._k_off * self._spread + 0.5 * self._floor

        if self._n < self._warmup_n:
            return None

        if not self._active:
            if value >= thr_on:
                self._active = True
                self._t_onset = t
                self._peak = value
            return None

        # Active — track the peak, wait for release.
        self._peak = max(self._peak, value)
        if value > thr_off and (t - self._t_onset) * 1000.0 <= self._max_ms:
            return None

        # Release.
        self._active = False
        dur_ms = (t - self._t_onset) * 1000.0
        if dur_ms < self._min_ms:
            return None
        span = max(thr_on - self._baseline, _EPS)
        confidence = float(min(1.0, (self._peak - thr_off) / (2.0 * span)))
        return _Excursion(
            t_onset=self._t_onset,
            t_offset=t,
            peak=self._peak,
            confidence=max(0.0, confidence),
        )


class ArtifactMonitor:
    """Detects blinks, jaw clenches and gross motion from raw EEG windows.

    Fed the most-recent raw window each cascade tick, it processes only the
    samples it has not seen before (deduplicated by timestamp), so it can share
    the cascade's overlapping windows without double-counting. Returns a list
    of :class:`ArtifactEvent` for any transients that completed in the chunk.
    """

    def __init__(
        self,
        sample_rate: float,
        channel_labels: list[str] | None = None,
        *,
        frontal: tuple[int, ...] | None = None,
        double_blink_ms: tuple[float, float] = (60.0, 600.0),
    ):
        self._sr = float(sample_rate) or 250.0
        self._dt = 1.0 / self._sr
        self._labels = list(channel_labels or [])
        # Frontal channels carry blinks; default to the first two electrodes
        # (typically frontal on a PiEEG montage) when no labels pin them down.
        self._frontal = frontal or self._guess_frontal()
        self._dbl_lo, self._dbl_hi = double_blink_ms

        # Low-band envelope (~5 Hz HP) for blinks/motion; high-band (steeper
        # 2nd-order ~30 Hz HP) for EMG / jaw clench. Built lazily once the
        # channel count is known.
        self._low: _EnvelopeBank | None = None
        self._high: _EnvelopeBank | None = None
        self._n_ch = 0

        self._blink = _SchmittTransient(
            dt=self._dt, k_on=4.0, k_off=2.5, floor=1.5,
            min_ms=40.0, max_ms=900.0,
        )
        self._jaw = _SchmittTransient(
            dt=self._dt, k_on=4.5, k_off=2.0, floor=6.0,
            min_ms=120.0, max_ms=3000.0,
        )
        self._motion = _SchmittTransient(
            dt=self._dt, k_on=6.0, k_off=3.0, floor=3.0,
            min_ms=80.0, max_ms=3000.0,
        )
        self._last_blink_onset: float | None = None
        self._last_ts: float = -math.inf

    # ── public surface ──────────────────────────────────────────────────
    def update(self, data: np.ndarray, timestamps: np.ndarray) -> list[ArtifactEvent]:
        """Process the unseen tail of ``data`` (n, n_ch) with per-sample ``timestamps``."""
        if data.ndim != 2 or data.shape[0] == 0:
            return []
        n, n_ch = data.shape
        if self._low is None or n_ch != self._n_ch:
            self._init_filters(n_ch)

        ts = np.asarray(timestamps, dtype=np.float64)
        fresh = ts > self._last_ts
        if not fresh.any():
            return []
        x = np.asarray(data, dtype=np.float64)[fresh]
        ts = ts[fresh]
        self._last_ts = float(ts[-1])

        low = self._low.process(x)          # (m, n_ch)
        high = self._high.process(x)        # (m, n_ch)

        frontal_sig = low[:, self._frontal].mean(axis=1)
        # Motion is a *spatially shared* disturbance. The median across channels
        # only rises when most electrodes move together, so a one- or two-channel
        # event (a blink, a single-electrode pop) cannot masquerade as motion.
        motion_sig = np.median(low, axis=1)
        jaw_sig = high.max(axis=1)
        jaw_idx = int(high.mean(axis=0).argmax())

        events: list[ArtifactEvent] = []
        for i in range(x.shape[0]):
            t = float(ts[i])
            blink = self._blink.step(float(frontal_sig[i]), t)
            if blink is not None:
                events.append(self._classify_blink(blink))
            jaw = self._jaw.step(float(jaw_sig[i]), t)
            if jaw is not None:
                events.append(self._make_jaw(jaw, jaw_idx))
            motion = self._motion.step(float(motion_sig[i]), t)
            if motion is not None:
                events.append(self._make_motion(motion))
        return events

    def reset(self) -> None:
        self._low = self._high = None
        self._last_ts = -math.inf
        self._last_blink_onset = None

    # ── internals ───────────────────────────────────────────────────────
    def _init_filters(self, n_ch: int) -> None:
        self._n_ch = n_ch
        # Blink/motion band: a ~5 Hz high-pass keeps the transient edges while
        # discarding drift and the long restoration tail that would otherwise
        # smear two close blinks together; a short envelope keeps timing crisp.
        self._low = _EnvelopeBank(n_ch, hp_cutoff=5.0, env_tau=0.02, dt=self._dt)
        # Jaw/EMG band: a steeper 2nd-order ~30 Hz high-pass so a large, slow
        # blink does not leak into the muscle band and read as a clench.
        self._high = _EnvelopeBank(n_ch, hp_cutoff=30.0, env_tau=0.02, dt=self._dt, order=2)
        if not self._frontal or max(self._frontal) >= n_ch:
            self._frontal = tuple(range(min(2, n_ch)))

    def _guess_frontal(self) -> tuple[int, ...]:
        wanted = ("fp1", "fp2", "af7", "af8", "fpz")
        hits = [
            i for i, lab in enumerate(self._labels)
            if lab.lower() in wanted
        ]
        return tuple(hits) if hits else (0, 1)

    def _classify_blink(self, exc: _Excursion) -> ArtifactEvent:
        is_double = False
        if self._last_blink_onset is not None:
            gap = (exc.t_onset - self._last_blink_onset) * 1000.0
            if self._dbl_lo <= gap <= self._dbl_hi:
                is_double = True
        self._last_blink_onset = exc.t_onset
        if is_double:
            return ArtifactEvent(
                timestamp=exc.t_offset, type="blink_double", channel="frontal",
                duration_ms=exc.duration_ms, confidence=exc.confidence,
                detail="two blinks in quick succession",
            )
        return ArtifactEvent(
            timestamp=exc.t_offset, type="blink", channel="frontal",
            duration_ms=exc.duration_ms, confidence=exc.confidence,
            detail=f"eye blink ({exc.duration_ms:.0f} ms)",
        )

    def _make_jaw(self, exc: _Excursion, ch_idx: int) -> ArtifactEvent:
        label = self._labels[ch_idx] if ch_idx < len(self._labels) else f"Ch{ch_idx}"
        return ArtifactEvent(
            timestamp=exc.t_offset, type="jaw_clench", channel=label,
            duration_ms=exc.duration_ms, confidence=exc.confidence,
            detail=f"muscle / jaw burst near {label}", severity="warn",
        )

    def _make_motion(self, exc: _Excursion) -> ArtifactEvent:
        return ArtifactEvent(
            timestamp=exc.t_offset, type="motion", channel="global",
            duration_ms=exc.duration_ms, confidence=exc.confidence,
            detail="broadband motion / electrode disturbance", severity="warn",
        )
