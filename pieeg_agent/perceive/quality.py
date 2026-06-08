"""Per-channel signal-quality assessment.

The agent must never confidently report "you are relaxed" from a railed or
disconnected electrode. This module turns each channel's raw window (and the
PSD already computed by :mod:`features`) into a small, honest verdict:

* **rms** — broadband amplitude in µV (after DC removal).
* **line_ratio** — power at the mains frequency vs. the broadband median;
  large values mean 50/60 Hz contamination.
* **rail_frac** — fraction of samples pinned near the ADC rails (clipping).
* **status / score** — a one-word verdict and a 0…1 goodness used by the
  state estimator and event detector.

These are descriptive, single-window metrics — cheap, interpretable, and good
enough to flag the failure modes that actually occur (dead lead, motion,
mains hum). They are not a substitute for impedance measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelQuality:
    """Quality verdict for a single channel."""

    index: int
    label: str
    rms: float            # µV
    line_ratio: float     # mains power / broadband median (dimensionless)
    rail_frac: float      # fraction of samples near the rails
    status: str           # "good" | "flat" | "rail" | "noisy" | "line"
    score: float          # 0 (unusable) … 1 (clean)


@dataclass
class SignalQuality:
    """Quality across all channels for one window."""

    timestamp: float
    channels: list[ChannelQuality]
    overall: float        # mean channel score, 0…1

    @property
    def worst(self) -> ChannelQuality | None:
        return min(self.channels, key=lambda c: c.score) if self.channels else None

    def bad_channels(self) -> list[ChannelQuality]:
        """Channels whose verdict is anything other than 'good'."""
        return [c for c in self.channels if c.status != "good"]


class QualityMonitor:
    """Scores every channel of a raw window into a :class:`SignalQuality`."""

    def __init__(
        self,
        sample_rate: float,
        *,
        mains_hz: float = 50.0,
        flat_uv: float = 2.0,
        noisy_uv: float = 250.0,
        rail_uv: float = 150_000.0,
        line_threshold: float = 6.0,
    ):
        self.sample_rate = float(sample_rate)
        self.mains_hz = float(mains_hz)
        self.flat_uv = float(flat_uv)
        self.noisy_uv = float(noisy_uv)
        self.rail_uv = float(rail_uv)
        self.line_threshold = float(line_threshold)

    def compute(
        self,
        data: np.ndarray,
        labels: list[str],
        timestamp: float,
        bp=None,
    ) -> SignalQuality:
        """Assess ``data`` (n_samples, n_channels).

        ``bp`` is the matching :class:`~pieeg_agent.perceive.features.BandPowers`
        if available — its PSD is reused for the mains-noise ratio so no second
        FFT is needed.
        """
        x = np.asarray(data, dtype=np.float64)
        n, n_ch = x.shape if x.ndim == 2 else (0, 0)
        channels: list[ChannelQuality] = []

        centred = x - x.mean(axis=0, keepdims=True) if n else x
        rms_all = np.sqrt(np.mean(centred**2, axis=0)) if n else np.zeros(n_ch)
        rail_all = (
            np.mean(np.abs(x) >= self.rail_uv, axis=0) if n else np.zeros(n_ch)
        )
        line_all = self._line_ratios(bp, n_ch)

        for ch in range(n_ch):
            label = labels[ch] if ch < len(labels) else f"Ch{ch}"
            rms = float(rms_all[ch])
            rail = float(rail_all[ch])
            line = float(line_all[ch])
            status, score = self._verdict(rms, rail, line)
            channels.append(
                ChannelQuality(ch, label, rms, line, rail, status, score)
            )

        overall = float(np.mean([c.score for c in channels])) if channels else 0.0
        return SignalQuality(timestamp=timestamp, channels=channels, overall=overall)

    # ── internals ───────────────────────────────────────────────────────
    def _verdict(self, rms: float, rail: float, line: float) -> tuple[str, float]:
        """Map raw metrics to a status word and a 0…1 score (worst wins)."""
        if rail > 0.005:
            return "rail", max(0.0, 1.0 - rail * 10.0) * 0.4
        if rms < self.flat_uv:
            return "flat", min(rms / self.flat_uv, 1.0) * 0.3
        if rms > self.noisy_uv:
            return "noisy", min(self.noisy_uv / rms, 1.0) * 0.6
        if line > self.line_threshold:
            return "line", min(self.line_threshold / line, 1.0) * 0.7
        return "good", 1.0

    def _line_ratios(self, bp, n_ch: int) -> np.ndarray:
        """Per-channel mains-power / broadband-median ratio from the PSD."""
        if bp is None or getattr(bp, "psd", None) is None:
            return np.zeros(n_ch)
        freqs = bp.freqs
        psd = bp.psd  # (n_ch, n_freq)
        if psd.shape[0] != n_ch:
            return np.zeros(n_ch)

        mains = (freqs >= self.mains_hz - 1.0) & (freqs <= self.mains_hz + 1.0)
        broad = (freqs >= 2.0) & (freqs <= 45.0)
        # Exclude the mains line itself from the broadband reference.
        broad &= ~(
            (freqs >= self.mains_hz - 2.0) & (freqs <= self.mains_hz + 2.0)
        )
        if not mains.any() or not broad.any():
            return np.zeros(n_ch)

        line_power = psd[:, mains].max(axis=1)
        baseline = np.median(psd[:, broad], axis=1)
        baseline = np.where(baseline > 1e-12, baseline, 1e-12)
        return line_power / baseline
