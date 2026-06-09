"""The feature vector shared by training and live scoring.

A trainable pattern is only as good as the numbers it sees. This module turns
one cascade frame — a :class:`~pieeg_agent.perceive.features.BandPowers` plus
the raw window that produced it — into a single fixed-length vector that both
calibration and the live detector consume, so what the agent *learns* is
exactly what it later *scores*.

The layout follows the PiEEG **Face Trainer** experience: a handful of cheap,
log-scaled features **per channel**, kept in contiguous per-channel blocks so a
group-lasso penalty and a channel-importance read fall straight out of the
weight vector. Per channel we take

* ``logDelta … logGamma`` — the five band powers (spectral shape), and
* ``logRMS`` — broadband amplitude (overall drive), and
* ``logLineLength`` — mean absolute sample-to-sample change (high-frequency /
  complexity content the band split alone blurs).

Everything is ``log10``-scaled because band powers and amplitudes are roughly
log-normal; in log space they are far closer to Gaussian, which is what the
linear detector and the Cohen's-d ranking both assume. Standardisation (mean /
spread removal) is deliberately *not* done here — that is the calibrator's job,
since it depends on the rest baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perceive.features import BAND_NAMES

_EPS = 1e-9

# Per-channel feature names, in the order they appear inside each channel block.
PER_CHANNEL_FEATURES: tuple[str, ...] = (
    *(f"log{b}" for b in BAND_NAMES),  # logDelta, logTheta, logAlpha, logBeta, logGamma
    "logRMS",
    "logLineLength",
)
BLOCK = len(PER_CHANNEL_FEATURES)  # features per channel


@dataclass(frozen=True)
class FeatureLayout:
    """Describes a feature vector: which channels, which per-channel features.

    Holds no data — just the bookkeeping that maps a flat feature index back to
    a channel (for channel importance) and groups indices by channel (for the
    group-lasso penalty).
    """

    channel_labels: tuple[str, ...]
    per_channel: tuple[str, ...] = PER_CHANNEL_FEATURES

    @property
    def n_channels(self) -> int:
        return len(self.channel_labels)

    @property
    def block(self) -> int:
        return len(self.per_channel)

    @property
    def dim(self) -> int:
        return self.n_channels * self.block

    @property
    def names(self) -> tuple[str, ...]:
        """Flat feature names, e.g. ``"Fp1/logAlpha"``."""
        return tuple(
            f"{ch}/{feat}"
            for ch in self.channel_labels
            for feat in self.per_channel
        )

    def channel_of(self, index: int) -> int:
        """Channel index that flat feature ``index`` belongs to."""
        if not 0 <= index < self.dim:
            raise IndexError(index)
        return index // self.block

    def channel_groups(self) -> list[tuple[int, ...]]:
        """Flat indices grouped per channel — the group-lasso blocks."""
        return [
            tuple(range(c * self.block, (c + 1) * self.block))
            for c in range(self.n_channels)
        ]


def _line_length(window: np.ndarray) -> np.ndarray:
    """Mean absolute sample-to-sample difference per channel (n_ch,)."""
    if window.shape[0] < 2:
        return np.zeros(window.shape[1], dtype=np.float64)
    return np.abs(np.diff(window, axis=0)).mean(axis=0)


def extract_frame(per_channel_bands: np.ndarray, raw_window: np.ndarray) -> np.ndarray:
    """Build the feature vector from band powers + the raw window.

    ``per_channel_bands`` is ``(n_ch, n_bands)`` µV²/Hz (``BandPowers.per_channel``)
    and ``raw_window`` is ``(n_samples, n_ch)`` raw samples. Returns a 1-D
    ``(n_ch * BLOCK,)`` vector laid out channel-by-channel.
    """
    bands = np.asarray(per_channel_bands, dtype=np.float64)
    raw = np.asarray(raw_window, dtype=np.float64)
    if bands.ndim != 2:
        raise ValueError("per_channel_bands must be 2-D (n_ch, n_bands)")
    if raw.ndim != 2:
        raise ValueError("raw_window must be 2-D (n_samples, n_ch)")
    n_ch = bands.shape[0]
    if raw.shape[1] != n_ch:
        raise ValueError("raw_window and band powers disagree on channel count")

    # Time-domain amplitude features, computed on the DC-removed window so a
    # standing electrode offset does not masquerade as drive.
    centred = raw - raw.mean(axis=0, keepdims=True)
    rms = np.sqrt((centred**2).mean(axis=0))        # (n_ch,)
    line = _line_length(centred)                    # (n_ch,)

    log_bands = np.log10(bands + _EPS)              # (n_ch, n_bands)
    log_rms = np.log10(rms + _EPS)[:, None]         # (n_ch, 1)
    log_line = np.log10(line + _EPS)[:, None]       # (n_ch, 1)

    per_ch = np.concatenate([log_bands, log_rms, log_line], axis=1)  # (n_ch, BLOCK)
    return per_ch.reshape(-1)                         # channel-major flat vector


def extract(bp, raw_window: np.ndarray) -> np.ndarray:
    """Convenience wrapper: feature vector straight from a :class:`BandPowers`."""
    return extract_frame(bp.per_channel, raw_window)


def layout_for(channel_labels: list[str] | tuple[str, ...] | None, n_channels: int) -> FeatureLayout:
    """Build a :class:`FeatureLayout`, inventing ``Ch{i}`` labels when missing."""
    labels = tuple(channel_labels or ())
    if len(labels) != n_channels:
        labels = tuple(f"Ch{i}" for i in range(n_channels))
    return FeatureLayout(channel_labels=labels)
