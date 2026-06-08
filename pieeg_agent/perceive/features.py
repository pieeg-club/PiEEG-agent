"""T1 feature extraction — sliding-window band powers.

This is the cascade's first reduction: a high-rate window of raw samples
(``fft_size`` × ``n_channels`` float32) collapses into five band-power numbers
per channel at a few hertz. Everything downstream (state, events, the LLM)
reasons about *these*, never the raw stream.

The math mirrors PiEEG-server's ``osc_vrchat`` bridge (Hanning window + real
FFT + the canonical Delta…Gamma bands) but produces a properly scaled
one-sided power-spectral density in µV²/Hz, so band *ratios* — which is what
the focus/relax indices are built from — are physically meaningful and
window-length independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical EEG bands (Hz). Identical to the dashboard / osc_vrchat bridge so
# the agent speaks the same language as the rest of PiEEG.
BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 100.0),
}
BAND_NAMES: tuple[str, ...] = tuple(BANDS)


@dataclass
class BandPowers:
    """One feature frame: band powers + the PSD that produced them."""

    timestamp: float                 # wall-clock time of the newest sample
    bands: dict[str, float]          # channel-averaged µV²/Hz per band
    per_channel: np.ndarray          # (n_channels, n_bands) µV²/Hz
    psd: np.ndarray                  # (n_channels, n_freq) one-sided PSD
    freqs: np.ndarray                # (n_freq,) bin centre frequencies (Hz)
    n_samples: int                   # window length the FFT used
    n_channels: int

    @property
    def total(self) -> float:
        """Sum of channel-averaged band powers (the relative-power denominator)."""
        return float(sum(self.bands.values()))

    def relative(self) -> dict[str, float]:
        """Band powers normalised to sum≈1 — scale-invariant spectral shape."""
        tot = self.total or 1e-12
        return {b: v / tot for b, v in self.bands.items()}

    def dominant(self) -> str:
        """Name of the strongest band (by channel-averaged power)."""
        return max(self.bands, key=self.bands.__getitem__)


class BandPowerExtractor:
    """Turns a raw sample window into :class:`BandPowers`.

    One real FFT per channel per call. Precomputes the window, frequency axis
    and per-band masks so the hot path is a single ``rfft`` plus masked means.
    """

    def __init__(self, sample_rate: float, fft_size: int = 512):
        if fft_size < 8:
            raise ValueError("fft_size must be >= 8")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self._window = np.hanning(self.fft_size).astype(np.float64)
        # PSD normalisation: a one-sided density estimate in µV²/Hz needs the
        # window's power and the sample rate. (Welch single-segment form.)
        self._win_norm = float(np.sum(self._window**2)) * self.sample_rate
        self._freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        self._masks = {
            band: (self._freqs >= lo) & (self._freqs < hi)
            for band, (lo, hi) in BANDS.items()
        }

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs

    def compute(self, data: np.ndarray, timestamp: float) -> BandPowers | None:
        """Extract band powers from the most-recent ``fft_size`` samples.

        ``data`` is ``(n_samples, n_channels)``. Returns ``None`` while fewer
        than ``fft_size`` samples are available (warm-up).
        """
        if data.ndim != 2:
            raise ValueError("data must be 2-D (n_samples, n_channels)")
        n, n_ch = data.shape
        if n < self.fft_size:
            return None

        window = data[-self.fft_size :, :].astype(np.float64)
        # Remove per-channel DC so slow drift / electrode offset does not leak
        # into the Delta band.
        window -= window.mean(axis=0, keepdims=True)
        windowed = window * self._window[:, None]

        spec = np.fft.rfft(windowed, axis=0)            # (n_freq, n_ch)
        psd = (np.abs(spec) ** 2) / self._win_norm      # µV²/Hz
        if psd.shape[0] > 2:
            psd[1:-1, :] *= 2.0                          # one-sided correction
        psd = psd.T                                      # (n_ch, n_freq)

        per_channel = np.empty((n_ch, len(BAND_NAMES)), dtype=np.float64)
        for j, band in enumerate(BAND_NAMES):
            mask = self._masks[band]
            per_channel[:, j] = psd[:, mask].mean(axis=1) if mask.any() else 0.0

        bands = {
            band: float(per_channel[:, j].mean())
            for j, band in enumerate(BAND_NAMES)
        }
        return BandPowers(
            timestamp=timestamp,
            bands=bands,
            per_channel=per_channel,
            psd=psd,
            freqs=self._freqs,
            n_samples=self.fft_size,
            n_channels=n_ch,
        )
