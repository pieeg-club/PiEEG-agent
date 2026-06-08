"""Tests for the band-power feature extractor (offline, synthetic signals)."""

import numpy as np

from pieeg_agent.perceive.features import BAND_NAMES, BandPowerExtractor

FS = 250.0
N = 512


def _sine(freq: float, n: int = N, fs: float = FS, amp: float = 20.0, n_ch: int = 2):
    t = np.arange(n) / fs
    sig = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([sig] * n_ch, axis=1).astype(np.float32)


def test_warmup_returns_none():
    ext = BandPowerExtractor(FS, N)
    assert ext.compute(np.zeros((N - 1, 2), np.float32), 0.0) is None


def test_alpha_dominates_for_10hz():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0), 1.0)
    assert bp is not None
    assert bp.dominant() == "Alpha"


def test_theta_dominates_for_6hz():
    bp = BandPowerExtractor(FS, N).compute(_sine(6.0), 1.0)
    assert bp.dominant() == "Theta"


def test_beta_dominates_for_20hz():
    bp = BandPowerExtractor(FS, N).compute(_sine(20.0), 1.0)
    assert bp.dominant() == "Beta"


def test_relative_sums_to_one():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0), 1.0)
    assert abs(sum(bp.relative().values()) - 1.0) < 1e-6


def test_dc_offset_does_not_inflate_delta():
    # A large DC offset must not masquerade as Delta power.
    sig = _sine(10.0) + 1000.0
    bp = BandPowerExtractor(FS, N).compute(sig, 1.0)
    assert bp.dominant() == "Alpha"


def test_per_channel_shape_and_count():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0, n_ch=4), 1.0)
    assert bp.per_channel.shape == (4, len(BAND_NAMES))
    assert bp.n_channels == 4
    assert bp.psd.shape[0] == 4
