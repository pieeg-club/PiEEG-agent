"""Tests for advanced spectral analysis (IAF, 1/f slope, asymmetry, entropy)."""

import numpy as np

from pieeg_agent.decode import spectral
from pieeg_agent.perceive.features import BAND_NAMES, BandPowerExtractor

FS = 250.0
N = 512


def _sine(freq: float, amp: float = 20.0, n_ch: int = 4):
    t = np.arange(N) / FS
    sig = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([sig] * n_ch, axis=1).astype(np.float32)


def test_individual_alpha_peak_near_10hz():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0), 1.0)
    peak = spectral.individual_alpha_peak(bp.psd.mean(axis=0), bp.freqs)
    assert peak is not None
    assert abs(peak["peak_hz"] - 10.0) < 0.7


def test_alpha_peak_none_outside_band():
    bp = BandPowerExtractor(FS, N).compute(_sine(20.0), 1.0)
    # A pure 20 Hz tone has no interior maximum in the 7-13 Hz alpha band.
    assert spectral.individual_alpha_peak(bp.psd.mean(axis=0), bp.freqs) is None


def test_aperiodic_slope_recovers_exponent():
    freqs = np.linspace(1.0, 45.0, 200)
    psd = freqs ** (-2.0)  # exact 1/f^2
    fit = spectral.aperiodic_fit(psd, freqs)
    assert fit is not None
    assert abs(fit["exponent"] - 2.0) < 0.05
    assert fit["r2"] > 0.99


def test_spectral_entropy_flat_vs_peaked():
    freqs = np.linspace(1.0, 45.0, 200)
    flat = np.ones_like(freqs)
    peaked = np.zeros_like(freqs)
    peaked[100] = 1.0
    assert spectral.spectral_entropy(flat, freqs) > 0.95
    assert spectral.spectral_entropy(peaked, freqs) < 0.1


def test_theta_beta_ratio():
    assert abs(spectral.theta_beta_ratio({"Theta": 4.0, "Beta": 2.0}) - 2.0) < 1e-6


def test_frontal_alpha_asymmetry_pairs():
    n_bands = len(BAND_NAMES)
    per_ch = np.ones((2, n_bands))
    ai = BAND_NAMES.index("Alpha")
    per_ch[0, ai] = 1.0   # F3 (left)
    per_ch[1, ai] = np.e  # F4 (right) → ln(R)-ln(L) = +1
    out = spectral.frontal_alpha_asymmetry(per_ch, ["F3", "F4"])
    assert out is not None
    assert abs(out["mean"] - 1.0) < 1e-3


def test_frontal_alpha_asymmetry_none_without_pairs():
    per_ch = np.ones((2, len(BAND_NAMES)))
    assert spectral.frontal_alpha_asymmetry(per_ch, ["A", "B"]) is None


def test_analyze_spectrum_smoke():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0), 1.0)
    report = spectral.analyze_spectrum(bp, ["F3", "F4", "C3", "C4"])
    assert report["dominant_band"] == "Alpha"
    assert "alpha_peak" in report and "aperiodic" in report
    assert report["scope"] == "channel-average"


def test_analyze_spectrum_single_channel():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0), 1.0)
    report = spectral.analyze_spectrum(bp, ["F3", "F4", "C3", "C4"], channel=2)
    assert report["scope"] == "C3"
    # Asymmetry needs the montage; single-channel reports omit it.
    assert "frontal_alpha_asymmetry" not in report


def test_analyze_spectrum_bad_channel():
    bp = BandPowerExtractor(FS, N).compute(_sine(10.0), 1.0)
    assert "error" in spectral.analyze_spectrum(bp, [], channel=99)
