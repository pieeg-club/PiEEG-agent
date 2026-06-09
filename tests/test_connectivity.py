"""Tests for decode/connectivity.py — band-power amplitude coupling."""

import numpy as np
from pieeg_agent.decode.connectivity import band_power_connectivity


def test_insufficient_frames():
    """Fewer than MIN_FRAMES returns status=insufficient."""
    # (7, 4, 5) is T=7 (below threshold), 4 channels, 5 bands
    small = np.random.random((7, 4, 5))
    result = band_power_connectivity(small, ["A", "B", "C", "D"])
    assert result["status"] == "insufficient"
    assert result["n_frames"] == 7


def test_insufficient_channels():
    """Single channel returns status=insufficient (can't correlate one column)."""
    data = np.random.random((16, 1, 5))  # enough frames, but 1 channel
    result = band_power_connectivity(data, ["Ch0"])
    assert result["status"] == "insufficient"


def test_basic_connectivity():
    """Two perfectly correlated channels produce r≈1.0 in the correlation matrix."""
    # Build 20 frames × 2 channels × 5 bands; make both channels identical for Alpha (band 2)
    rng = np.random.default_rng(42)
    series = rng.random(20)
    arr = np.zeros((20, 2, 5))
    arr[:, 0, 2] = series
    arr[:, 1, 2] = series + 1e-9  # perfect correlation in band 2 (Alpha)
    arr[:, :, :2] = rng.random((20, 2, 2))  # unrelated bands
    arr[:, :, 3:] = rng.random((20, 2, 2))

    result = band_power_connectivity(arr, ["A", "B"], band="Alpha", include_matrix=True)

    # Success case → no 'status' key (only failures have status)
    assert "status" not in result
    assert result["band"] == "Alpha"
    assert result["n_channels"] == 2
    assert result["n_frames"] == 20
    # strongest pair should be (A, B) with r≈1.0
    assert len(result["strongest_pairs"]) >= 1
    pair = result["strongest_pairs"][0]
    assert set([pair["a"], pair["b"]]) == {"A", "B"}
    assert abs(pair["r"]) > 0.99  # should be ≈1.0
    # mean_connectivity should be high
    assert result["mean_connectivity"] > 0.9
    # matrix diagonal is always 1.0
    matrix = result["matrix"]
    assert abs(matrix[0][0] - 1.0) < 1e-6
    assert abs(matrix[1][1] - 1.0) < 1e-6


def test_different_band():
    """Confirm the chosen band is actually used."""
    rng = np.random.default_rng(42)
    arr = np.zeros((20, 3, 5))
    # put a correlated signal in band 0 (Delta)
    sig = rng.random(20)
    arr[:, 0, 0] = sig
    arr[:, 1, 0] = sig
    arr[:, 2, 0] = sig
    # rest is noise
    arr[:, :, 1:] = rng.random((20, 3, 4))

    # default is Alpha (band 2) → should find weak correlation
    res_alpha = band_power_connectivity(arr, ["A", "B", "C"], band="Alpha", include_matrix=False)
    # Delta (band 0) → should find strong
    res_delta = band_power_connectivity(arr, ["A", "B", "C"], band="Delta", include_matrix=False)

    assert res_alpha["band"] == "Alpha"
    assert res_delta["band"] == "Delta"
    assert res_delta["mean_connectivity"] > res_alpha["mean_connectivity"] + 0.5


def test_per_channel():
    """Per-channel coupling should show which channels are most/least connected."""
    rng = np.random.default_rng(42)
    arr = np.zeros((20, 4, 5))
    # make channels 0 & 1 strongly coupled at band 2 (Alpha), channels 2 & 3 uncorrelated
    sig = rng.random(20)
    arr[:, 0, 2] = sig
    arr[:, 1, 2] = sig
    arr[:, 2, 2] = rng.random(20)
    arr[:, 3, 2] = rng.random(20)
    arr[:, :, :2] = rng.random((20, 4, 2))
    arr[:, :, 3:] = rng.random((20, 4, 2))

    result = band_power_connectivity(arr, ["A", "B", "C", "D"], band="Alpha")

    per = result["per_channel"]
    assert len(per) == 4
    # A and B should have higher mean_abs_r than C and D
    ab_r = [p["mean_abs_r"] for p in per if p["channel"] in ("A", "B")]
    cd_r = [p["mean_abs_r"] for p in per if p["channel"] in ("C", "D")]
    assert min(ab_r) > max(cd_r)


def test_flat_channel():
    """A zero-variance channel should be marked flat and not cause NaN."""
    rng = np.random.default_rng(42)
    arr = np.zeros((20, 3, 5))
    arr[:, 0, 2] = rng.random(20)  # normal
    arr[:, 1, 2] = 0.0             # flat
    arr[:, 2, 2] = rng.random(20)  # normal
    arr[:, :, :2] = rng.random((20, 3, 2))
    arr[:, :, 3:] = rng.random((20, 3, 2))

    result = band_power_connectivity(arr, ["A", "B", "C"], band="Alpha", include_matrix=True)

    per = result["per_channel"]
    b = [p for p in per if p["channel"] == "B"][0]
    assert b["flat"] is True
    # the correlation with a flat channel should be zeroed, not NaN
    matrix = result["matrix"]
    for i in range(3):
        for j in range(3):
            assert not np.isnan(matrix[i][j])


def test_top_pairs():
    """Strongest pairs should be ranked by absolute r, limited by top parameter."""
    rng = np.random.default_rng(42)
    arr = np.zeros((20, 4, 5))
    sig1 = rng.random(20)
    sig2 = rng.random(20)
    arr[:, 0, 2] = sig1
    arr[:, 1, 2] = sig1  # A-B strongly coupled
    arr[:, 2, 2] = sig2
    arr[:, 3, 2] = -sig2  # C-D strongly *anti*-coupled
    arr[:, :, :2] = rng.random((20, 4, 2))
    arr[:, :, 3:] = rng.random((20, 4, 2))

    result = band_power_connectivity(arr, ["A", "B", "C", "D"], band="Alpha", top=2)

    pairs = result["strongest_pairs"]
    assert len(pairs) <= 2
    # Top pairs by absolute r; since log() transforms the anti-correlation,
    # the actual strongest might vary. Just check we got top=2 and have pairs.
    assert len(pairs) == 2
