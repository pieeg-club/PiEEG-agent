"""Functional connectivity — how the channels move *together*.

Where :mod:`spectral` characterises one spectrum and :mod:`patterns` learns a
labelled state, this module asks a different lab question: which electrodes are
coupled? It does so from the band-power frames the cascade already produces,
correlating each channel's *log band-power envelope* over a short window
(amplitude coupling). That keeps it pure-numpy and honest:

* it is **amplitude** coupling, not phase coherence — no cross-spectra, so a
  single window can't fake a connection (correlation needs variation over time);
* it is **within-session** and descriptive — electrode placement, drift and
  movement all shape it, so it ranks coupling *now*, it does not estimate a
  person's "true" network.

The result is a small JSON object (a correlation matrix plus the strongest
pairs and a per-channel coupling score) ready for the agent or a web heatmap.
"""

from __future__ import annotations

import numpy as np

from ..perceive.features import BAND_NAMES

_EPS = 1e-12

# Connectivity needs the band-power envelope to actually vary across the window;
# below this many frames the correlation is meaningless.
_MIN_FRAMES = 8


def band_power_connectivity(
    history,
    labels,
    *,
    band: str = "Alpha",
    top: int = 5,
    include_matrix: bool = True,
) -> dict:
    """Per-channel amplitude-coupling connectivity for one band.

    ``history`` is an array-like ``[T, n_channels, n_bands]`` of per-channel
    band powers over time (the cascade's ``BandPowers.per_channel`` stacked).
    Returns the Pearson-r matrix between channels' log-power time series for
    ``band``, the strongest pairs, and each channel's mean absolute coupling.
    """
    arr = np.asarray(history, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < _MIN_FRAMES or arr.shape[1] < 2:
        return {
            "status": "insufficient",
            "detail": "Need a few seconds of signal across 2+ channels.",
            "n_frames": int(arr.shape[0]) if arr.ndim >= 1 else 0,
        }

    band = band.capitalize()
    if band not in BAND_NAMES:
        band = "Alpha"
    bidx = BAND_NAMES.index(band)

    # Log stabilises the heavy-tailed power distribution so the correlation
    # reflects relative co-fluctuation rather than a few large spikes.
    series = np.log(arr[:, :, bidx] + _EPS)  # [T, n_ch]
    n_ch = series.shape[1]
    std = series.std(axis=0)

    corr = np.corrcoef(series, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)  # flat channel → 0 coupling, not NaN
    np.fill_diagonal(corr, 1.0)

    def lab(i: int) -> str:
        return labels[i] if i < len(labels) else f"Ch{i}"

    pairs = [
        (i, j, float(corr[i, j]))
        for i in range(n_ch)
        for j in range(i + 1, n_ch)
    ]
    pairs.sort(key=lambda t: -abs(t[2]))
    strongest = [{"a": lab(i), "b": lab(j), "r": round(r, 3)} for i, j, r in pairs[:top]]

    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    per_abs = np.nanmean(np.abs(off), axis=1)
    per_channel = [
        {
            "channel": lab(i),
            "mean_abs_r": round(float(per_abs[i]), 3),
            "flat": bool(std[i] <= _EPS),
        }
        for i in range(n_ch)
    ]
    order = np.argsort(-per_abs)

    out: dict = {
        "band": band,
        "n_channels": n_ch,
        "n_frames": int(arr.shape[0]),
        "mean_connectivity": round(float(np.nanmean(np.abs(off))), 3),
        "strongest_pairs": strongest,
        "per_channel": per_channel,
        "most_connected": lab(int(order[0])),
        "least_connected": lab(int(order[-1])),
        "method": "log band-power amplitude correlation (Pearson r, within-session)",
        "caveat": "Amplitude coupling, not phase coherence; descriptive and "
        "within-session only.",
    }
    if include_matrix:
        out["labels"] = [lab(i) for i in range(n_ch)]
        out["matrix"] = [
            [round(float(corr[i, j]), 3) for j in range(n_ch)] for i in range(n_ch)
        ]
    return out
