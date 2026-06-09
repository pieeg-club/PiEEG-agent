"""Advanced spectral analysis — the lab features the agent can pull on demand.

Where :mod:`pieeg_agent.perceive.features` produces the five canonical band
powers every tick, this module answers the deeper, *on-demand* questions a
researcher actually asks of a spectrum — computed straight from the PSD the
cascade already has, so the LLM can request them without any extra DSP thread:

* **Individual alpha frequency (IAF)** — the precise alpha-peak location, which
  varies person to person and tracks arousal far better than a fixed 10 Hz.
* **Aperiodic (1/f) slope** — the broadband exponent of the spectrum, a robust
  index of excitation/inhibition balance (a FOOOF-lite log-log fit).
* **Theta/beta ratio** — a classic attention marker.
* **Spectral entropy** — how flat vs peaked the spectrum is.
* **Frontal alpha asymmetry** — left/right alpha imbalance, when the montage
  labels identify hemispheres (10-20 odd = left, even = right).

Everything is descriptive and within-session; the functions take a
:class:`~pieeg_agent.perceive.features.BandPowers` (or its raw PSD) and return
plain JSON-friendly dicts.
"""

from __future__ import annotations

import re

import numpy as np

from ..perceive.features import BAND_NAMES, BandPowers

_EPS = 1e-12
_LABEL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def individual_alpha_peak(
    psd: np.ndarray, freqs: np.ndarray, lo: float = 7.0, hi: float = 13.0
) -> dict | None:
    """Locate the alpha peak (IAF) in ``[lo, hi)`` Hz with parabolic refinement.

    Returns ``{"peak_hz", "peak_power", "band_hz"}`` or ``None`` if the band is
    empty or shows no interior maximum (a flat or edge-rising spectrum).
    """
    psd = np.asarray(psd, dtype=np.float64).ravel()
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 3:
        return None
    band_psd = psd[mask]
    band_f = freqs[mask]
    k = int(band_psd.argmax())
    # Reject a peak pinned to either edge — that is a slope, not a peak.
    if k == 0 or k == len(band_psd) - 1:
        return None
    # Parabolic interpolation across the three points around the max for a
    # sub-bin frequency estimate.
    y0, y1, y2 = band_psd[k - 1], band_psd[k], band_psd[k + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > _EPS else 0.0
    df = band_f[1] - band_f[0]
    peak_hz = float(band_f[k] + delta * df)
    return {
        "peak_hz": round(peak_hz, 2),
        "peak_power": round(float(y1), 4),
        "band_hz": [lo, hi],
    }


def aperiodic_fit(
    psd: np.ndarray, freqs: np.ndarray, lo: float = 2.0, hi: float = 40.0
) -> dict | None:
    """Fit the 1/f aperiodic component: ``log10(psd) = offset - exponent*log10(f)``.

    Returns ``{"exponent", "offset", "r2"}``. A larger exponent means a steeper
    (more "inhibited") spectrum. ``None`` if too few usable bins.
    """
    psd = np.asarray(psd, dtype=np.float64).ravel()
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mask = (freqs >= lo) & (freqs <= hi) & (freqs > 0) & (psd > 0)
    if mask.sum() < 4:
        return None
    x = np.log10(freqs[mask])
    y = np.log10(psd[mask])
    slope, offset = np.polyfit(x, y, 1)
    pred = slope * x + offset
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or _EPS
    return {
        "exponent": round(float(-slope), 3),
        "offset": round(float(offset), 3),
        "r2": round(1.0 - ss_res / ss_tot, 3),
        "fit_hz": [lo, hi],
    }


def spectral_entropy(
    psd: np.ndarray, freqs: np.ndarray, lo: float = 1.0, hi: float = 45.0
) -> float:
    """Normalised Shannon entropy of the PSD over ``[lo, hi]`` (0 peaked … 1 flat)."""
    psd = np.asarray(psd, dtype=np.float64).ravel()
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mask = (freqs >= lo) & (freqs <= hi)
    p = psd[mask]
    total = p.sum()
    if p.size < 2 or total <= 0:
        return 0.0
    p = p / total
    h = -np.sum(p * np.log(p + _EPS))
    return float(h / np.log(p.size))


def theta_beta_ratio(bands: dict[str, float]) -> float:
    """Theta / Beta power ratio — a classic sustained-attention marker."""
    return float(bands.get("Theta", 0.0) / (bands.get("Beta", 0.0) + _EPS))


def _parse_label(label: str) -> tuple[str, int] | None:
    m = _LABEL_RE.match(label.strip())
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def frontal_alpha_asymmetry(
    per_channel: np.ndarray, labels: list[str]
) -> dict | None:
    """Frontal alpha asymmetry ln(R) − ln(L) over matched 10-20 electrode pairs.

    Uses the 10-20 convention (odd index = left hemisphere, even = right) to
    pair electrodes sharing a letter prefix (e.g. F3/F4, AF7/AF8). Positive
    values indicate relatively more right-hemisphere alpha (less right cortical
    activity). Returns ``None`` if no hemispheric pairs are present.
    """
    alpha_idx = BAND_NAMES.index("Alpha")
    by_key: dict[tuple[str, bool], list[float]] = {}
    for i, lab in enumerate(labels):
        parsed = _parse_label(lab)
        if parsed is None or i >= per_channel.shape[0]:
            continue
        prefix, num = parsed
        is_left = num % 2 == 1
        by_key.setdefault((prefix, is_left), []).append(
            float(per_channel[i, alpha_idx])
        )

    pairs: list[dict] = []
    prefixes = {k[0] for k in by_key}
    for prefix in sorted(prefixes):
        left = by_key.get((prefix, True))
        right = by_key.get((prefix, False))
        if not left or not right:
            continue
        la = float(np.mean(left))
        ra = float(np.mean(right))
        pairs.append(
            {
                "pair": f"{prefix}(L/R)",
                "asymmetry": round(float(np.log(ra + _EPS) - np.log(la + _EPS)), 3),
            }
        )
    if not pairs:
        return None
    mean_asym = float(np.mean([p["asymmetry"] for p in pairs]))
    return {"mean": round(mean_asym, 3), "pairs": pairs}


def analyze_spectrum(
    bp: BandPowers, labels: list[str] | None = None, *, channel: int | None = None
) -> dict:
    """Assemble a full spectral report for the latest feature frame.

    With ``channel=None`` the PSD is averaged across channels; pass a channel
    index for a single electrode. Frontal alpha asymmetry is only reported in
    the channel-averaged, multi-channel case (it needs the montage).
    """
    labels = labels or []
    if channel is not None:
        if not (0 <= channel < bp.psd.shape[0]):
            return {"error": f"channel {channel} out of range (0..{bp.psd.shape[0] - 1})"}
        psd = bp.psd[channel]
        bands = {b: float(bp.per_channel[channel, j]) for j, b in enumerate(BAND_NAMES)}
        scope = labels[channel] if channel < len(labels) else f"Ch{channel}"
    else:
        psd = bp.psd.mean(axis=0)
        bands = dict(bp.bands)
        scope = "channel-average"

    total = sum(bands.values()) or _EPS
    rel = {b: round(v / total, 4) for b, v in bands.items()}
    report: dict = {
        "scope": scope,
        "timestamp": bp.timestamp,
        "dominant_band": max(bands, key=bands.__getitem__),
        "relative_bands": rel,
        "alpha_peak": individual_alpha_peak(psd, bp.freqs),
        "aperiodic": aperiodic_fit(psd, bp.freqs),
        "theta_beta_ratio": round(theta_beta_ratio(bands), 3),
        "spectral_entropy": round(spectral_entropy(psd, bp.freqs), 3),
        "n_channels": bp.n_channels,
    }
    if channel is None and bp.n_channels > 1 and labels:
        report["frontal_alpha_asymmetry"] = frontal_alpha_asymmetry(
            bp.per_channel, labels
        )
    return report
