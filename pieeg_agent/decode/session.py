"""Sessions — record a window of brain, summarise it, compare two.

This is the agent's lab notebook. A :class:`SessionRecorder` accumulates the
band-power frames the cascade already emits over a labelled window (say 20 s of
"eyes-closed rest"), then condenses them into a small, honest summary: mean and
spread of each band, the focus/relax/engagement indices, signal quality,
artifact counts and a connectivity snapshot. :func:`compare_summaries` then
contrasts two such summaries with a within-session Cohen's d, so the agent can
answer "what changed between my rest and my focus block?" with an effect size
instead of a vibe.

Everything is numpy + plain dicts and persists through
:class:`~pieeg_agent.decode.store.SessionStore`, mirroring how patterns are
kept — small JSON files the user can inspect and share.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..perceive.features import BAND_NAMES
from .connectivity import band_power_connectivity

_EPS = 1e-12


def _r(value, nd: int = 4) -> float:
    """Round to ``nd`` places, mapping NaN/invalid to 0.0 for clean JSON."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(f):
        return 0.0
    return round(f, nd)


class SessionRecorder:
    """Accumulates feature frames for one labelled window, then summarises."""

    def __init__(self):
        self._open = False
        self._label = ""
        self._t0 = 0.0
        self._rel: list[list[float]] = []
        self._per_ch: list[np.ndarray] = []
        self._idx: list[tuple[float, float, float, float]] = []
        self._ts: list[float] = []
        self._labels: list[str] = []

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def label(self) -> str:
        return self._label

    @property
    def started_at(self) -> float:
        return self._t0

    @property
    def n_frames(self) -> int:
        return len(self._per_ch)

    def open(self, label: str, *, t0: float | None = None, channel_labels=None) -> None:
        """Arm recording. Subsequent :meth:`add` calls accumulate until close."""
        self._open = True
        self._label = label
        self._t0 = float(t0 if t0 is not None else time.time())
        self._rel.clear()
        self._per_ch.clear()
        self._idx.clear()
        self._ts.clear()
        self._labels = list(channel_labels or [])

    def add(self, bp, quality=None, state=None) -> None:
        """Append one feature frame (band powers + optional quality/state)."""
        if not self._open:
            return
        rel = bp.relative()
        self._rel.append([float(rel.get(b, 0.0)) for b in BAND_NAMES])
        self._per_ch.append(np.asarray(bp.per_channel, dtype=np.float64).copy())
        q = float(getattr(quality, "overall", np.nan)) if quality is not None else np.nan
        if state is not None:
            self._idx.append(
                (
                    float(getattr(state, "focus", np.nan)),
                    float(getattr(state, "relax", np.nan)),
                    float(getattr(state, "engagement", np.nan)),
                    q,
                )
            )
        else:
            self._idx.append((np.nan, np.nan, np.nan, q))
        self._ts.append(float(getattr(bp, "timestamp", time.time())))

    def close(self) -> "SessionRecording":
        """Stop recording and return the immutable recording for summarising."""
        self._open = False
        nb = len(BAND_NAMES)
        return SessionRecording(
            label=self._label,
            started_at=self._t0,
            rel=np.asarray(self._rel, dtype=np.float64) if self._rel else np.zeros((0, nb)),
            per_ch=np.asarray(self._per_ch, dtype=np.float64)
            if self._per_ch
            else np.zeros((0, 0, nb)),
            idx=np.asarray(self._idx, dtype=np.float64) if self._idx else np.zeros((0, 4)),
            ts=np.asarray(self._ts, dtype=np.float64) if self._ts else np.zeros((0,)),
            channel_labels=list(self._labels),
        )


@dataclass
class SessionRecording:
    """An immutable captured window, ready to summarise."""

    label: str
    started_at: float
    rel: np.ndarray  # [T, n_bands] relative band powers
    per_ch: np.ndarray  # [T, n_ch, n_bands] per-channel band powers
    idx: np.ndarray  # [T, 4] focus, relax, engagement, quality
    ts: np.ndarray  # [T] timestamps
    channel_labels: list = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.rel.shape[0])

    @property
    def duration_s(self) -> float:
        if self.ts.size >= 2:
            return float(self.ts[-1] - self.ts[0])
        return 0.0

    def _idx_stat(self, k: int) -> dict | None:
        if not self.n_frames:
            return None
        col = self.idx[:, k]
        col = col[~np.isnan(col)]
        if col.size == 0:
            return None
        return {"mean": _r(col.mean()), "std": _r(col.std()), "n": int(col.size)}

    def summary(self) -> dict:
        """Condense the window into a small, honest, JSON-friendly report."""
        n = self.n_frames
        bands: dict[str, dict] = {}
        for j, b in enumerate(BAND_NAMES):
            col = self.rel[:, j] if n else np.zeros(0)
            bands[b] = {
                "mean": _r(np.nanmean(col)) if n else 0.0,
                "std": _r(np.nanstd(col)) if n else 0.0,
            }
        dominant = max(bands, key=lambda b: bands[b]["mean"]) if n else None

        indices = {
            "focus": self._idx_stat(0),
            "relax": self._idx_stat(1),
            "engagement": self._idx_stat(2),
        }
        quality = self._idx_stat(3)

        if n and self.per_ch.size:
            per_ch_mean = self.per_ch.mean(axis=0)
        else:
            per_ch_mean = np.zeros((0, len(BAND_NAMES)))
        n_ch = per_ch_mean.shape[0]
        labels = self.channel_labels or [f"Ch{i}" for i in range(n_ch)]
        per_channel = {
            "labels": labels[:n_ch],
            "mean": [[_r(v) for v in row] for row in per_ch_mean],
        }

        if n >= 8 and self.per_ch.size:
            conn = band_power_connectivity(
                self.per_ch, labels, band="Alpha", include_matrix=False
            )
            connectivity = {
                "band": conn.get("band"),
                "mean_connectivity": conn.get("mean_connectivity"),
                "strongest_pairs": conn.get("strongest_pairs", [])[:3],
            } if "mean_connectivity" in conn else conn
        else:
            connectivity = {"status": "insufficient"}

        return {
            "label": self.label,
            "started_at": self.started_at,
            "duration_s": _r(self.duration_s),
            "n_frames": n,
            "bands": bands,
            "dominant_band": dominant,
            "indices": indices,
            "signal_quality": quality,
            "per_channel_bands": per_channel,
            "connectivity": connectivity,
            "band_names": list(BAND_NAMES),
        }


def _cohens_d(ma: float, sa: float, na: int, mb: float, sb: float, nb: int) -> float:
    """Pooled-SD Cohen's d for B vs A from summary statistics."""
    if na >= 2 and nb >= 2:
        sp = np.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / max(na + nb - 2, 1))
    else:
        sp = (abs(sa) + abs(sb)) / 2.0
    sp = sp or _EPS
    return (mb - ma) / sp


def _direction(delta: float) -> str:
    if delta > 0:
        return "higher in B"
    if delta < 0:
        return "lower in B"
    return "unchanged"


def _headline(top: dict | None) -> str:
    if top is None:
        return "No comparable features between the two sessions."
    return (
        f"Biggest change: {top['feature']} {_direction(top['delta'])} "
        f"(d={top['cohens_d']:+.2f})."
    )


def compare_summaries(a: dict, b: dict) -> dict:
    """Contrast two session summaries with within-session Cohen's d, ranked."""
    na = int(a.get("n_frames", 0))
    nb = int(b.get("n_frames", 0))
    feats: list[dict] = []

    band_names = a.get("band_names", list(BAND_NAMES))
    a_bands = a.get("bands", {})
    b_bands = b.get("bands", {})
    for band in band_names:
        ba, bb = a_bands.get(band), b_bands.get(band)
        if not ba or not bb:
            continue
        d = _cohens_d(ba["mean"], ba["std"], na, bb["mean"], bb["std"], nb)
        feats.append(
            {
                "feature": f"{band} power",
                "a": ba["mean"],
                "b": bb["mean"],
                "delta": _r(bb["mean"] - ba["mean"]),
                "cohens_d": _r(d, 3),
            }
        )

    a_idx, b_idx = a.get("indices") or {}, b.get("indices") or {}
    for key in ("focus", "relax", "engagement"):
        ia, ib = a_idx.get(key), b_idx.get(key)
        if not ia or not ib:
            continue
        d = _cohens_d(ia["mean"], ia["std"], ia["n"], ib["mean"], ib["std"], ib["n"])
        feats.append(
            {
                "feature": key,
                "a": ia["mean"],
                "b": ib["mean"],
                "delta": _r(ib["mean"] - ia["mean"]),
                "cohens_d": _r(d, 3),
            }
        )

    qa, qb = a.get("signal_quality"), b.get("signal_quality")
    if qa and qb:
        d = _cohens_d(qa["mean"], qa["std"], qa["n"], qb["mean"], qb["std"], qb["n"])
        feats.append(
            {
                "feature": "signal quality",
                "a": qa["mean"],
                "b": qb["mean"],
                "delta": _r(qb["mean"] - qa["mean"]),
                "cohens_d": _r(d, 3),
            }
        )

    feats.sort(key=lambda f: -abs(f["cohens_d"]))
    return {
        "a": a.get("label"),
        "b": b.get("label"),
        "n_a": na,
        "n_b": nb,
        "differences": feats,
        "headline": _headline(feats[0] if feats else None),
        "caveat": "Within-session Cohen's d (descriptive effect size), not a "
        "generalisation or clinical claim; sessions differ in electrode drift "
        "and time.",
    }
