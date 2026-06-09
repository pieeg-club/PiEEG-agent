"""Contrastive calibration — learn a pattern from *rest vs active* examples.

This is the PiEEG **Avatar Foundation** recipe. The user records two states —
a baseline ("rest") and the thing they want the agent to recognise ("active",
e.g. eyes-closed, mental arithmetic, a clenched jaw) — and we measure, feature
by feature, how far apart the two states sit. The separation is reported as
**Cohen's d**, so the agent can say *why* a pattern is (or is not) learnable and
*which* channels carry it, instead of returning an opaque score.

Two deliberate honesty choices:

* Statistics are accumulated with **Welford's** online algorithm, so a long
  recording never holds every sample in memory and never loses precision to
  catastrophic cancellation.
* Cohen's d here is a **within-session, descriptive** effect size: it says the
  two recordings differ, not that the detector will generalise to another day
  or another montage. The cross-validated balanced accuracy from
  :mod:`classifier` is the number to trust for that.

The calibrator also retains the labelled frames (tagged by *rep*) so the
classifier can split **leave-one-rep-out** and avoid the temporal leakage that
makes naive sample splits look better than they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureLayout

_EPS = 1e-12

REST = "rest"
ACTIVE = "active"


class Welford:
    """Vectorised online mean / variance (Welford) over fixed-length vectors."""

    def __init__(self, dim: int):
        self._n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def update_batch(self, batch: np.ndarray) -> None:
        for row in np.asarray(batch, dtype=np.float64):
            self.update(row)

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """Sample variance (ddof=1); zeros until at least two observations."""
        if self._n < 2:
            return np.zeros_like(self._m2)
        return self._m2 / (self._n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


def cohens_d(
    mean_a: np.ndarray, var_a: np.ndarray, n_a: int,
    mean_b: np.ndarray, var_b: np.ndarray, n_b: int,
) -> np.ndarray:
    """Signed Cohen's d (b − a) per feature with pooled standard deviation."""
    if n_a < 2 or n_b < 2:
        return np.zeros_like(mean_a)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    return (mean_b - mean_a) / (np.sqrt(pooled) + _EPS)


@dataclass
class FeatureRanking:
    """Per-feature rest→active separation, ready to explain to the user."""

    layout: FeatureLayout
    d: np.ndarray            # (dim,) signed Cohen's d, active − rest
    rest_mean: np.ndarray
    rest_std: np.ndarray
    active_mean: np.ndarray
    active_std: np.ndarray
    n_rest: int
    n_active: int

    def top(self, k: int = 8) -> list[dict]:
        """The ``k`` most separating features, strongest first."""
        order = np.argsort(-np.abs(self.d))[:k]
        names = self.layout.names
        return [
            {
                "feature": names[i],
                "channel": self.layout.channel_labels[self.layout.channel_of(i)],
                "cohens_d": round(float(self.d[i]), 3),
                "direction": "up" if self.d[i] > 0 else "down",
            }
            for i in order
        ]

    def channel_importance(self) -> list[dict]:
        """Per-channel separation = RMS of that channel's feature d's."""
        groups = self.layout.channel_groups()
        labels = self.layout.channel_labels
        out = []
        for ch, idx in enumerate(groups):
            block = self.d[list(idx)]
            out.append(
                {
                    "channel": labels[ch],
                    "strength": round(float(np.sqrt(np.mean(block**2))), 3),
                }
            )
        out.sort(key=lambda r: -r["strength"])
        return out

    def to_dict(self) -> dict:
        return {
            "n_rest": self.n_rest,
            "n_active": self.n_active,
            "top_features": self.top(8),
            "channel_importance": self.channel_importance(),
            "caveat": (
                "Cohen's d is a within-session, descriptive effect size "
                "(this recording only) — not a generalisation guarantee."
            ),
        }


class ContrastiveCalibrator:
    """Collects rest/active frames and reports their feature separation."""

    def __init__(self, layout: FeatureLayout):
        self.layout = layout
        self._rest = Welford(layout.dim)
        self._active = Welford(layout.dim)
        # Retained labelled frames (for the classifier's leave-one-rep-out CV).
        self._x: list[np.ndarray] = []
        self._y: list[int] = []
        self._groups: list[int] = []

    def add(self, label: str, features: np.ndarray, *, rep: int = 0) -> None:
        feat = np.asarray(features, dtype=np.float64)
        if feat.shape != (self.layout.dim,):
            raise ValueError(f"expected feature dim {self.layout.dim}, got {feat.shape}")
        if label == REST:
            self._rest.update(feat)
            self._y.append(0)
        elif label == ACTIVE:
            self._active.update(feat)
            self._y.append(1)
        else:
            raise ValueError(f"label must be {REST!r} or {ACTIVE!r}, got {label!r}")
        self._x.append(feat)
        self._groups.append(int(rep))

    def add_batch(self, label: str, batch: np.ndarray, *, rep: int = 0) -> None:
        for row in np.asarray(batch, dtype=np.float64):
            self.add(label, row, rep=rep)

    @property
    def n_rest(self) -> int:
        return self._rest.count

    @property
    def n_active(self) -> int:
        return self._active.count

    @property
    def reps(self) -> list[int]:
        return sorted(set(self._groups))

    def ranking(self) -> FeatureRanking:
        return FeatureRanking(
            layout=self.layout,
            d=cohens_d(
                self._rest.mean, self._rest.var, self._rest.count,
                self._active.mean, self._active.var, self._active.count,
            ),
            rest_mean=self._rest.mean,
            rest_std=self._rest.std,
            active_mean=self._active.mean,
            active_std=self._active.std,
            n_rest=self._rest.count,
            n_active=self._active.count,
        )

    def standardizer(self) -> tuple[np.ndarray, np.ndarray]:
        """Centre/scale derived from *both* states pooled (mean, std)."""
        x = np.asarray(self._x, dtype=np.float64)
        if x.size == 0:
            dim = self.layout.dim
            return np.zeros(dim), np.ones(dim)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < _EPS] = 1.0
        return mean, std

    def dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The labelled set: ``(X (n, dim), y (n,), groups (n,))``."""
        if not self._x:
            dim = self.layout.dim
            return np.empty((0, dim)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)
        return (
            np.asarray(self._x, dtype=np.float64),
            np.asarray(self._y, dtype=int),
            np.asarray(self._groups, dtype=int),
        )
