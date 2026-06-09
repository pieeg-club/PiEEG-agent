"""The trainable detector — a regularised linear read of the feature vector.

At this data scale (hundreds of frames, ~tens of features) a regularised
**linear** model beats anything deeper: it trains in milliseconds, needs no
accelerator, and — crucially for this project — it is *legible*. The PiEEG
**Face Trainer** experience is the template:

* **Logistic regression** for a calibrated 0..1 probability.
* An **L2 (ridge)** penalty to keep the fit well-conditioned, plus a
  **group-lasso** penalty over per-channel blocks. The group term drives an
  entire channel's weights to zero when it does not help, so the surviving
  weights *are* a channel-importance map — the agent can say "I'm reading this
  mostly from O1/O2".
* **Leave-one-rep-out** cross-validation reporting **balanced accuracy**, so a
  class imbalance (lots of rest, little active) cannot inflate the score and
  temporally-correlated frames in one rep cannot leak into their own test.

Optimisation is proximal gradient (ISTA): an ordinary gradient step on the
smooth logistic + ridge part, followed by block soft-thresholding for the
group-lasso part. Pure NumPy, no SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureLayout

_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of sensitivity and specificity — immune to class imbalance."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    pos = y_true == 1
    neg = ~pos
    sens = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
    spec = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
    if not pos.any() or not neg.any():
        return float("nan")
    return 0.5 * (sens + spec)


@dataclass
class CVResult:
    """Leave-one-rep-out cross-validation summary."""

    balanced_accuracy: float | None
    sensitivity: float | None
    specificity: float | None
    n_folds: int
    n_samples: int

    def to_dict(self) -> dict:
        def _r(v):
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, 3)

        return {
            "balanced_accuracy": _r(self.balanced_accuracy),
            "sensitivity": _r(self.sensitivity),
            "specificity": _r(self.specificity),
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
        }


class PatternClassifier:
    """L2 + group-lasso logistic detector over the per-channel feature blocks."""

    def __init__(
        self,
        layout: FeatureLayout,
        *,
        l2: float = 1e-2,
        group_lasso: float = 5e-3,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.layout = layout
        self.l2 = float(l2)
        self.group_lasso = float(group_lasso)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.w = np.zeros(layout.dim, dtype=np.float64)
        self.b = 0.0
        self.mean = np.zeros(layout.dim, dtype=np.float64)
        self.std = np.ones(layout.dim, dtype=np.float64)
        self.fitted = False

    # ── training ────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> "PatternClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.layout.dim:
            raise ValueError(f"X must be (n, {self.layout.dim})")
        if mean is None:
            mean = X.mean(axis=0)
        if std is None:
            std = X.std(axis=0)
        std = np.asarray(std, dtype=np.float64).copy()
        std[std < _EPS] = 1.0
        self.mean, self.std = np.asarray(mean, dtype=np.float64), std

        xs = (X - self.mean) / self.std
        self.w, self.b = self._solve(xs, y)
        self.fitted = True
        return self

    def _solve(self, xs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        n = xs.shape[0]
        groups = [list(g) for g in self.layout.channel_groups()]
        # Step size from the smooth part's Lipschitz constant.
        smax = float(np.linalg.norm(xs, 2)) if n else 1.0
        lip = 0.25 * smax * smax / max(n, 1) + self.l2
        lr = 1.0 / max(lip, _EPS)

        w = np.zeros(xs.shape[1], dtype=np.float64)
        b = 0.0
        for _ in range(self.max_iter):
            p = _sigmoid(xs @ w + b)
            resid = p - y
            grad_w = xs.T @ resid / n + self.l2 * w
            grad_b = float(resid.mean())
            w_new = w - lr * grad_w
            b_new = b - lr * grad_b
            # Group-lasso proximal step (block soft-threshold), bias untouched.
            if self.group_lasso > 0:
                thresh = lr * self.group_lasso
                for g in groups:
                    block = w_new[g]
                    norm = float(np.linalg.norm(block))
                    if norm <= thresh:
                        w_new[g] = 0.0
                    else:
                        w_new[g] = block * (1.0 - thresh / norm)
            if np.linalg.norm(w_new - w) < self.tol and abs(b_new - b) < self.tol:
                w, b = w_new, b_new
                break
            w, b = w_new, b_new
        return w, b

    # ── inference ───────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        xs = (X - self.mean) / self.std
        return _sigmoid(xs @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score_one(self, features: np.ndarray) -> float:
        """Probability for a single feature vector (the live-scoring hot path)."""
        return float(self.predict_proba(np.asarray(features)[None, :])[0])

    # ── explainability ──────────────────────────────────────────────────
    def channel_importance(self) -> list[dict]:
        """Per-channel weight magnitude, normalised so the strongest is 1.0."""
        groups = self.layout.channel_groups()
        labels = self.layout.channel_labels
        norms = np.array([np.linalg.norm(self.w[list(g)]) for g in groups])
        top = float(norms.max()) if norms.size and norms.max() > 0 else 1.0
        out = [
            {"channel": labels[c], "importance": round(float(norms[c] / top), 3)}
            for c in range(len(groups))
        ]
        out.sort(key=lambda r: -r["importance"])
        return out

    # ── cross-validation ────────────────────────────────────────────────
    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> CVResult:
        """Leave-one-rep-out CV; balanced accuracy over pooled held-out frames."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups, dtype=int)
        unique = np.unique(groups)
        if unique.size < 2:
            return CVResult(None, None, None, n_folds=int(unique.size), n_samples=int(y.size))

        oof_true: list[int] = []
        oof_pred: list[int] = []
        for held in unique:
            test = groups == held
            train = ~test
            if y[train].min() == y[train].max():
                continue  # a fold whose training data is single-class is useless
            clf = PatternClassifier(
                self.layout, l2=self.l2, group_lasso=self.group_lasso,
                max_iter=self.max_iter, tol=self.tol,
            ).fit(X[train], y[train])
            oof_true.extend(y[test].tolist())
            oof_pred.extend(clf.predict(X[test]).tolist())

        if not oof_true:
            return CVResult(None, None, None, n_folds=int(unique.size), n_samples=int(y.size))
        yt = np.asarray(oof_true)
        yp = np.asarray(oof_pred)
        pos, neg = yt == 1, yt == 0
        sens = float((yp[pos] == 1).mean()) if pos.any() else None
        spec = float((yp[neg] == 0).mean()) if neg.any() else None
        bacc = balanced_accuracy(yt, yp)
        return CVResult(
            balanced_accuracy=None if np.isnan(bacc) else bacc,
            sensitivity=sens,
            specificity=spec,
            n_folds=int(unique.size),
            n_samples=int(y.size),
        )

    # ── persistence ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "channel_labels": list(self.layout.channel_labels),
            "per_channel_features": list(self.layout.per_channel),
            "l2": self.l2,
            "group_lasso": self.group_lasso,
            "weights": self.w.tolist(),
            "bias": self.b,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatternClassifier":
        layout = FeatureLayout(
            channel_labels=tuple(data["channel_labels"]),
            per_channel=tuple(data["per_channel_features"]),
        )
        clf = cls(layout, l2=data.get("l2", 1e-2), group_lasso=data.get("group_lasso", 5e-3))
        clf.w = np.asarray(data["weights"], dtype=np.float64)
        clf.b = float(data["bias"])
        clf.mean = np.asarray(data["mean"], dtype=np.float64)
        clf.std = np.asarray(data["std"], dtype=np.float64)
        clf.fitted = bool(data.get("fitted", True))
        return clf
