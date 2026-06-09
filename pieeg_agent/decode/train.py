"""Teaching a pattern by example — the recording → fit step.

:class:`PatternTrainer` is the pure, side-effect-free core of "train a pattern
by doing it a few times". It collects labelled feature frames (``rest`` vs
``active``) grouped by **rep**, then fits the :mod:`classifier` and packages the
result — detector, cross-validated score and rest-vs-active ranking — into a
:class:`~pieeg_agent.decode.patterns.TrainedPattern` ready for the live bank and
the store.

Keeping this layer free of threads, timers and the cascade makes the training
protocol trivial to test and lets two very different front-ends drive it the
same way: the CLI brackets each segment with a timed ``record``, while the web
UI opens and closes segments from its guided overlay. Both just call
``open_segment`` / ``add`` / ``close_segment`` / ``fit``.
"""

from __future__ import annotations

from .calibrate import ACTIVE, REST, ContrastiveCalibrator
from .classifier import PatternClassifier
from .features import FeatureLayout
from .patterns import TrainedPattern

import numpy as np


class TrainingError(RuntimeError):
    """Raised when a pattern cannot be fit (missing a class, too few reps)."""


class PatternTrainer:
    """Collects labelled frames by rep and fits a detector from them."""

    def __init__(self, name: str, layout: FeatureLayout):
        self.name = name
        self.layout = layout
        self._cal = ContrastiveCalibrator(layout)
        self._rep = 0
        self._open_label: str | None = None
        self._open_count = 0

    # ── recording protocol ──────────────────────────────────────────────
    def open_segment(self, label: str) -> None:
        if label not in (REST, ACTIVE):
            raise TrainingError(f"label must be {REST!r} or {ACTIVE!r}")
        if self._open_label is not None:
            raise TrainingError("a segment is already open")
        self._open_label = label
        self._open_count = 0

    def add(self, features: np.ndarray) -> None:
        """Add one live frame to the open segment (ignored if none is open)."""
        if self._open_label is None:
            return
        self._cal.add(self._open_label, features, rep=self._rep)
        self._open_count += 1

    def close_segment(self) -> int:
        """Close the open segment; an ``active`` segment ends the rep."""
        if self._open_label is None:
            return 0
        n = self._open_count
        if self._open_label == ACTIVE:
            self._rep += 1            # each active take is its own CV fold
        self._open_label = None
        self._open_count = 0
        return n

    # ── state ───────────────────────────────────────────────────────────
    def counts(self) -> dict:
        return {
            "rest": self._cal.n_rest,
            "active": self._cal.n_active,
            "reps": len(self._cal.reps),
            "recording": self._open_label,
        }

    @property
    def ready(self) -> bool:
        return self._cal.n_rest >= 2 and self._cal.n_active >= 2

    # ── fit ─────────────────────────────────────────────────────────────
    def fit(self, *, threshold: float = 0.6, l2: float = 1e-2,
            group_lasso: float = 5e-3, note: str = "") -> TrainedPattern:
        if self._cal.n_rest < 2 or self._cal.n_active < 2:
            raise TrainingError(
                "need at least 2 rest and 2 active frames "
                f"(have {self._cal.n_rest} rest, {self._cal.n_active} active)"
            )
        X, y, groups = self._cal.dataset()
        mean, std = self._cal.standardizer()
        clf = PatternClassifier(self.layout, l2=l2, group_lasso=group_lasso)
        clf.fit(X, y, mean=mean, std=std)
        cv = clf.cross_validate(X, y, groups)
        ranking = self._cal.ranking()
        return TrainedPattern(
            name=self.name,
            classifier=clf,
            threshold=float(threshold),
            cv=cv.to_dict(),
            ranking=ranking.to_dict(),
            note=note,
        )
