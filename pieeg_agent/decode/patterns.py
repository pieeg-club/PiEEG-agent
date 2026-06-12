"""The live pattern bank — trained detectors scored on every frame.

:mod:`calibrate` and :mod:`classifier` turn examples into a detector; this
module is what runs that detector *live*. Registered with the cascade's
``on_frame`` callback, the bank extracts the shared feature vector once per
frame, scores every trained pattern, smooths each probability and applies
hysteresis so a noisy frame near the threshold does not chatter. A clean
rising edge (a pattern becoming active) is reported through ``on_detection``;
the always-current activations are available as a snapshot for the UI and the
``detect_patterns`` tool.

A :class:`TrainedPattern` bundles the detector with the honest numbers behind
it — the cross-validated balanced accuracy and the rest-vs-active ranking — and
serialises through :class:`~pieeg_agent.decode.store.PatternStore`, so a pattern
the user taught today survives a restart with its provenance intact.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .classifier import PatternClassifier
from .features import FeatureLayout, extract_frame
from .store import PatternStore

SCHEMA_VERSION = 1


@dataclass
class PatternDetection:
    """A pattern crossing from inactive to active (or its live activation)."""

    name: str
    probability: float
    active: bool
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "probability": round(self.probability, 3),
            "active": self.active,
            "timestamp": self.timestamp,
        }


@dataclass
class TrainedPattern:
    """A detector plus the provenance that makes it trustworthy."""

    name: str
    classifier: PatternClassifier
    threshold: float = 0.6
    cv: dict = field(default_factory=dict)
    ranking: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    note: str = ""
    # Training statistics for health monitoring
    training_mean: np.ndarray | None = None
    training_std: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "threshold": self.threshold,
            # Top-level mirrors so the store's lightweight listing shows the
            # honest score without deserialising the whole model.
            "labels": list(self.classifier.layout.channel_labels),
            "score": self.cv.get("balanced_accuracy"),
            "metric": "balanced_accuracy",
            "n_reps": self.cv.get("n_folds"),
            "cv": self.cv,
            "ranking": self.ranking,
            "created_at": self.created_at,
            "note": self.note,
            "classifier": self.classifier.to_dict(),
            "training_mean": self.training_mean.tolist() if self.training_mean is not None else None,
            "training_std": self.training_std.tolist() if self.training_std is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainedPattern":
        training_mean = data.get("training_mean")
        training_std = data.get("training_std")
        return cls(
            name=data["name"],
            classifier=PatternClassifier.from_dict(data["classifier"]),
            threshold=float(data.get("threshold", 0.6)),
            cv=data.get("cv", {}),
            ranking=data.get("ranking", {}),
            created_at=float(data.get("created_at", time.time())),
            note=data.get("note", ""),
            training_mean=np.array(training_mean) if training_mean is not None else None,
            training_std=np.array(training_std) if training_std is not None else None,
        )

    @property
    def layout(self) -> FeatureLayout:
        return self.classifier.layout


OnDetection = Callable[[PatternDetection], None]


@dataclass
class _Runtime:
    ema: float = 0.0
    active: bool = False


class PatternBank:
    """Holds trained patterns and scores them frame by frame."""

    def __init__(
        self,
        *,
        frame_hz: float = 8.0,
        smooth_tau: float = 0.4,
        release_margin: float = 0.15,
        on_detection: OnDetection | None = None,
    ):
        from .health import PatternHealthMonitor

        self._frame_hz = float(frame_hz) or 8.0
        dt = 1.0 / self._frame_hz
        self._alpha = 1.0 - math.exp(-dt / max(smooth_tau, 1e-3))
        self._release_margin = float(release_margin)
        self._on_detection = on_detection
        self._patterns: dict[str, TrainedPattern] = {}
        self._runtime: dict[str, _Runtime] = {}
        self._health = PatternHealthMonitor()

    # ── registry ────────────────────────────────────────────────────────
    def add(self, pattern: TrainedPattern) -> None:
        self._patterns[pattern.name] = pattern
        self._runtime[pattern.name] = _Runtime()
        # Register with health monitor
        self._health.register_pattern(
            pattern.name,
            training_mean=pattern.training_mean,
            training_std=pattern.training_std,
        )

    def remove(self, name: str) -> bool:
        self._runtime.pop(name, None)
        self._health.unregister_pattern(name)
        return self._patterns.pop(name, None) is not None

    def names(self) -> list[str]:
        return list(self._patterns)

    def get(self, name: str) -> TrainedPattern | None:
        return self._patterns.get(name)

    def __len__(self) -> int:
        return len(self._patterns)

    # ── live scoring ────────────────────────────────────────────────────
    def score_features(self, features: np.ndarray, timestamp: float) -> list[PatternDetection]:
        """Score every pattern on one feature vector; return new activations."""
        fired: list[PatternDetection] = []
        for name, pat in self._patterns.items():
            if features.shape[0] != pat.layout.dim:
                continue  # montage mismatch — skip rather than guess
            rt = self._runtime[name]
            p = pat.classifier.score_one(features)
            rt.ema += self._alpha * (p - rt.ema)
            # Track prediction for health monitoring
            self._health.track_prediction(name, features, p)
            on = pat.threshold
            off = pat.threshold - self._release_margin
            was = rt.active
            if not rt.active and rt.ema >= on:
                rt.active = True
            elif rt.active and rt.ema < off:
                rt.active = False
            if rt.active and not was:
                det = PatternDetection(name, rt.ema, True, timestamp)
                fired.append(det)
                if self._on_detection is not None:
                    self._on_detection(det)
        return fired

    def score_frame(self, bp, raw_window: np.ndarray) -> list[PatternDetection]:
        """Adapter for the cascade ``on_frame`` callback (band powers + window)."""
        if not self._patterns:
            return []
        features = extract_frame(bp.per_channel, raw_window)
        return self.score_features(features, bp.timestamp)

    def snapshot(self) -> list[dict]:
        """Current smoothed activation of every pattern (for the UI / tools)."""
        out = []
        for name, pat in self._patterns.items():
            rt = self._runtime[name]
            entry = {
                "name": name,
                "probability": round(rt.ema, 3),
                "active": rt.active,
                "threshold": pat.threshold,
                "balanced_accuracy": pat.cv.get("balanced_accuracy"),
            }
            # Add health metrics if available
            health = self._health.get_health(name)
            if health is not None:
                entry["health"] = health.to_dict()
            out.append(entry)
        return out

    # ── persistence ─────────────────────────────────────────────────────
    def persist(self, store: PatternStore, name: str) -> str:
        pat = self._patterns[name]
        return store.save(name, pat.to_dict())

    def load_all(self, store: PatternStore) -> list[str]:
        """Load every saved pattern into the bank; return the names loaded."""
        loaded = []
        for meta in store.list_meta():
            try:
                data = store.load(meta["slug"])
                if data is None:
                    continue
                self.add(TrainedPattern.from_dict(data))
                loaded.append(meta["name"])
            except Exception:  # pragma: no cover - skip a corrupt file, keep going
                continue
        return loaded
