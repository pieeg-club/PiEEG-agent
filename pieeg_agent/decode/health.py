"""Pattern health monitoring — confidence scoring and degradation detection.

A :class:`PatternHealthMonitor` tracks predictions over time and computes a
**confidence score** alongside each pattern's probability. High confidence means
the current feature distribution matches what the pattern saw during training;
low confidence suggests electrode drift, placement changes, or other systematic
shifts that make the pattern's predictions unreliable.

The monitor uses two signals:

1. **Feature drift** — How far have the current features drifted from the
   training distribution? Measured via simplified KL divergence between running
   statistics and the training mean/std stored at pattern creation.

2. **Prediction consistency** — How stable are recent predictions? High variance
   suggests the pattern is firing erratically (noisy data, poor signal quality).

**Confidence formula:**

.. code-block:: python

    confidence = (1 - drift_penalty) * (1 - variance_penalty)
    
    where:
        drift_penalty = min(1.0, kl_divergence / drift_threshold)
        variance_penalty = min(1.0, prediction_std / variance_threshold)

**Status classification:**

- ``healthy``: confidence ≥ 0.70
- ``degraded``: 0.50 ≤ confidence < 0.70  
- ``needs_retrain``: confidence < 0.50

The monitor is **purely observational** — it never modifies pattern weights or
predictions. It provides honest metrics for users to decide when retraining is
warranted.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

# Tuning constants (can be made configurable later)
_HISTORY_SIZE = 20  # Track last N predictions for variance
_DRIFT_THRESHOLD = 1.0  # KL divergence threshold for drift penalty (research: <0.5=similar, >1.0=different)
_VARIANCE_THRESHOLD = 0.25  # Std threshold for variance penalty
_MIN_SAMPLES_FOR_CONFIDENCE = 5  # Need this many samples before reporting confidence


class HealthStatus(str, Enum):
    """Pattern health classification."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    NEEDS_RETRAIN = "needs_retrain"


@dataclass
class HealthMetrics:
    """Health assessment for one pattern at one moment."""

    confidence: float  # 0..1, higher = more trustworthy
    status: HealthStatus
    drift_score: float  # 0..1+, 0 = perfect match to training
    prediction_std: float  # Std of recent predictions
    n_predictions: int  # How many predictions have been tracked
    last_updated: float  # Timestamp of last update

    def to_dict(self) -> dict:
        return {
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "drift_score": round(self.drift_score, 3),
            "prediction_std": round(self.prediction_std, 3),
            "n_predictions": self.n_predictions,
            "last_updated": self.last_updated,
        }


@dataclass
class _PatternHealthState:
    """Internal state for one pattern's health tracking."""

    # Training baseline (captured when pattern is added)
    training_mean: np.ndarray | None = None
    training_std: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)

    # Running statistics (online Welford for current feature distribution)
    n_samples: int = 0
    running_mean: np.ndarray | None = None
    running_m2: np.ndarray | None = None  # For computing variance

    # Prediction history (for consistency check)
    predictions: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_SIZE))

    # Last computed metrics
    last_confidence: float = 1.0
    last_status: HealthStatus = HealthStatus.HEALTHY
    last_drift: float = 0.0


class PatternHealthMonitor:
    """Tracks pattern reliability over time via confidence scoring.

    **Usage:**

    1. When a pattern is trained, call :meth:`register_pattern` with its
       training feature statistics (mean, std).
    2. Each frame, after scoring, call :meth:`track_prediction` with the
       feature vector and prediction.
    3. Query :meth:`get_health` to retrieve current confidence and status.
    4. Use :meth:`should_retrain` to get a boolean recommendation.

    **Thread safety:** This class is **not** thread-safe. If the pattern bank
    runs on a cascade thread, the monitor should be accessed only from that
    thread or protected by the same lock.
    """

    def __init__(
        self,
        *,
        drift_threshold: float = _DRIFT_THRESHOLD,
        variance_threshold: float = _VARIANCE_THRESHOLD,
        min_samples: int = _MIN_SAMPLES_FOR_CONFIDENCE,
    ):
        self._drift_threshold = drift_threshold
        self._variance_threshold = variance_threshold
        self._min_samples = min_samples
        self._state: dict[str, _PatternHealthState] = {}

    # ── registration ─────────────────────────────────────────────────────
    def register_pattern(
        self,
        pattern_id: str,
        training_mean: np.ndarray | None = None,
        training_std: np.ndarray | None = None,
    ) -> None:
        """Register a newly trained pattern with its training statistics.

        :param pattern_id: Unique pattern name.
        :param training_mean: Mean feature vector from training data (optional).
        :param training_std: Std feature vector from training data (optional).

        If training statistics are not provided, drift detection will be
        disabled for this pattern (only prediction variance will be tracked).
        """
        state = _PatternHealthState(
            training_mean=np.array(training_mean, dtype=np.float64) if training_mean is not None else None,
            training_std=np.array(training_std, dtype=np.float64) if training_std is not None else None,
            created_at=time.time(),
        )
        self._state[pattern_id] = state

    def unregister_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern from health tracking.

        :returns: True if pattern was tracked, False if not found.
        """
        return self._state.pop(pattern_id, None) is not None

    def is_registered(self, pattern_id: str) -> bool:
        """Check if a pattern is currently being tracked."""
        return pattern_id in self._state

    # ── tracking ─────────────────────────────────────────────────────────
    def track_prediction(
        self,
        pattern_id: str,
        features: np.ndarray,
        probability: float,
    ) -> None:
        """Update health state with a new prediction.

        :param pattern_id: Pattern being scored.
        :param features: Feature vector used for this prediction.
        :param probability: Predicted probability (0..1).

        Updates running feature statistics and prediction history. Does not
        return anything; call :meth:`get_health` afterward to retrieve metrics.
        """
        state = self._state.get(pattern_id)
        if state is None:
            return  # Pattern not registered, silently ignore

        features = np.asarray(features, dtype=np.float64).flatten()

        # Update Welford online mean/variance
        state.n_samples += 1
        if state.running_mean is None:
            state.running_mean = features.copy()
            state.running_m2 = np.zeros_like(features)
        else:
            delta = features - state.running_mean
            state.running_mean += delta / state.n_samples
            delta2 = features - state.running_mean
            state.running_m2 += delta * delta2

        # Store prediction for variance tracking
        state.predictions.append(float(probability))

        # Recompute confidence
        self._update_metrics(pattern_id, state)

    def _update_metrics(self, pattern_id: str, state: _PatternHealthState) -> None:
        """Recompute confidence, drift, and status for one pattern."""
        # Not enough data yet
        if state.n_samples < self._min_samples:
            state.last_confidence = 1.0  # Optimistic until proven otherwise
            state.last_status = HealthStatus.HEALTHY
            state.last_drift = 0.0
            return

        # Compute drift penalty
        drift_penalty = 0.0
        if state.training_mean is not None and state.training_std is not None:
            kl = self._compute_kl_divergence(
                state.running_mean,
                np.sqrt(state.running_m2 / state.n_samples),
                state.training_mean,
                state.training_std,
            )
            drift_penalty = min(1.0, kl / self._drift_threshold)
            state.last_drift = kl

        # Compute variance penalty
        variance_penalty = 0.0
        if len(state.predictions) >= self._min_samples:
            pred_std = float(np.std(state.predictions))
            variance_penalty = min(1.0, pred_std / self._variance_threshold)

        # Combine into confidence
        confidence = (1.0 - drift_penalty) * (1.0 - variance_penalty)
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Classify status
        if confidence >= 0.70:
            status = HealthStatus.HEALTHY
        elif confidence >= 0.50:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.NEEDS_RETRAIN

        state.last_confidence = confidence
        state.last_status = status

    def _compute_kl_divergence(
        self,
        mean1: np.ndarray,
        std1: np.ndarray,
        mean2: np.ndarray,
        std2: np.ndarray,
    ) -> float:
        """Simplified KL divergence between two Gaussian distributions.

        Assumes independence (diagonal covariance). Returns average KL per feature.
        """
        eps = 1e-8
        std1 = np.maximum(std1, eps)
        std2 = np.maximum(std2, eps)

        var1 = std1**2
        var2 = std2**2

        # KL(N1 || N2) = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 1/2
        kl = np.log(std2 / std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5

        # Average across features
        return float(np.mean(kl))

    # ── querying ─────────────────────────────────────────────────────────
    def get_health(self, pattern_id: str) -> HealthMetrics | None:
        """Retrieve current health metrics for a pattern.

        :returns: :class:`HealthMetrics` if pattern is registered and has
                  enough data, else None.
        """
        state = self._state.get(pattern_id)
        if state is None:
            return None

        pred_std = float(np.std(state.predictions)) if state.predictions else 0.0

        return HealthMetrics(
            confidence=state.last_confidence,
            status=state.last_status,
            drift_score=state.last_drift,
            prediction_std=pred_std,
            n_predictions=state.n_samples,
            last_updated=time.time(),
        )

    def should_retrain(self, pattern_id: str) -> bool:
        """Quick boolean check: does this pattern need retraining?

        :returns: True if status is ``needs_retrain``, False otherwise.
        """
        health = self.get_health(pattern_id)
        return health is not None and health.status == HealthStatus.NEEDS_RETRAIN

    def get_all_health(self) -> dict[str, HealthMetrics]:
        """Retrieve health metrics for all registered patterns.

        :returns: Mapping from pattern_id to HealthMetrics.
        """
        out = {}
        for pattern_id in self._state:
            health = self.get_health(pattern_id)
            if health is not None:
                out[pattern_id] = health
        return out

    # ── alerts ───────────────────────────────────────────────────────────
    def check_degradation(
        self,
        min_status: Literal["degraded", "needs_retrain"] = "degraded",
    ) -> list[str]:
        """List patterns that have degraded below a threshold.

        :param min_status: Minimum status to trigger alert.
        :returns: List of pattern IDs with status at or below threshold.

        Example::

            alerts = monitor.check_degradation("degraded")
            if alerts:
                print(f"⚠️ Patterns need attention: {alerts}")
        """
        threshold_map = {
            "degraded": [HealthStatus.DEGRADED, HealthStatus.NEEDS_RETRAIN],
            "needs_retrain": [HealthStatus.NEEDS_RETRAIN],
        }
        target_statuses = threshold_map[min_status]

        alerts = []
        for pattern_id, state in self._state.items():
            if state.last_status in target_statuses:
                alerts.append(pattern_id)
        return alerts

    def reset_pattern(self, pattern_id: str) -> bool:
        """Reset health tracking for a pattern (e.g., after retraining).

        Clears running statistics and prediction history, keeping training
        baseline. Returns True if pattern was found.
        """
        state = self._state.get(pattern_id)
        if state is None:
            return False

        state.n_samples = 0
        state.running_mean = None
        state.running_m2 = None
        state.predictions.clear()
        state.last_confidence = 1.0
        state.last_status = HealthStatus.HEALTHY
        state.last_drift = 0.0

        return True
