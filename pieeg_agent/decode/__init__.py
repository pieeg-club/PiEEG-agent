"""Pattern recognition — the agent's higher-order read of the brain.

Where :mod:`pieeg_agent.perceive` produces the always-on tiers (band powers,
state, events, artifacts), this package is what makes the agent more than a
neurofeedback dial: on-demand spectral analysis and *trainable* pattern
detectors the user teaches by example.

* :mod:`spectral`   — IAF, 1/f slope, theta/beta, entropy, alpha asymmetry.
* :mod:`store`      — JSON persistence for trained patterns.
* :mod:`features`   — the feature vector shared by training and live scoring.
* :mod:`calibrate`  — contrastive (rest vs active) calibration + Cohen's d.
* :mod:`classifier` — numpy logistic detector with leave-one-rep-out CV.
* :mod:`patterns`   — the live bank that scores trained detectors each frame.
* :mod:`connectivity` — cross-channel amplitude coupling (a small heatmap).
* :mod:`session`    — record / summarise / compare labelled windows.

``decode`` depends on ``perceive`` (it reads band powers / windows), never the
other way round.
"""

from __future__ import annotations

from .spectral import (
    analyze_spectrum,
    aperiodic_fit,
    frontal_alpha_asymmetry,
    individual_alpha_peak,
    spectral_entropy,
    theta_beta_ratio,
)
from .store import (
    PatternStore,
    SessionStore,
    default_pattern_dir,
    default_session_dir,
    slugify,
)
from .connectivity import band_power_connectivity
from .session import (
    SessionRecorder,
    SessionRecording,
    compare_summaries,
)
from .features import (
    FeatureLayout,
    PER_CHANNEL_FEATURES,
    extract,
    extract_frame,
    layout_for,
)
from .calibrate import (
    ACTIVE,
    REST,
    ContrastiveCalibrator,
    FeatureRanking,
    Welford,
    cohens_d,
)
from .classifier import CVResult, PatternClassifier, balanced_accuracy
from .patterns import PatternBank, PatternDetection, TrainedPattern
from .train import PatternTrainer, TrainingError

__all__ = [
    "analyze_spectrum",
    "individual_alpha_peak",
    "aperiodic_fit",
    "spectral_entropy",
    "theta_beta_ratio",
    "frontal_alpha_asymmetry",
    "PatternStore",
    "SessionStore",
    "default_pattern_dir",
    "default_session_dir",
    "slugify",
    # connectivity
    "band_power_connectivity",
    # sessions
    "SessionRecorder",
    "SessionRecording",
    "compare_summaries",
    # features
    "FeatureLayout",
    "PER_CHANNEL_FEATURES",
    "extract",
    "extract_frame",
    "layout_for",
    # calibration
    "ContrastiveCalibrator",
    "FeatureRanking",
    "Welford",
    "cohens_d",
    "REST",
    "ACTIVE",
    # classifier
    "PatternClassifier",
    "CVResult",
    "balanced_accuracy",
    # live bank
    "PatternBank",
    "PatternDetection",
    "TrainedPattern",
    # training
    "PatternTrainer",
    "TrainingError",
]
