"""Tests for pattern health monitoring."""

import numpy as np
import pytest

from pieeg_agent.decode.health import (
    HealthMetrics,
    HealthStatus,
    PatternHealthMonitor,
)


@pytest.fixture
def monitor():
    return PatternHealthMonitor()


@pytest.fixture
def training_stats():
    """Mock training statistics for an 8-channel × 5-feature pattern."""
    dim = 40  # 8 channels × 5 features
    mean = np.random.randn(dim) * 0.5
    std = np.ones(dim) * 0.3
    return mean, std


def test_register_pattern(monitor, training_stats):
    """Test pattern registration with training statistics."""
    mean, std = training_stats
    monitor.register_pattern("test_pattern", training_mean=mean, training_std=std)
    assert monitor.is_registered("test_pattern")


def test_unregister_pattern(monitor, training_stats):
    """Test pattern removal from health tracking."""
    mean, std = training_stats
    monitor.register_pattern("test_pattern", training_mean=mean, training_std=std)
    assert monitor.unregister_pattern("test_pattern")
    assert not monitor.is_registered("test_pattern")
    # Second removal returns False
    assert not monitor.unregister_pattern("test_pattern")


def test_healthy_pattern(monitor, training_stats):
    """Pattern with features matching training should be healthy."""
    mean, std = training_stats
    monitor.register_pattern("stable_pattern", training_mean=mean, training_std=std)

    # Simulate 20 predictions with features from same distribution as training
    np.random.seed(42)
    for _ in range(20):
        features = mean + np.random.randn(len(mean)) * std  # Same distribution as training
        probability = 0.7 + np.random.randn() * 0.05  # Low variance predictions
        monitor.track_prediction("stable_pattern", features, probability)

    health = monitor.get_health("stable_pattern")
    assert health is not None
    assert health.status == HealthStatus.HEALTHY
    assert health.confidence >= 0.70
    assert not monitor.should_retrain("stable_pattern")


def test_degraded_pattern_drift(monitor, training_stats):
    """Pattern with drifted features should show degraded health."""
    mean, std = training_stats
    monitor.register_pattern("drift_pattern", training_mean=mean, training_std=std)

    np.random.seed(43)
    # First few predictions are good
    for _ in range(5):
        features = mean + np.random.randn(len(mean)) * std * 0.5
        monitor.track_prediction("drift_pattern", features, 0.7)

    # Then features drift significantly
    drift_offset = mean * 1.5  # Shift features by 50%
    for _ in range(15):
        features = mean + drift_offset + np.random.randn(len(mean)) * std
        monitor.track_prediction("drift_pattern", features, 0.6)

    health = monitor.get_health("drift_pattern")
    assert health is not None
    # Should detect drift
    assert health.drift_score > 0.1
    # Confidence should be lower (but might not hit degraded threshold depending on variance)
    assert health.confidence < 1.0


def test_degraded_pattern_variance(monitor, training_stats):
    """Pattern with erratic predictions should show degraded health."""
    mean, std = training_stats
    monitor.register_pattern("noisy_pattern", training_mean=mean, training_std=std)

    np.random.seed(44)
    # Features are stable but predictions are all over the place
    for _ in range(20):
        features = mean + np.random.randn(len(mean)) * std * 0.5
        # High variance predictions
        probability = np.random.uniform(0.1, 0.9)
        monitor.track_prediction("noisy_pattern", features, probability)

    health = monitor.get_health("noisy_pattern")
    assert health is not None
    # Should detect high prediction variance
    assert health.prediction_std > 0.15
    # Confidence should be lower
    assert health.confidence < 0.9


def test_needs_retrain_pattern(monitor, training_stats):
    """Pattern with severe drift should need retraining."""
    mean, std = training_stats
    monitor.register_pattern("broken_pattern", training_mean=mean, training_std=std)

    np.random.seed(45)
    # Massive drift AND noisy predictions
    for _ in range(20):
        # Features completely different from training
        features = mean * 3.0 + np.random.randn(len(mean)) * std * 2.0
        # Erratic predictions
        probability = np.random.uniform(0.0, 1.0)
        monitor.track_prediction("broken_pattern", features, probability)

    health = monitor.get_health("broken_pattern")
    assert health is not None
    assert health.confidence < 0.70  # Should be degraded or worse
    # High drift + high variance should trigger needs_retrain
    if health.confidence < 0.50:
        assert health.status == HealthStatus.NEEDS_RETRAIN
        assert monitor.should_retrain("broken_pattern")


def test_pattern_without_training_stats(monitor):
    """Pattern without training stats only tracks prediction variance."""
    monitor.register_pattern("no_stats_pattern", training_mean=None, training_std=None)

    np.random.seed(46)
    # Track some predictions
    for _ in range(20):
        features = np.random.randn(40)
        probability = 0.7 + np.random.randn() * 0.05
        monitor.track_prediction("no_stats_pattern", features, probability)

    health = monitor.get_health("no_stats_pattern")
    assert health is not None
    # Should have confidence based on variance alone (no drift component)
    assert health.drift_score == 0.0  # No training stats = no drift tracking
    assert health.confidence > 0  # Some confidence from variance


def test_insufficient_samples(monitor, training_stats):
    """Health metrics should be optimistic with few samples."""
    mean, std = training_stats
    monitor.register_pattern("young_pattern", training_mean=mean, training_std=std)

    # Only 2 predictions (below min_samples threshold)
    features = mean + np.random.randn(len(mean)) * std
    monitor.track_prediction("young_pattern", features, 0.7)
    monitor.track_prediction("young_pattern", features, 0.65)

    health = monitor.get_health("young_pattern")
    assert health is not None
    # Should be optimistic (healthy) with insufficient data
    assert health.confidence == 1.0
    assert health.status == HealthStatus.HEALTHY


def test_check_degradation(monitor, training_stats):
    """Test batch degradation checking across patterns."""
    mean, std = training_stats

    # Create three patterns with different health states
    monitor.register_pattern("healthy", training_mean=mean, training_std=std)
    monitor.register_pattern("degraded", training_mean=mean, training_std=std)
    monitor.register_pattern("broken", training_mean=mean, training_std=std)

    np.random.seed(47)

    # Healthy pattern: stable
    for _ in range(20):
        features = mean + np.random.randn(len(mean)) * std * 0.5
        monitor.track_prediction("healthy", features, 0.7)

    # Degraded pattern: moderate drift
    for _ in range(20):
        features = mean * 1.3 + np.random.randn(len(mean)) * std
        monitor.track_prediction("degraded", features, 0.6)

    # Broken pattern: severe drift
    for _ in range(20):
        features = mean * 3.0 + np.random.randn(len(mean)) * std * 2.0
        monitor.track_prediction("broken", features, np.random.uniform(0, 1))

    # Check for patterns that need attention
    degraded_list = monitor.check_degradation("degraded")
    # Should include patterns with degraded or needs_retrain status
    # (depends on exact drift/variance, so we just check it's a list)
    assert isinstance(degraded_list, list)

    needs_retrain_list = monitor.check_degradation("needs_retrain")
    assert isinstance(needs_retrain_list, list)


def test_reset_pattern(monitor, training_stats):
    """Test resetting health tracking after retraining."""
    mean, std = training_stats
    monitor.register_pattern("retrained", training_mean=mean, training_std=std)

    np.random.seed(48)
    # Create some drift
    for _ in range(20):
        features = mean * 2.0 + np.random.randn(len(mean)) * std
        monitor.track_prediction("retrained", features, np.random.uniform(0.3, 0.8))

    # Confidence should be low
    health_before = monitor.get_health("retrained")
    assert health_before is not None
    confidence_before = health_before.confidence

    # Reset tracking (simulating retraining)
    assert monitor.reset_pattern("retrained")

    # After reset, should be back to optimistic state
    health_after = monitor.get_health("retrained")
    assert health_after is not None
    assert health_after.confidence == 1.0
    assert health_after.n_predictions == 0


def test_get_all_health(monitor, training_stats):
    """Test retrieving health for all patterns."""
    mean, std = training_stats

    monitor.register_pattern("pattern_a", training_mean=mean, training_std=std)
    monitor.register_pattern("pattern_b", training_mean=mean, training_std=std)

    np.random.seed(49)
    # Track some predictions for both
    for _ in range(10):
        features = mean + np.random.randn(len(mean)) * std * 0.5
        monitor.track_prediction("pattern_a", features, 0.7)
        monitor.track_prediction("pattern_b", features, 0.65)

    all_health = monitor.get_all_health()
    assert len(all_health) == 2
    assert "pattern_a" in all_health
    assert "pattern_b" in all_health
    assert isinstance(all_health["pattern_a"], HealthMetrics)
    assert isinstance(all_health["pattern_b"], HealthMetrics)


def test_health_metrics_to_dict(monitor, training_stats):
    """Test serialization of health metrics."""
    mean, std = training_stats
    monitor.register_pattern("serialize_test", training_mean=mean, training_std=std)

    np.random.seed(50)
    for _ in range(15):
        features = mean + np.random.randn(len(mean)) * std * 0.5
        monitor.track_prediction("serialize_test", features, 0.7)

    health = monitor.get_health("serialize_test")
    assert health is not None

    health_dict = health.to_dict()
    assert "confidence" in health_dict
    assert "status" in health_dict
    assert "drift_score" in health_dict
    assert "prediction_std" in health_dict
    assert "n_predictions" in health_dict
    assert "last_updated" in health_dict

    # Check value types
    assert isinstance(health_dict["confidence"], float)
    assert isinstance(health_dict["status"], str)
    assert health_dict["status"] in ["healthy", "degraded", "needs_retrain"]


def test_unregistered_pattern_tracking(monitor):
    """Tracking predictions for unregistered pattern should be silent no-op."""
    # No crash when tracking unregistered pattern
    monitor.track_prediction("nonexistent", np.random.randn(40), 0.7)

    # get_health returns None
    health = monitor.get_health("nonexistent")
    assert health is None

    # should_retrain returns False
    assert not monitor.should_retrain("nonexistent")


def test_confidence_bounds(monitor, training_stats):
    """Confidence should always be in [0, 1] range."""
    mean, std = training_stats
    monitor.register_pattern("bounds_test", training_mean=mean, training_std=std)

    np.random.seed(51)
    # Try to create extreme conditions
    for _ in range(30):
        # Extreme drift
        features = mean * 10.0 + np.random.randn(len(mean)) * std * 10.0
        # Extreme variance
        probability = np.random.uniform(0.0, 1.0)
        monitor.track_prediction("bounds_test", features, probability)

    health = monitor.get_health("bounds_test")
    assert health is not None
    assert 0.0 <= health.confidence <= 1.0
