"""Tests for the trainable-pattern pipeline (features → calibrate → classify → bank)."""

import numpy as np

from pieeg_agent.decode import (
    ContrastiveCalibrator,
    FeatureLayout,
    PatternBank,
    PatternClassifier,
    PatternStore,
    TrainedPattern,
    Welford,
    balanced_accuracy,
    cohens_d,
    extract_frame,
    layout_for,
)
from pieeg_agent.decode.features import BLOCK, PER_CHANNEL_FEATURES


# ── feature extraction ──────────────────────────────────────────────────
def test_extract_frame_shape_and_layout():
    n_ch, n_bands = 4, 5
    bands = np.abs(np.random.default_rng(0).normal(1.0, 0.2, size=(n_ch, n_bands)))
    raw = np.random.default_rng(1).normal(0.0, 10.0, size=(256, n_ch))
    vec = extract_frame(bands, raw)
    assert vec.shape == (n_ch * BLOCK,)
    assert BLOCK == n_bands + 2  # five bands + logRMS + logLineLength


def test_layout_groups_and_names():
    lay = FeatureLayout(channel_labels=("Fp1", "Fp2", "O1"))
    assert lay.dim == 3 * BLOCK
    groups = lay.channel_groups()
    assert len(groups) == 3 and len(groups[0]) == BLOCK
    assert lay.channel_of(BLOCK) == 1          # first index of 2nd block
    assert lay.names[0] == f"Fp1/{PER_CHANNEL_FEATURES[0]}"


def test_layout_for_invents_labels():
    lay = layout_for(None, 8)
    assert lay.channel_labels == tuple(f"Ch{i}" for i in range(8))


def test_louder_channel_has_larger_rms_feature():
    bands = np.ones((2, 5))
    raw = np.zeros((256, 2))
    rng = np.random.default_rng(2)
    raw[:, 0] = rng.normal(0, 1.0, 256)
    raw[:, 1] = rng.normal(0, 50.0, 256)       # channel 1 much louder
    vec = extract_frame(bands, raw)
    rms_idx = PER_CHANNEL_FEATURES.index("logRMS")
    assert vec[BLOCK + rms_idx] > vec[rms_idx]  # ch1 logRMS > ch0 logRMS


# ── Welford / Cohen's d ─────────────────────────────────────────────────
def test_welford_matches_numpy():
    rng = np.random.default_rng(3)
    data = rng.normal(2.0, 1.5, size=(500, 4))
    w = Welford(4)
    w.update_batch(data)
    assert np.allclose(w.mean, data.mean(axis=0))
    assert np.allclose(w.var, data.var(axis=0, ddof=1))


def test_cohens_d_sign_and_zero():
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, size=(400, 3))
    b = rng.normal(1.0, 1.0, size=(400, 3))
    d = cohens_d(a.mean(0), a.var(0, ddof=1), 400, b.mean(0), b.var(0, ddof=1), 400)
    assert np.all(d > 0.5)  # active (b) clearly above rest (a)
    assert np.allclose(
        cohens_d(np.zeros(3), np.ones(3), 1, np.zeros(3), np.ones(3), 1), 0.0
    )  # too few samples → zero


# ── synthetic separable dataset ─────────────────────────────────────────
def _make_dataset(layout, *, reps=4, per=25, shift_channel=2, seed=0):
    """Rest vs active where one channel's block separates the classes."""
    rng = np.random.default_rng(seed)
    cal = ContrastiveCalibrator(layout)
    block = layout.channel_groups()[shift_channel]
    for r in range(reps):
        rep_bias = rng.normal(0.0, 0.05, size=layout.dim)  # benign per-rep offset
        rest = rng.normal(0.0, 0.3, size=(per, layout.dim)) + rep_bias
        active = rng.normal(0.0, 0.3, size=(per, layout.dim)) + rep_bias
        active[:, list(block)] += 1.6                      # the learnable effect
        cal.add_batch("rest", rest, rep=r)
        cal.add_batch("active", active, rep=r)
    return cal


def test_calibrator_ranks_the_informative_channel():
    lay = FeatureLayout(channel_labels=tuple(f"C{i}" for i in range(4)))
    cal = _make_dataset(lay, shift_channel=2)
    ranking = cal.ranking()
    assert ranking.channel_importance()[0]["channel"] == "C2"
    assert ranking.to_dict()["n_active"] == cal.n_active
    assert "caveat" in ranking.to_dict()


def test_classifier_learns_and_cross_validates():
    lay = FeatureLayout(channel_labels=tuple(f"C{i}" for i in range(4)))
    cal = _make_dataset(lay, shift_channel=1)
    X, y, groups = cal.dataset()
    clf = PatternClassifier(lay).fit(X, y)
    # Trains well in-sample…
    assert balanced_accuracy(y, clf.predict(X)) > 0.9
    # …and generalises across reps (leave-one-rep-out).
    cv = clf.cross_validate(X, y, groups)
    assert cv.balanced_accuracy is not None and cv.balanced_accuracy > 0.8
    assert cv.n_folds == 4
    # Group-lasso highlights the channel that actually carries the signal.
    assert clf.channel_importance()[0]["channel"] == "C1"


def test_cv_needs_two_reps():
    lay = FeatureLayout(channel_labels=("C0", "C1"))
    cal = _make_dataset(lay, reps=1, shift_channel=0)
    X, y, groups = cal.dataset()
    cv = PatternClassifier(lay).fit(X, y).cross_validate(X, y, groups)
    assert cv.balanced_accuracy is None  # cannot leave a rep out with only one


def test_classifier_round_trip():
    lay = FeatureLayout(channel_labels=tuple(f"C{i}" for i in range(3)))
    cal = _make_dataset(lay, shift_channel=0)
    X, y, _ = cal.dataset()
    clf = PatternClassifier(lay).fit(X, y)
    restored = PatternClassifier.from_dict(clf.to_dict())
    assert np.allclose(restored.predict_proba(X), clf.predict_proba(X))


# ── live bank ───────────────────────────────────────────────────────────
class _BP:
    """Minimal stand-in for BandPowers (per_channel + timestamp)."""

    def __init__(self, per_channel, timestamp):
        self.per_channel = per_channel
        self.timestamp = timestamp


def _trained_pattern(name="eyes-closed", shift_channel=2):
    lay = FeatureLayout(channel_labels=tuple(f"C{i}" for i in range(4)))
    cal = _make_dataset(lay, shift_channel=shift_channel)
    X, y, groups = cal.dataset()
    clf = PatternClassifier(lay).fit(X, y)
    cv = clf.cross_validate(X, y, groups)
    return TrainedPattern(name=name, classifier=clf, threshold=0.6,
                          cv=cv.to_dict(), ranking=cal.ranking().to_dict())


def test_pattern_bank_fires_on_active_and_clears():
    pat = _trained_pattern()
    lay = pat.layout
    block = lay.channel_groups()[2]
    bank = PatternBank(frame_hz=8.0, smooth_tau=0.2)
    bank.add(pat)

    rng = np.random.default_rng(9)
    detections = []
    # Rest frames → no activation.
    for _ in range(10):
        rest = rng.normal(0.0, 0.3, size=lay.dim)
        detections += bank.score_features(rest, 0.0)
    assert detections == []
    assert not bank.snapshot()[0]["active"]

    # Active frames → smoothed probability crosses the threshold and fires once.
    fired = []
    for _ in range(20):
        active = rng.normal(0.0, 0.3, size=lay.dim)
        active[list(block)] += 1.6
        fired += bank.score_features(active, 1.0)
    assert any(d.name == "eyes-closed" and d.active for d in fired)
    assert bank.snapshot()[0]["active"]


def test_pattern_bank_score_frame_adapter():
    pat = _trained_pattern()
    bank = PatternBank()
    bank.add(pat)
    n_ch = pat.layout.n_channels
    bands = np.ones((n_ch, 5))
    raw = np.random.default_rng(0).normal(0, 5.0, size=(256, n_ch))
    out = bank.score_frame(_BP(bands, 123.0), raw)
    assert isinstance(out, list)  # adapter runs end-to-end without error


def test_trained_pattern_persist_round_trip(tmp_path):
    store = PatternStore(tmp_path)
    pat = _trained_pattern(name="Jaw Clench")
    bank = PatternBank()
    bank.add(pat)
    bank.persist(store, "Jaw Clench")
    assert store.exists("Jaw Clench")
    meta = store.list_meta()[0]
    assert meta["name"] == "Jaw Clench"
    assert meta["metric"] == "balanced_accuracy"

    # A fresh bank reloads it and can score with it.
    bank2 = PatternBank()
    loaded = bank2.load_all(store)
    assert "Jaw Clench" in loaded
    assert bank2.get("Jaw Clench") is not None
