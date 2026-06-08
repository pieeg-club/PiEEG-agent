"""Tests for the state estimator (smoothing, indices, normalisation)."""

import numpy as np

from pieeg_agent.perceive.features import BAND_NAMES, BandPowers
from pieeg_agent.perceive.quality import ChannelQuality, SignalQuality
from pieeg_agent.perceive.state import StateEstimator

ALPHA = {"Delta": 1.0, "Theta": 1.0, "Alpha": 6.0, "Beta": 1.0, "Gamma": 1.0}
BETA = {"Delta": 1.0, "Theta": 1.0, "Alpha": 1.0, "Beta": 6.0, "Gamma": 2.0}


def mk_bp(bands: dict, n_ch: int = 4) -> BandPowers:
    per = np.tile(np.array([bands[b] for b in BAND_NAMES], float), (n_ch, 1))
    return BandPowers(
        timestamp=0.0,
        bands=dict(bands),
        per_channel=per,
        psd=np.zeros((n_ch, 5)),
        freqs=np.zeros(5),
        n_samples=512,
        n_channels=n_ch,
    )


def mk_q(overall: float = 1.0, bad=()) -> SignalQuality:
    chans = [
        ChannelQuality(i, f"Ch{i}", 20.0, 1.0, 0.0, "good", 1.0) for i in range(4)
    ]
    return SignalQuality(timestamp=0.0, channels=chans, overall=overall)


def test_emit_none_before_update():
    assert StateEstimator().emit(0.0) is None


def test_warming_up_flag_clears():
    est = StateEstimator(ema_tau=0.01)
    est.update(mk_bp(ALPHA), mk_q(), 0.125)
    assert est.emit(0.1).warming_up
    # Varied input gives the normalisers a real range to calibrate against.
    for k in range(20):
        est.update(mk_bp(BETA if k % 2 else ALPHA), mk_q(), 0.125)
        est.emit(k * 0.125)
    assert not est.emit(2.0).warming_up


def test_steady_signal_stays_uncalibrated_at_half():
    # A perfectly steady spectrum has no spread to rank against: indices sit at
    # 0.5 and the state honestly reports it is still warming up.
    est = StateEstimator(ema_tau=0.01)
    for _ in range(30):
        est.update(mk_bp(ALPHA), mk_q(), 0.125)
    s = est.emit(5.0)
    assert s.focus == 0.5 and s.relax == 0.5
    assert s.warming_up


def test_dominant_band_tracks_input():
    est = StateEstimator(ema_tau=0.01)
    for _ in range(20):
        est.update(mk_bp(ALPHA), mk_q(), 0.125)
    s = est.emit(1.0)
    assert s.dominant_band == "Alpha"
    assert 0.0 <= s.focus <= 1.0
    assert 0.0 <= s.relax <= 1.0
    assert 0.0 <= s.engagement <= 1.0


def test_focus_higher_for_beta_relax_higher_for_alpha():
    est = StateEstimator(ema_tau=0.01)
    # Populate the rolling normaliser with both extremes.
    for k in range(30):
        est.update(mk_bp(BETA if k % 2 else ALPHA), mk_q(), 0.125)
        est.emit(k * 0.125)
    for _ in range(20):
        est.update(mk_bp(BETA), mk_q(), 0.125)
    beta_state = est.emit(100.0)
    for _ in range(20):
        est.update(mk_bp(ALPHA), mk_q(), 0.125)
    alpha_state = est.emit(200.0)

    assert beta_state.focus >= alpha_state.focus
    assert alpha_state.relax >= beta_state.relax


def test_quality_propagates():
    est = StateEstimator(ema_tau=0.01)
    est.update(mk_bp(ALPHA), mk_q(overall=0.4), 0.125)
    assert est.emit(0.0).signal_quality == 0.4
