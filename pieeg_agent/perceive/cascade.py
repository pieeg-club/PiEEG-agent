"""The perception cascade — ring → features → quality → state → events.

A dedicated thread reads the most-recent window from the inlet's ring at the
feature rate, extracts band powers and quality, folds them into the state
estimator, and — once per state period — emits a :class:`NeuralState` and any
debounced :class:`NeuralEvent`s. Results are cached for pull-based access and
pushed to optional callbacks.

The cascade never touches ``pylsl`` and never blocks the intake thread: it
only *reads* the ring (copy-on-read), so a slow consumer here can never apply
back-pressure to acquisition. This is the boundary between T0 (raw, high-rate)
and the language-sized tiers the LLM will eventually see.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .artifacts import ArtifactEvent, ArtifactMonitor
from .events import EventDetector, NeuralEvent
from .features import BandPowerExtractor, BandPowers
from .quality import QualityMonitor, SignalQuality
from .state import NeuralState, StateEstimator

logger = logging.getLogger("pieeg.agent.perceive")


@dataclass
class CascadeConfig:
    """Tunables for the perception cascade."""

    fft_size: int = 512
    feature_hz: float = 8.0       # how often features are extracted
    state_hz: float = 1.0         # how often a NeuralState is emitted
    mains_hz: float = 50.0        # powerline frequency for the line-noise check
    ema_tau: float = 1.5          # spectral-shape smoothing time constant (s)
    norm_window: int = 600        # rolling window for index normalisation
    event_hi: float = 0.70
    event_lo: float = 0.30
    event_min_dwell: float = 2.0
    quality_floor: float = 0.5
    events_keep: int = 256        # ring depth for the recent-event log
    detect_artifacts: bool = True  # run the blink / jaw / motion monitor


OnState = Callable[[NeuralState], None]
OnEvent = Callable[[NeuralEvent], None]
OnArtifact = Callable[[ArtifactEvent], None]
# (band_powers, quality, raw_window, timestamps) — for downstream decoders that
# want every feature frame without the cascade importing them.
OnFrame = Callable[[BandPowers, SignalQuality, np.ndarray, np.ndarray], None]


class PerceptionCascade:
    """Runs the feature→state→event pipeline over an :class:`LSLInlet`'s ring."""

    def __init__(
        self,
        inlet,
        config: CascadeConfig | None = None,
        *,
        on_state: OnState | None = None,
        on_event: OnEvent | None = None,
        on_artifact: OnArtifact | None = None,
        on_frame: OnFrame | None = None,
    ):
        self._inlet = inlet
        self._cfg = config or CascadeConfig()
        self._on_state = on_state
        self._on_event = on_event
        self._on_artifact = on_artifact
        self._on_frame = on_frame

        srate = inlet.sample_rate or 250.0
        self._extractor = BandPowerExtractor(srate, self._cfg.fft_size)
        self._quality = QualityMonitor(srate, mains_hz=self._cfg.mains_hz)
        self._artifacts = (
            ArtifactMonitor(srate, inlet.channel_labels)
            if self._cfg.detect_artifacts else None
        )
        self._estimator = StateEstimator(
            ema_tau=self._cfg.ema_tau, norm_window=self._cfg.norm_window
        )
        self._events = EventDetector(
            hi=self._cfg.event_hi,
            lo=self._cfg.event_lo,
            min_dwell=self._cfg.event_min_dwell,
            quality_floor=self._cfg.quality_floor,
        )

        self._latest_state: NeuralState | None = None
        self._latest_bp: BandPowers | None = None
        self._latest_quality: SignalQuality | None = None
        self._event_log: deque[NeuralEvent] = deque(maxlen=self._cfg.events_keep)
        self._artifact_log: deque[ArtifactEvent] = deque(maxlen=self._cfg.events_keep)
        self._lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Telemetry.
        self._ticks = 0
        self._features = 0
        self._states = 0

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="pieeg-perceive", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None

    def set_on_frame(self, callback: OnFrame | None) -> None:
        """Attach or replace the per-frame callback.

        Lets a live decoder (e.g. the pattern bank) be wired after the cascade
        is built, since the two reference each other. A plain attribute write is
        atomic in CPython, so this is safe to call while the worker runs.
        """
        self._on_frame = callback

    def set_on_artifact(self, callback: OnArtifact | None) -> None:
        """Attach or replace the artifact callback (see :meth:`set_on_frame`)."""
        self._on_artifact = callback

    # ── pull-based access ───────────────────────────────────────────────
    def latest_state(self) -> NeuralState | None:
        with self._lock:
            return self._latest_state

    def channel_labels(self) -> list[str]:
        """The inlet's channel labels (read-only convenience for tools)."""
        return list(self._inlet.channel_labels)

    def latest_band_powers(self) -> BandPowers | None:
        with self._lock:
            return self._latest_bp

    def latest_quality(self) -> SignalQuality | None:
        with self._lock:
            return self._latest_quality

    def recent_events(self, n: int = 20) -> list[NeuralEvent]:
        with self._lock:
            if n <= 0 or n >= len(self._event_log):
                return list(self._event_log)
            return list(self._event_log)[-n:]

    def recent_artifacts(self, n: int = 20) -> list[ArtifactEvent]:
        with self._lock:
            if n <= 0 or n >= len(self._artifact_log):
                return list(self._artifact_log)
            return list(self._artifact_log)[-n:]

    def stats(self) -> dict:
        with self._lock:
            last = self._latest_state
            return {
                "ticks": self._ticks,
                "features": self._features,
                "states": self._states,
                "events": len(self._event_log),
                "last_summary": last.summary() if last else "",
            }

    # ── worker ──────────────────────────────────────────────────────────
    def _run(self) -> None:
        period = 1.0 / self._cfg.feature_hz
        ticks_per_state = max(1, int(round(self._cfg.feature_hz / self._cfg.state_hz)))
        next_tick = time.monotonic()
        prev_mono = next_tick
        since_state = 0

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                self._stop.wait(min(period, next_tick - now))
                continue
            next_tick += period
            if next_tick < now:  # fell behind — resync instead of spinning
                next_tick = now + period

            ring = self._inlet.ring
            if ring is None:
                continue
            data, ts = ring.latest(self._cfg.fft_size)
            self._ticks += 1
            if data.shape[0] < self._cfg.fft_size:
                continue

            wall = time.time()
            bp = self._extractor.compute(data, wall)
            if bp is None:
                continue
            quality = self._quality.compute(
                data, self._inlet.channel_labels, wall, bp
            )
            dt = max(now - prev_mono, 1e-3)
            prev_mono = now
            self._estimator.update(bp, quality, dt)

            artifacts: list[ArtifactEvent] = []
            if self._artifacts is not None:
                artifacts = self._artifacts.update(data, ts)

            with self._lock:
                self._latest_bp = bp
                self._latest_quality = quality
                if artifacts:
                    self._artifact_log.extend(artifacts)
            self._features += 1
            self._dispatch_frame(bp, quality, data, ts, artifacts)

            since_state += 1
            if since_state < ticks_per_state:
                continue
            since_state = 0

            state = self._estimator.emit(wall)
            if state is None:
                continue
            events = self._events.update(state)
            with self._lock:
                self._latest_state = state
                self._event_log.extend(events)
                self._states += 1

            self._dispatch(state, events)

    def _dispatch(self, state: NeuralState, events: list[NeuralEvent]) -> None:
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:  # pragma: no cover - callback is user code
                logger.exception("on_state callback raised")
        for ev in events:
            if self._on_event is not None:
                try:
                    self._on_event(ev)
                except Exception:  # pragma: no cover - callback is user code
                    logger.exception("on_event callback raised")

    def _dispatch_frame(
        self,
        bp: BandPowers,
        quality: SignalQuality,
        data: np.ndarray,
        ts: np.ndarray,
        artifacts: list[ArtifactEvent],
    ) -> None:
        if self._on_frame is not None:
            try:
                self._on_frame(bp, quality, data, ts)
            except Exception:  # pragma: no cover - callback is user code
                logger.exception("on_frame callback raised")
        for art in artifacts:
            if self._on_artifact is not None:
                try:
                    self._on_artifact(art)
                except Exception:  # pragma: no cover - callback is user code
                    logger.exception("on_artifact callback raised")
