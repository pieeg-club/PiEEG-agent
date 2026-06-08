"""LSL inlet — the high-rate perception intake.

A dedicated background thread pulls EEG sample *chunks* from a PiEEG-server
Lab Streaming Layer outlet and appends them to a :class:`RingBuffer`. This is
the only place that talks to ``pylsl``.

Design choices that matter for high-rate intake:

* **Chunked pulls.** ``pull_chunk`` drains everything buffered in one call,
  so the loop keeps up with 250–500 Hz × 8–32 ch without per-sample overhead.
* **Decoupled from reasoning.** The thread only writes to the ring. Slow
  downstream consumers (DSP, the LLM) never back-pressure intake; stale data
  is overwritten, not queued.
* **Clock-aligned timestamps.** ``proc_clocksync | proc_dejitter`` post-
  processing maps timestamps into the local clock and smooths jitter, so the
  ring's timeline needs no manual correction.
* **Self-healing.** On a lost stream the thread re-resolves and reconnects.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .ring import RingBuffer

logger = logging.getLogger("pieeg.agent.lsl")

# pylsl exception lives in pylsl.util on modern (modular) pylsl; older
# releases re-export it at top level. Import defensively.
try:  # pragma: no cover - import shim
    from pylsl import LostError  # type: ignore
except ImportError:  # pragma: no cover
    from pylsl.util import LostError  # type: ignore

# How long a single pull_chunk waits for the first sample before returning.
_PULL_TIMEOUT = 0.2
# Upper bound on samples returned per pull (a few seconds of head-room).
_PULL_MAX_SAMPLES = 2048
# EMA smoothing for the recent-rate estimate.
_RATE_EMA_ALPHA = 0.2
# Fallback sample rate when a stream advertises irregular/zero nominal rate.
_FALLBACK_SRATE = 250.0


@dataclass
class LSLStreamConfig:
    """Which LSL stream to attach to and how to buffer it."""

    name: str = "PiEEG"
    stype: str = "EEG"
    resolve_by: str = "type"          # "name" | "type"
    resolve_timeout: float = 5.0      # seconds to wait for the outlet
    ring_seconds: float = 60.0        # ring depth in seconds of signal
    recover: bool = True              # let LSL silently recover the stream


class LSLInlet:
    """Resolves one LSL stream and pumps it into a ring buffer in a thread."""

    def __init__(self, config: LSLStreamConfig | None = None):
        self._cfg = config or LSLStreamConfig()
        self._inlet = None
        self._ring: RingBuffer | None = None

        # Stream metadata (populated on resolve()).
        self._stream_name = ""
        self._num_channels = 0
        self._nominal_srate = 0.0
        self._channel_labels: list[str] = []

        # Threading.
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Telemetry.
        self._n_total = 0
        self._t0 = 0.0
        self._last_recv = 0.0
        self._last_ts = 0.0
        self._ema_rate = 0.0
        self._lost_count = 0
        self._connected = False

    # ── Public metadata ────────────────────────────────────────────────
    @property
    def ring(self) -> RingBuffer | None:
        return self._ring

    @property
    def stream_name(self) -> str:
        return self._stream_name

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def sample_rate(self) -> float:
        return self._nominal_srate

    @property
    def channel_labels(self) -> list[str]:
        return list(self._channel_labels)

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Resolve + connect ──────────────────────────────────────────────
    def resolve(self) -> bool:
        """Find the configured outlet and build the inlet + ring.

        Returns ``True`` on success. Safe to call again to reconnect.
        """
        from pylsl import resolve_byprop

        prop = "name" if self._cfg.resolve_by == "name" else "type"
        value = self._cfg.name if prop == "name" else self._cfg.stype
        logger.info("Resolving LSL stream by %s=%r …", prop, value)

        results = resolve_byprop(prop, value, 1, self._cfg.resolve_timeout)
        if not results:
            logger.warning(
                "No LSL stream found with %s=%r within %.1fs",
                prop, value, self._cfg.resolve_timeout,
            )
            return False
        return self.connect_info(results[0])

    def connect_info(self, info) -> bool:
        """Build the inlet + ring from an already-resolved ``StreamInfo``.

        This is the shared connect path used both by :meth:`resolve` and by
        callers that pre-select a specific outlet (e.g. the EEG group out of a
        multi-group PiEEG profile). After a successful connect the inlet
        retargets future reconnects to this stream *by name*, so a self-heal
        re-finds the same group rather than whatever responds first.
        """
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter

        self._inlet = StreamInlet(
            info,
            max_buflen=max(1, int(round(self._cfg.ring_seconds))),
            max_chunklen=0,
            recover=self._cfg.recover,
            processing_flags=proc_clocksync | proc_dejitter,
        )

        # Pull the full info (includes channel descriptions).
        full = self._inlet.info()
        self._stream_name = full.name()
        self._num_channels = int(full.channel_count())
        srate = float(full.nominal_srate())
        self._nominal_srate = srate if srate > 0 else _FALLBACK_SRATE
        self._channel_labels = _read_channel_labels(full, self._num_channels)

        # Re-resolve by this exact name on reconnect (stable group selection).
        if self._stream_name:
            self._cfg.name = self._stream_name
            self._cfg.resolve_by = "name"

        capacity = int(np.ceil(self._nominal_srate * self._cfg.ring_seconds))
        self._ring = RingBuffer(max(capacity, 1), self._num_channels)
        self._connected = True

        logger.info(
            "Connected to %r — %d ch @ %.1f Hz (ring=%d samples / %.0fs)",
            self._stream_name, self._num_channels, self._nominal_srate,
            self._ring.capacity, self._cfg.ring_seconds,
        )
        return True

    # ── Thread lifecycle ───────────────────────────────────────────────
    def start(self) -> None:
        """Begin pumping samples into the ring (resolves if needed)."""
        if self._thread and self._thread.is_alive():
            return
        if self._inlet is None and not self.resolve():
            raise RuntimeError(
                f"Could not resolve LSL stream "
                f"{self._cfg.resolve_by}={self._cfg.name!r}/{self._cfg.stype!r}. "
                f"Is PiEEG-server running with --lsl?"
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="pieeg-lsl-inlet", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to stop and wait briefly for it to exit."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:  # pragma: no cover - best effort
                pass

    # ── Worker loop ────────────────────────────────────────────────────
    def _run(self) -> None:
        from pylsl import local_clock

        self._t0 = local_clock()
        self._last_recv = self._t0

        while not self._stop.is_set():
            try:
                samples, timestamps = self._inlet.pull_chunk(
                    timeout=_PULL_TIMEOUT, max_samples=_PULL_MAX_SAMPLES
                )
            except LostError:
                self._connected = False
                self._lost_count += 1
                logger.warning("LSL stream lost — attempting to reconnect …")
                self._reconnect()
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("LSL pull error: %s", exc)
                time.sleep(0.1)
                continue

            if not samples:
                continue

            data = np.asarray(samples, dtype=np.float32)
            ts = np.asarray(timestamps, dtype=np.float64)
            if self._ring is not None:
                self._ring.push_chunk(data, ts)

            self._update_rate(len(samples), float(ts[-1]))

    def _update_rate(self, n: int, last_ts: float) -> None:
        from pylsl import local_clock

        now = local_clock()
        dt = now - self._last_recv
        if dt > 0:
            inst = n / dt
            self._ema_rate = (
                inst if self._ema_rate == 0.0
                else (1 - _RATE_EMA_ALPHA) * self._ema_rate + _RATE_EMA_ALPHA * inst
            )
        self._last_recv = now
        self._last_ts = last_ts
        self._n_total += n

    def _reconnect(self) -> None:
        """Re-resolve the stream after a loss, honouring the stop signal."""
        backoff = 0.5
        while not self._stop.is_set():
            if self.resolve():
                logger.info("Reconnected to %r", self._stream_name)
                return
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 5.0)

    # ── Telemetry ──────────────────────────────────────────────────────
    def stats(self) -> dict:
        from pylsl import local_clock

        overall = 0.0
        if self._n_total and self._t0:
            elapsed = local_clock() - self._t0
            if elapsed > 0:
                overall = self._n_total / elapsed
        # Staleness is measured purely on the consumer clock (time since we
        # last received a chunk), so it is robust no matter which clock domain
        # the producer stamps samples in. PiEEG-server, for instance, stamps
        # Unix time rather than LSL local_clock, which would make any absolute
        # transport-latency figure meaningless.
        staleness = (local_clock() - self._last_recv) if self._last_recv else None
        # Backlog waiting in the inlet — the real "are we keeping up?" signal.
        queued = 0
        try:
            if self._inlet is not None:
                queued = int(self._inlet.samples_available())
        except Exception:  # pragma: no cover - best effort
            queued = -1
        return {
            "stream": self._stream_name,
            "channels": self._num_channels,
            "nominal_srate": self._nominal_srate,
            "samples": self._n_total,
            "effective_rate": overall,
            "recent_rate": self._ema_rate,
            "ring_fill": self._ring.fill if self._ring else 0,
            "ring_capacity": self._ring.capacity if self._ring else 0,
            "last_ts": self._last_ts,
            "staleness": staleness,
            "queued": queued,
            "connected": self._connected,
            "lost_count": self._lost_count,
        }


def _read_channel_labels(info, num_channels: int) -> list[str]:
    """Extract channel labels from an LSL StreamInfo description.

    Falls back to ``Ch0, Ch1, …`` for any missing entries so callers always
    get exactly ``num_channels`` labels.
    """
    labels: list[str] = []
    try:
        ch = info.desc().child("channels").child("channel")
        while not ch.empty() and len(labels) < num_channels:
            label = ch.child_value("label")
            labels.append(label if label else f"Ch{len(labels)}")
            ch = ch.next_sibling()
    except Exception:  # pragma: no cover - description is optional in LSL
        pass
    while len(labels) < num_channels:
        labels.append(f"Ch{len(labels)}")
    return labels[:num_channels]


# ── Multi-group discovery ──────────────────────────────────────────────────
#
# A PiEEG profile can publish several outlets that all advertise type "EEG"
# (e.g. EEG_PiEEG, EOG_PiEEG, AUX_PiEEG). Resolving by type alone is therefore
# ambiguous — it returns whichever outlet answers first. These helpers let the
# caller see every candidate and pick the brain-EEG group deterministically.

# Name fragments that signal a *non*-brain auxiliary group.
_AUX_HINTS = ("eog", "aux", "acc", "gyro", "ppg", "trig", "marker", "temp")


def discover_streams(timeout: float = 2.0) -> list:
    """Return every LSL ``StreamInfo`` currently resolvable on the network."""
    from pylsl import resolve_streams

    return list(resolve_streams(timeout))


def _eeg_score(info) -> tuple[int, int]:
    """Rank key for an EEG candidate: (name preference, channel count).

    A name beginning with ``EEG`` is the strongest signal that this is the
    brain group; auxiliary names are demoted; ties break on channel count.
    """
    name = info.name().lower()
    pref = 0
    if name.startswith("eeg"):
        pref += 100
    if any(hint in name for hint in _AUX_HINTS):
        pref -= 100
    return (pref, int(info.channel_count()))


def rank_eeg_streams(streams: list, stype: str = "EEG") -> list:
    """Filter ``streams`` to the given type and rank best-EEG-group first."""
    candidates = [s for s in streams if s.type() == stype]
    return sorted(candidates, key=_eeg_score, reverse=True)
