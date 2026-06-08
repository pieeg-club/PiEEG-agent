"""Thread-safe multi-channel ring buffer — the agent's short-term memory.

A single producer (the LSL inlet thread) appends chunks; multiple consumers
(DSP/feature workers, agent tools) read the most-recent window. Reads return
an independent copy in chronological order, so consumers never see a torn
write and never alias the live buffer.

Storage is a fixed ``(capacity, num_channels)`` float32 array with a parallel
``(capacity,)`` float64 timestamp array. Appends are O(chunk) with at most one
wrap split; reads are O(window).
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer for multi-channel samples + timestamps."""

    def __init__(self, capacity: int, num_channels: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if num_channels < 1:
            raise ValueError("num_channels must be >= 1")
        self._cap = int(capacity)
        self._ch = int(num_channels)
        self._data = np.zeros((self._cap, self._ch), dtype=np.float32)
        self._ts = np.zeros(self._cap, dtype=np.float64)
        self._write = 0          # index of the next slot to write
        self._count = 0          # total samples ever written (monotonic)
        self._lock = threading.Lock()

    # ── Properties ─────────────────────────────────────────────────────
    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def num_channels(self) -> int:
        return self._ch

    @property
    def fill(self) -> int:
        """Number of valid samples currently stored (<= capacity)."""
        return min(self._count, self._cap)

    @property
    def total(self) -> int:
        """Total samples ever written (monotonic, ignores overwrites)."""
        return self._count

    def __len__(self) -> int:
        return self.fill

    # ── Write ──────────────────────────────────────────────────────────
    def push_chunk(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Append a block of samples.

        ``samples`` has shape ``(n, num_channels)``; ``timestamps`` has shape
        ``(n,)``. Both are copied into the ring. If ``n`` exceeds capacity,
        only the most-recent ``capacity`` samples are retained.
        """
        samples = np.asarray(samples, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != self._ch:
            raise ValueError(
                f"samples must be (n, {self._ch}); got {samples.shape}"
            )
        n = samples.shape[0]
        if n == 0:
            return

        with self._lock:
            cap = self._cap
            if n >= cap:
                # Incoming block is larger than the ring — keep only the tail.
                self._data[:] = samples[-cap:]
                self._ts[:] = timestamps[-cap:]
                self._write = 0
                self._count += n
                return

            w = self._write
            end = w + n
            if end <= cap:
                self._data[w:end] = samples
                self._ts[w:end] = timestamps
            else:
                first = cap - w
                self._data[w:] = samples[:first]
                self._ts[w:] = timestamps[:first]
                self._data[: end - cap] = samples[first:]
                self._ts[: end - cap] = timestamps[first:]
            self._write = end % cap
            self._count += n

    # ── Read ───────────────────────────────────────────────────────────
    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return up to the ``n`` most-recent samples, oldest-first.

        Returns ``(data, timestamps)`` where ``data`` is ``(m, num_channels)``
        and ``timestamps`` is ``(m,)`` with ``m = min(n, fill)``. Both are
        fresh contiguous copies safe to hand to NumPy/FFT code.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        with self._lock:
            valid = min(self._count, self._cap)
            m = min(n, valid)
            if m == 0:
                return (
                    np.empty((0, self._ch), dtype=np.float32),
                    np.empty(0, dtype=np.float64),
                )
            cap = self._cap
            start = (self._write - m) % cap
            if start + m <= cap:
                data = self._data[start : start + m].copy()
                ts = self._ts[start : start + m].copy()
            else:
                first = cap - start
                data = np.concatenate(
                    (self._data[start:], self._data[: m - first]), axis=0
                )
                ts = np.concatenate((self._ts[start:], self._ts[: m - first]))
            return data, ts

    def clear(self) -> None:
        """Reset the buffer to empty (keeps capacity/channels)."""
        with self._lock:
            self._write = 0
            self._count = 0
