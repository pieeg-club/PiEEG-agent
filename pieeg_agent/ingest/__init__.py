"""High-rate EEG ingestion: LSL inlet thread + thread-safe ring buffer.

This package is the agent's perception intake. It owns the only ``pylsl``
dependency and exposes two primitives the rest of the agent builds on:

* :class:`RingBuffer` — short-term signal memory (fixed window, copy-on-read)
* :class:`LSLInlet`   — background thread that drains a PiEEG-server LSL
  outlet into the ring
"""

from __future__ import annotations

from .lsl_inlet import LSLInlet, LSLStreamConfig
from .ring import RingBuffer

__all__ = ["LSLInlet", "LSLStreamConfig", "RingBuffer"]
