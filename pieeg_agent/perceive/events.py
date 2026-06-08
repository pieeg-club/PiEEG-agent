"""T3 events — sparse, debounced state transitions.

A raw 0…1 index that hovers near a threshold would otherwise emit a storm of
flip-flop events. Each tracked signal therefore runs through a Schmitt trigger
(separate enter/exit thresholds) *and* a minimum-dwell timer: the level must
hold past the threshold for a sustained period before a single transition
event fires. This is what makes the event stream cheap enough to feed an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import NeuralState


@dataclass
class NeuralEvent:
    """A single debounced transition in the neural state."""

    timestamp: float
    type: str            # e.g. "focus_high", "relax_low", "quality_drop"
    value: float
    detail: str
    severity: str = "info"   # "info" | "warn"

    def summary(self) -> str:
        return f"{self.type}: {self.detail}"


class _Schmitt:
    """Two-threshold latch with a minimum-dwell debounce.

    ``state`` is ``True`` once ``value`` has stayed at/above ``hi`` for
    ``min_dwell`` seconds, and flips back to ``False`` once it has stayed
    at/below ``lo`` for the same. Only sustained crossings emit an event.
    """

    def __init__(
        self,
        hi: float,
        lo: float,
        min_dwell: float,
        rising_type: str,
        falling_type: str,
        label: str,
        initial: bool = False,
    ):
        self.hi = hi
        self.lo = lo
        self.min_dwell = min_dwell
        self.rising_type = rising_type
        self.falling_type = falling_type
        self.label = label
        self.state = initial
        self._cand: bool | None = None
        self._since = 0.0

    def update(self, value: float, t: float) -> NeuralEvent | None:
        target = self.state
        if not self.state and value >= self.hi:
            target = True
        elif self.state and value <= self.lo:
            target = False

        if target == self.state:
            self._cand = None
            return None

        # A crossing is pending — require it to persist for min_dwell.
        if self._cand != target:
            self._cand = target
            self._since = t
            return None
        if t - self._since < self.min_dwell:
            return None

        self.state = target
        self._cand = None
        etype = self.rising_type if target else self.falling_type
        return NeuralEvent(timestamp=t, type=etype, value=value, detail="")


class EventDetector:
    """Wires Schmitt triggers over the state's indices and signal quality."""

    def __init__(
        self,
        *,
        hi: float = 0.70,
        lo: float = 0.30,
        min_dwell: float = 2.0,
        quality_floor: float = 0.5,
    ):
        self._triggers = {
            "focus": _Schmitt(hi, lo, min_dwell, "focus_high", "focus_low", "focus"),
            "relax": _Schmitt(hi, lo, min_dwell, "relax_high", "relax_low", "relax"),
            "engagement": _Schmitt(
                hi, lo, min_dwell, "engagement_high", "engagement_low", "engagement"
            ),
        }
        # Quality starts "OK" (True); a sustained dip below the floor trips it.
        self._quality = _Schmitt(
            quality_floor + 0.15,
            quality_floor,
            min_dwell,
            "quality_ok",
            "quality_drop",
            "signal quality",
            initial=True,
        )

    def update(self, state: NeuralState) -> list[NeuralEvent]:
        events: list[NeuralEvent] = []

        values = {
            "focus": state.focus,
            "relax": state.relax,
            "engagement": state.engagement,
        }
        for name, trig in self._triggers.items():
            ev = trig.update(values[name], state.timestamp)
            if ev is not None:
                direction = "rose" if ev.type.endswith("_high") else "fell"
                ev.detail = f"{name} {direction} to {ev.value:.2f}"
                events.append(ev)

        qev = self._quality.update(state.signal_quality, state.timestamp)
        if qev is not None:
            if qev.type == "quality_drop":
                qev.severity = "warn"
                detail = f"signal quality degraded to {state.signal_quality:.2f}"
                if state.bad_channels:
                    detail += " (" + ", ".join(state.bad_channels) + ")"
            else:
                detail = f"signal quality recovered to {state.signal_quality:.2f}"
            qev.detail = detail
            events.append(qev)

        return events
