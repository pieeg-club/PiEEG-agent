"""Read-only neural tools — the agent's senses into the live cascade.

The LLM never sees the raw stream. Instead it *pulls* language-sized facts
through these tools, each a thin, side-effect-free read over a
:class:`~pieeg_agent.perceive.cascade.PerceptionCascade`:

* ``get_neural_state``    — the current ~1 Hz smoothed state.
* ``get_band_powers``     — relative band powers (and per-channel detail).
* ``get_recent_events``   — the debounced event log.
* ``get_channel_quality`` — per-channel signal-quality verdicts.
* ``summarize_last``      — a compact natural-language status line.

A :class:`Tool` couples a :class:`ToolSpec` (what the model sees) with a
handler (what actually runs). :class:`NeuralTools` binds the set to one
cascade and dispatches calls by name. Everything returned is JSON-friendly so
the agent loop can serialise results straight into a tool-result message.

These are deliberately *read-only*; the gated server actions arrive in a later
phase and live in their own module so the boundary stays obvious.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..llm.provider import ToolSpec
from ..perceive.cascade import PerceptionCascade

ToolHandler = Callable[[dict], dict]


@dataclass
class Tool:
    """A model-callable tool: its advertised spec plus the handler to run."""

    spec: ToolSpec
    handler: ToolHandler

    @property
    def name(self) -> str:
        return self.spec.name


def _spec(name: str, description: str, properties: dict | None = None,
          required: list[str] | None = None) -> ToolSpec:
    """Build a ToolSpec with a standard object JSON Schema."""
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    )


class NeuralTools:
    """The read-only tool set bound to one perception cascade."""

    def __init__(self, cascade: PerceptionCascade):
        self._cascade = cascade
        self._tools: dict[str, Tool] = {}
        self._register_all()

    # ── registry surface ────────────────────────────────────────────────
    def specs(self) -> list[ToolSpec]:
        """The tool specs to advertise to the model."""
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Dispatch a tool call by name, returning a JSON-friendly dict.

        Unknown tools and handler errors are returned as ``{"error": …}``
        rather than raised, so a model that fabricates a tool name or bad
        arguments gets a usable message instead of crashing the loop.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "available": self.names()}
        try:
            return tool.handler(arguments or {})
        except Exception as exc:  # defensive: a tool must never kill the loop
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── registration ────────────────────────────────────────────────────
    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_all(self) -> None:
        self._add(Tool(
            _spec(
                "get_neural_state",
                "Current smoothed neural state (~1 Hz): focus, relax and "
                "engagement indices (0..1, within-session relative), dominant "
                "band, relative band powers, signal quality and warm-up flag.",
            ),
            self._get_neural_state,
        ))
        self._add(Tool(
            _spec(
                "get_band_powers",
                "Relative EEG band powers (delta/theta/alpha/beta/gamma) "
                "averaged across channels. Set per_channel=true to also get "
                "the per-channel breakdown.",
                {
                    "per_channel": {
                        "type": "boolean",
                        "description": "Include per-channel band powers.",
                    }
                },
            ),
            self._get_band_powers,
        ))
        self._add(Tool(
            _spec(
                "get_recent_events",
                "The most recent debounced neural events (focus/relax/"
                "engagement transitions and signal-quality changes).",
                {
                    "limit": {
                        "type": "integer",
                        "description": "Max events to return (default 10).",
                        "minimum": 1,
                        "maximum": 100,
                    }
                },
            ),
            self._get_recent_events,
        ))
        self._add(Tool(
            _spec(
                "get_channel_quality",
                "Per-channel signal-quality verdicts (good/flat/rail/noisy/"
                "line) with RMS, line-noise ratio and a 0..1 score.",
            ),
            self._get_channel_quality,
        ))
        self._add(Tool(
            _spec(
                "summarize_last",
                "A compact one-line natural-language summary of the latest "
                "neural state — handy for a quick status check.",
            ),
            self._summarize_last,
        ))

    # ── handlers (pull from the cascade) ─────────────────────────────────
    def _get_neural_state(self, _args: dict) -> dict:
        state = self._cascade.latest_state()
        if state is None:
            return _not_ready()
        return state.to_dict()

    def _get_band_powers(self, args: dict) -> dict:
        bp = self._cascade.latest_band_powers()
        if bp is None:
            return _not_ready()
        out: dict[str, Any] = {
            "timestamp": bp.timestamp,
            "relative": {k: round(v, 4) for k, v in bp.relative().items()},
            "dominant": bp.dominant(),
            "n_channels": bp.n_channels,
            "units": "relative power (sums to ~1)",
        }
        if args.get("per_channel"):
            labels = self._cascade.channel_labels()
            out["per_channel"] = _per_channel_bands(bp, labels)
        return out

    def _get_recent_events(self, args: dict) -> dict:
        limit = _as_int(args.get("limit"), 10, lo=1, hi=100)
        events = self._cascade.recent_events(limit)
        return {
            "count": len(events),
            "events": [
                {
                    "timestamp": ev.timestamp,
                    "type": ev.type,
                    "value": round(ev.value, 3),
                    "detail": ev.detail,
                    "severity": ev.severity,
                }
                for ev in events
            ],
        }

    def _get_channel_quality(self, _args: dict) -> dict:
        q = self._cascade.latest_quality()
        if q is None:
            return _not_ready()
        return {
            "timestamp": q.timestamp,
            "overall": round(q.overall, 3),
            "worst": q.worst.label if q.worst else None,
            "channels": [
                {
                    "index": c.index,
                    "label": c.label,
                    "status": c.status,
                    "score": round(c.score, 3),
                    "rms_uv": round(c.rms, 2),
                    "line_ratio": round(c.line_ratio, 2),
                    "rail_frac": round(c.rail_frac, 4),
                }
                for c in q.channels
            ],
        }

    def _summarize_last(self, _args: dict) -> dict:
        state = self._cascade.latest_state()
        if state is None:
            return _not_ready()
        return {
            "summary": state.summary(),
            "warming_up": state.warming_up,
        }


# ── helpers ─────────────────────────────────────────────────────────────────


def _not_ready() -> dict:
    return {
        "status": "no_data",
        "detail": "The perception cascade has not produced a state yet "
        "(still filling the first analysis window).",
    }


def _per_channel_bands(bp, labels) -> list[dict]:
    from ..perceive.features import BAND_NAMES

    rows: list[dict] = []
    per = bp.per_channel
    n_ch = per.shape[0]
    for i in range(n_ch):
        total = float(per[i].sum()) or 1.0
        label = labels[i] if i < len(labels) else f"Ch{i}"
        rows.append(
            {
                "channel": label,
                **{
                    b[0].lower(): round(float(per[i, j]) / total, 4)
                    for j, b in enumerate(BAND_NAMES)
                },
            }
        )
    return rows


def _as_int(value, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))
