"""Gated actuator tools — the agent's *hands* on the device.

These are the LLM-callable tools that can actually change PiEEG-server, and
they are deliberately kept apart from the read-only senses in
:mod:`pieeg_agent.agent.tools`. Every mutating call is routed through a
:class:`~pieeg_agent.server.actions.ServerActions` facade whose
:class:`~pieeg_agent.server.gate.ActionGate` enforces allowlist, dry-run,
cooldown and audit — so the model's reach is exactly what the policy permits,
and nothing more.

The set is curated: it exposes a safe, human-meaningful subset (filter,
recording, OSC, register *presets*, webhook listing) and intentionally does
*not* expose raw register writes. Two tools are plain reads (``server_status``,
``list_webhooks``); the rest return the gate's result envelope, so the model
always learns whether an action ran, was previewed, or was denied.

This module is opt-in: a copilot only gains hands if it is explicitly built
with these tools (see ``--allow-actions``). Default sessions stay read-only.
"""

from __future__ import annotations

from ..server.actions import ServerActions
from .tools import Tool, _spec

# The register presets the server understands, surfaced to the model.
_PRESETS = ["normal", "internal_short", "test_signal", "temp_sensor"]
_OSC_MODES = ["chatbox", "parameters", "both"]


class ActuatorTools:
    """The gated control tool set bound to one server-actions facade."""

    def __init__(self, actions: ServerActions):
        self._actions = actions
        self._tools: dict[str, Tool] = {}
        self._register_all()

    # ── registry surface (mirrors NeuralTools) ──────────────────────────
    def specs(self):
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "available": self.names()}
        try:
            return tool.handler(arguments or {})
        except Exception as exc:  # an action must never kill the loop
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── registration ────────────────────────────────────────────────────
    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_all(self) -> None:
        self._add(Tool(
            _spec(
                "server_status",
                "Read the PiEEG-server status snapshot: sample rate, channel "
                "count, whether the band-pass filter is on, mock vs hardware, "
                "and LSL/recording state. Read-only.",
            ),
            self._server_status,
        ))
        self._add(Tool(
            _spec(
                "set_filter",
                "Enable/disable or retune the server's band-pass filter. "
                "Bounds must satisfy 0 < lowcut < highcut <= 125 Hz. This is a "
                "gated action and may be previewed (dry-run) rather than sent.",
                {
                    "enabled": {"type": "boolean",
                                "description": "Turn the filter on or off."},
                    "lowcut": {"type": "number",
                               "description": "High-pass corner in Hz (e.g. 1)."},
                    "highcut": {"type": "number",
                                "description": "Low-pass corner in Hz (e.g. 40)."},
                },
            ),
            self._set_filter,
        ))
        self._add(Tool(
            _spec(
                "start_recording",
                "Start recording EEG to a CSV file on the server. Gated action.",
            ),
            self._start_recording,
        ))
        self._add(Tool(
            _spec(
                "stop_recording",
                "Stop the current server-side recording. Gated action.",
            ),
            self._stop_recording,
        ))
        self._add(Tool(
            _spec(
                "list_webhooks",
                "List the server's configured webhook rules. Read-only.",
            ),
            self._list_webhooks,
        ))
        self._add(Tool(
            _spec(
                "apply_register_preset",
                "Apply an ADS1299 register preset to all channels: 'normal' "
                "(measure), 'internal_short' (offset/noise check), "
                "'test_signal' (square-wave self-test) or 'temp_sensor'. "
                "Gated action.",
                {
                    "preset": {
                        "type": "string",
                        "enum": _PRESETS,
                        "description": "Which register preset to apply.",
                    }
                },
                required=["preset"],
            ),
            self._apply_register_preset,
        ))
        self._add(Tool(
            _spec(
                "start_osc",
                "Start streaming neural values out over OSC (e.g. to VRChat or "
                "a synth). Gated action.",
                {
                    "host": {"type": "string",
                             "description": "Destination host (default 127.0.0.1)."},
                    "port": {"type": "integer",
                             "description": "Destination UDP port (default 9000)."},
                    "mode": {"type": "string", "enum": _OSC_MODES,
                             "description": "OSC payload style."},
                    "channel": {"type": "integer", "minimum": 0, "maximum": 15,
                                "description": "EEG channel to send (omit for "
                                "the channel average)."},
                },
            ),
            self._start_osc,
        ))
        self._add(Tool(
            _spec(
                "stop_osc",
                "Stop the OSC output stream. Gated action.",
            ),
            self._stop_osc,
        ))

    # ── handlers ────────────────────────────────────────────────────────
    def _server_status(self, _args: dict) -> dict:
        return self._actions.server_info()

    def _set_filter(self, args: dict) -> dict:
        return self._actions.set_filter(
            enabled=bool(args.get("enabled", True)),
            lowcut=float(args.get("lowcut", 1.0)),
            highcut=float(args.get("highcut", 40.0)),
        )

    def _start_recording(self, _args: dict) -> dict:
        return self._actions.start_recording()

    def _stop_recording(self, _args: dict) -> dict:
        return self._actions.stop_recording()

    def _list_webhooks(self, _args: dict) -> dict:
        return self._actions.list_webhooks()

    def _apply_register_preset(self, args: dict) -> dict:
        preset = str(args.get("preset", "")).strip()
        return self._actions.apply_register_preset(preset)

    def _start_osc(self, args: dict) -> dict:
        config: dict = {}
        for key in ("host", "mode"):
            if args.get(key):
                config[key] = args[key]
        if args.get("port") is not None:
            config["port"] = int(args["port"])
        if args.get("channel") is not None:
            config["channel"] = int(args["channel"])
        return self._actions.start_osc(config)

    def _stop_osc(self, _args: dict) -> dict:
        return self._actions.stop_osc()


# The default safe allowlist for an LLM-driven session (raw register writes are
# never included). Used by the CLI when --allow-actions is given.
SAFE_ACTIONS = (
    "set_filter",
    "start_record",
    "stop_record",
    "reg_preset",
    "osc_start",
    "osc_stop",
)
