"""High-level, typed PiEEG-server actions over the control client.

This is the boundary between "what the agent wants to do" and the raw protocol
in :mod:`pieeg_agent.server.client`. Two kinds of method live here:

* **Reads** (``server_info``, ``list_webhooks``, ``read_registers``,
  ``osc_status``, ``lsl_status``) — safe queries, *not* gated.
* **Actions** (``set_filter``, ``start_recording``, ``create_webhook``,
  ``apply_register_preset``, ``start_osc``, …) — every one is routed through
  an :class:`~pieeg_agent.server.gate.ActionGate`, so allowlist, dry-run,
  cooldown and audit apply uniformly. Each returns the gate's result envelope.

Action names match the policy allowlist (and, where natural, the server's
``cmd`` names) so enabling an action is unambiguous.
"""

from __future__ import annotations

from .client import ServerControlClient
from .gate import ActionGate

# Registers the server accepts (CHnSET only, 0x05–0x0C). We mirror the limit
# client-side so a bad write is refused before it ever reaches the wire.
_ALLOWED_REGS = range(0x05, 0x0D)
_REG_PRESETS = ("normal", "internal_short", "test_signal", "temp_sensor")


class ServerActions:
    """Typed control-plane operations, with mutations behind the gate."""

    def __init__(self, client: ServerControlClient, gate: ActionGate):
        self._client = client
        self._gate = gate

    # ── reads (not gated) ────────────────────────────────────────────────
    def server_info(self) -> dict:
        """The welcome snapshot (sample rate, channels, filter, mock, …)."""
        return self._client.welcome

    def list_webhooks(self) -> dict:
        return self._client.request("webhook_list", reply_key="webhook_rules")

    def read_registers(self) -> dict:
        return self._client.request("reg_read", reply_key="reg_config")

    def osc_status(self) -> dict:
        return self._client.request("osc_status", reply_key="osc_status")

    def lsl_status(self) -> dict:
        return self._client.request("lsl_status", reply_key="lsl_status")

    # ── actions (gated) ──────────────────────────────────────────────────
    def set_filter(
        self, *, enabled: bool = True, lowcut: float = 1.0, highcut: float = 40.0
    ) -> dict:
        params = {"enabled": enabled, "lowcut": lowcut, "highcut": highcut}

        def _do() -> dict:
            if enabled and not (0 < lowcut < highcut <= 125):
                raise ValueError(
                    f"filter bounds out of range: {lowcut}-{highcut} "
                    "(need 0 < lowcut < highcut <= 125)"
                )
            # set_filter has no reply; fire-and-forget then report what we sent.
            self._client.send("set_filter", **params)
            return {"sent": params}

        return self._gate.run("set_filter", params, _do)

    def start_recording(self) -> dict:
        return self._gate.run(
            "start_record", {},
            lambda: self._client.request(
                "start_record", reply_key="record_status"
            ),
        )

    def stop_recording(self) -> dict:
        return self._gate.run(
            "stop_record", {},
            lambda: self._client.request(
                "stop_record", reply_key="record_status"
            ),
        )

    def create_webhook(self, rule: dict) -> dict:
        return self._gate.run(
            "webhook_create", {"rule": rule},
            lambda: self._client.request(
                "webhook_create", reply_key="webhook_created", rule=rule
            ),
        )

    def delete_webhook(self, rule_id: str) -> dict:
        return self._gate.run(
            "webhook_delete", {"rule_id": rule_id},
            lambda: self._client.request(
                "webhook_delete", reply_key="webhook_deleted", rule_id=rule_id
            ),
        )

    def test_webhook(self, rule_id: str) -> dict:
        return self._gate.run(
            "webhook_test", {"rule_id": rule_id},
            lambda: self._client.request(
                "webhook_test", reply_key="webhook_test", rule_id=rule_id
            ),
        )

    def apply_register_preset(self, preset: str) -> dict:
        params = {"preset": preset}

        def _do() -> dict:
            if preset not in _REG_PRESETS:
                raise ValueError(
                    f"unknown preset {preset!r}; choose from "
                    f"{', '.join(_REG_PRESETS)}"
                )
            return self._client.request(
                "reg_preset", reply_key="reg_config", preset=preset
            )

        return self._gate.run("reg_preset", params, _do)

    def write_registers(self, regs: dict) -> dict:
        """Write CHnSET registers (0x05–0x0C). Not exposed to the LLM."""
        params = {"regs": regs}

        def _do() -> dict:
            normalized = _normalize_regs(regs)
            return self._client.request(
                "reg_write", reply_key="reg_config", regs=normalized
            )

        return self._gate.run("reg_write", params, _do)

    def start_osc(self, config: dict | None = None) -> dict:
        params = {"config": config or {}}
        return self._gate.run(
            "osc_start", params,
            lambda: self._client.request(
                "osc_start", reply_key="osc_status", config=config or {}
            ),
        )

    def stop_osc(self) -> dict:
        return self._gate.run(
            "osc_stop", {},
            lambda: self._client.request("osc_stop", reply_key="osc_status"),
        )

    def start_lsl(self, config: dict | None = None) -> dict:
        params = {"config": config or {}}
        return self._gate.run(
            "lsl_start", params,
            lambda: self._client.request(
                "lsl_start", reply_key="lsl_status", config=config or {}
            ),
        )

    def stop_lsl(self) -> dict:
        return self._gate.run(
            "lsl_stop", {},
            lambda: self._client.request("lsl_stop", reply_key="lsl_status"),
        )


def _normalize_regs(regs: dict) -> dict:
    """Validate + render a register map as the server expects ({"0x05":"0x00"}).

    Raises ``ValueError`` for any address outside the CHnSET range so a bad
    write never reaches the device.
    """
    out: dict[str, str] = {}
    for addr, value in regs.items():
        a = int(addr, 16) if isinstance(addr, str) else int(addr)
        v = int(value, 16) if isinstance(value, str) else int(value)
        if a not in _ALLOWED_REGS:
            raise ValueError(
                f"register {hex(a)} not allowed (only CHnSET 0x05-0x0C)"
            )
        out[f"0x{a:02x}"] = f"0x{v & 0xFF:02x}"
    return out
