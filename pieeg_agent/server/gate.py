"""Action gating and audit — the safety layer in front of every device action.

Reading the brain is free; *acting* on the device is not. Every mutating
control-plane call goes through an :class:`ActionGate` that enforces three
independent checks and records the outcome:

* **Allowlist** — only explicitly enabled actions may run. The default policy
  allows nothing, so actions are opt-in by construction.
* **Dry-run** — when set, an authorized action is *previewed* (what would be
  sent) but never actually executed. This is the safe default for the LLM.
* **Cooldown** — a minimum interval between real executions of the same
  action, so a misfiring loop can't hammer the hardware.

Every attempt — allowed, denied, dry-run or failed — is written to an
:class:`AuditLog` (in memory, and optionally appended to a JSONL file) so
there is always a record of what the agent did or tried to do.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

logger = logging.getLogger("pieeg.agent.gate")


@dataclass(frozen=True)
class ActionPolicy:
    """What the agent is permitted to do on the device.

    The defaults are intentionally cautious: an empty allowlist (nothing runs)
    and ``dry_run=True`` (preview only). Callers opt in explicitly.
    """

    allowed_actions: frozenset[str] = frozenset()
    dry_run: bool = True
    cooldown_s: float = 3.0

    def permits(self, action: str) -> bool:
        return action in self.allowed_actions

    @classmethod
    def allow(cls, *actions: str, dry_run: bool = True, cooldown_s: float = 3.0
              ) -> "ActionPolicy":
        """Convenience builder: ``ActionPolicy.allow('set_filter', ...)``."""
        return cls(frozenset(actions), dry_run=dry_run, cooldown_s=cooldown_s)


@dataclass
class Decision:
    """The gate's verdict for one action attempt."""

    action: str
    allowed: bool
    reason: str = ""


@dataclass
class AuditEntry:
    """One recorded action attempt."""

    timestamp: float
    action: str
    params: dict
    outcome: str            # "executed" | "dry_run" | "denied" | "error"
    reason: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    """Append-only log of action attempts (in memory + optional JSONL file)."""

    def __init__(self, *, keep: int = 512, path: str | None = None):
        self._entries: deque[AuditEntry] = deque(maxlen=keep)
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> str | None:
        """The JSONL file this log persists to, or ``None`` if in memory only."""
        return self._path

    def record(self, entry: AuditEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            logger.info(
                "AUDIT %s -> %s%s",
                entry.action,
                entry.outcome,
                f" ({entry.reason})" if entry.reason else "",
            )
            if self._path:
                self._append_file(entry)

    def recent(self, n: int = 20) -> list[AuditEntry]:
        with self._lock:
            if n <= 0 or n >= len(self._entries):
                return list(self._entries)
            return list(self._entries)[-n:]

    def to_list(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def _append_file(self, entry: AuditEntry) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as exc:  # pragma: no cover - disk/permission issue
            logger.warning("Could not write audit log %s: %s", self._path, exc)


class ActionGate:
    """Enforces the policy and records every attempt to the audit log."""

    def __init__(
        self,
        policy: ActionPolicy,
        audit: AuditLog | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.policy = policy
        self.audit = audit or AuditLog()
        self._clock = clock
        self._last_fired: dict[str, float] = {}
        self._lock = threading.Lock()

    def authorize(self, action: str) -> Decision:
        """Allowlist check only (cooldown is enforced at execution time)."""
        if not self.policy.permits(action):
            return Decision(action, False, "not in allowlist")
        return Decision(action, True)

    def run(self, action: str, params: dict, fn: Callable[[], dict]) -> dict:
        """Authorize, then dry-run / cooldown-guard / execute ``fn``.

        Returns a JSON-friendly result envelope with an ``outcome`` field, so
        callers (and the LLM) always learn what the gate decided. ``fn`` is
        only ever called for a real execution.
        """
        now = self._clock()

        # 1) Allowlist.
        decision = self.authorize(action)
        if not decision.allowed:
            return self._finish(action, params, "denied", decision.reason)

        # 2) Dry-run preview (no execution, no cooldown consumed).
        if self.policy.dry_run:
            return self._finish(
                action, params, "dry_run",
                "dry-run: not sent to the device",
                extra={"would_send": {"action": action, "params": params}},
            )

        # 3) Cooldown between real executions.
        with self._lock:
            last = self._last_fired.get(action)
            remaining = 0.0 if last is None else self.policy.cooldown_s - (now - last)
            if remaining > 0:
                return self._finish(
                    action, params, "denied",
                    f"cooldown active, {remaining:.1f}s remaining",
                )
            # Reserve the slot before executing so concurrent calls can't race.
            self._last_fired[action] = now

        # 4) Execute.
        try:
            result = fn() or {}
        except Exception as exc:
            return self._finish(
                action, params, "error", f"{type(exc).__name__}: {exc}"
            )
        return self._finish(
            action, params, "executed", "", extra={"result": result}
        )

    # ── helpers ──────────────────────────────────────────────────────────
    def _finish(
        self,
        action: str,
        params: dict,
        outcome: str,
        reason: str,
        *,
        extra: dict | None = None,
    ) -> dict:
        self.audit.record(
            AuditEntry(
                timestamp=time.time(),
                action=action,
                params=_redact(params),
                outcome=outcome,
                reason=reason,
            )
        )
        envelope = {"outcome": outcome, "action": action}
        if reason:
            envelope["reason"] = reason
        if extra:
            envelope.update(extra)
        return envelope


# Keys whose values must never reach the audit log or an LLM result.
_SECRET_KEYS = ("token", "authorization", "api_key", "secret", "password")


def _redact(params: dict) -> dict:
    out: dict = {}
    for key, value in params.items():
        if any(s in key.lower() for s in _SECRET_KEYS):
            out[key] = "***"
        elif isinstance(value, dict):
            out[key] = _redact(value)
        else:
            out[key] = value
    return out
