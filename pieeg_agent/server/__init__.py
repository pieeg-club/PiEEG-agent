"""PiEEG-server control plane — the agent's actuator side.

This package is the *only* place that can change the device, and every
mutation is gated:

* :mod:`pieeg_agent.server.client` — synchronous WebSocket control client.
* :mod:`pieeg_agent.server.gate` — allowlist / dry-run / cooldown + audit log.
* :mod:`pieeg_agent.server.actions` — typed reads and gated actions.

Kept deliberately separate from the read-only perception/agent layers so the
boundary between *observing* the brain and *acting* on the device is obvious.
"""

from __future__ import annotations

from .actions import ServerActions
from .client import ServerControlClient, ServerControlError
from .gate import ActionGate, ActionPolicy, AuditEntry, AuditLog, Decision

__all__ = [
    "ServerControlClient",
    "ServerControlError",
    "ServerActions",
    "ActionGate",
    "ActionPolicy",
    "AuditLog",
    "AuditEntry",
    "Decision",
]
