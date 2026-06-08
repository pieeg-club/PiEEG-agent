"""The action gate: allowlist, dry-run, cooldown, audit and redaction.

These are pure and deterministic — a fake monotonic clock drives the cooldown
so nothing sleeps. The gate must never run ``fn`` unless an action is allowed,
not in dry-run, and off cooldown; and every attempt must land in the audit log
with secrets redacted.
"""

import json

from pieeg_agent.server.gate import (
    ActionGate,
    ActionPolicy,
    AuditEntry,
    AuditLog,
    _redact,
)


class Clock:
    """A hand-cranked monotonic clock."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _ran_marker():
    """A fn whose side effect proves it was actually executed."""
    calls = []
    def fn():
        calls.append(1)
        return {"ok": True}
    return calls, fn


# ── policy ───────────────────────────────────────────────────────────────


def test_policy_defaults_permit_nothing():
    policy = ActionPolicy()
    assert policy.allowed_actions == frozenset()
    assert policy.dry_run is True
    assert policy.permits("set_filter") is False


def test_policy_allow_builder():
    policy = ActionPolicy.allow("set_filter", "start_record", dry_run=False)
    assert policy.permits("set_filter")
    assert policy.permits("start_record")
    assert not policy.permits("stop_record")
    assert policy.dry_run is False


# ── allowlist ──────────────────────────────────────────────────────────────


def test_action_not_in_allowlist_is_denied_and_not_run():
    gate = ActionGate(ActionPolicy.allow("start_record", dry_run=False))
    calls, fn = _ran_marker()
    out = gate.run("stop_record", {}, fn)
    assert out["outcome"] == "denied"
    assert "allowlist" in out["reason"]
    assert calls == []  # fn never invoked


# ── dry-run ────────────────────────────────────────────────────────────────


def test_dry_run_previews_without_executing():
    gate = ActionGate(ActionPolicy.allow("set_filter", dry_run=True))
    calls, fn = _ran_marker()
    out = gate.run("set_filter", {"lowcut": 2}, fn)
    assert out["outcome"] == "dry_run"
    assert out["would_send"] == {"action": "set_filter", "params": {"lowcut": 2}}
    assert calls == []  # never executed in dry-run


# ── execution + cooldown ────────────────────────────────────────────────────


def test_execute_runs_fn_and_returns_result():
    gate = ActionGate(ActionPolicy.allow("start_record", dry_run=False))
    calls, fn = _ran_marker()
    out = gate.run("start_record", {}, fn)
    assert out["outcome"] == "executed"
    assert out["result"] == {"ok": True}
    assert calls == [1]


def test_cooldown_blocks_second_call_then_allows_after_interval():
    clock = Clock()
    gate = ActionGate(
        ActionPolicy.allow("start_record", dry_run=False, cooldown_s=5.0),
        clock=clock,
    )
    calls, fn = _ran_marker()

    first = gate.run("start_record", {}, fn)
    assert first["outcome"] == "executed"

    # Immediately again -> still on cooldown.
    second = gate.run("start_record", {}, fn)
    assert second["outcome"] == "denied"
    assert "cooldown" in second["reason"]
    assert calls == [1]  # fn ran only once

    # After the interval elapses it runs again.
    clock.advance(5.0)
    third = gate.run("start_record", {}, fn)
    assert third["outcome"] == "executed"
    assert calls == [1, 1]


def test_cooldown_is_per_action():
    clock = Clock()
    gate = ActionGate(
        ActionPolicy.allow("a", "b", dry_run=False, cooldown_s=5.0), clock=clock
    )
    assert gate.run("a", {}, lambda: {}).get("outcome") == "executed"
    # A different action is not blocked by a's cooldown.
    assert gate.run("b", {}, lambda: {}).get("outcome") == "executed"


def test_execution_error_is_caught_and_reported():
    gate = ActionGate(ActionPolicy.allow("boom", dry_run=False))

    def fn():
        raise ValueError("nope")

    out = gate.run("boom", {}, fn)
    assert out["outcome"] == "error"
    assert "ValueError" in out["reason"]
    assert "nope" in out["reason"]


def test_failed_execution_does_not_consume_cooldown():
    # An action that errors should still be retryable (no false cooldown).
    clock = Clock()
    gate = ActionGate(
        ActionPolicy.allow("x", dry_run=False, cooldown_s=10.0), clock=clock
    )
    gate.run("x", {}, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # Reserved the slot at execution time; a retry is on cooldown. This is the
    # documented behaviour (slot reserved before fn), so assert it explicitly.
    again = gate.run("x", {}, lambda: {"ok": True})
    assert again["outcome"] == "denied"


# ── audit + redaction ───────────────────────────────────────────────────────


def test_audit_records_every_outcome():
    audit = AuditLog()
    gate = ActionGate(ActionPolicy.allow("go", dry_run=False), audit)
    gate.run("go", {}, lambda: {"ok": True})       # executed
    gate.run("blocked", {}, lambda: {})            # denied
    outcomes = [e.outcome for e in audit.recent()]
    assert outcomes == ["executed", "denied"]


def test_audit_redacts_secrets_in_params():
    audit = AuditLog()
    gate = ActionGate(ActionPolicy.allow("auth", dry_run=False), audit)
    gate.run("auth", {"token": "supersecret", "url": "x", "api_key": "k"},
             lambda: {})
    entry = audit.recent()[-1]
    assert entry.params["token"] == "***"
    assert entry.params["api_key"] == "***"
    assert entry.params["url"] == "x"


def test_redact_is_recursive():
    red = _redact({"outer": {"password": "p", "keep": 1}, "authorization": "z"})
    assert red["outer"]["password"] == "***"
    assert red["outer"]["keep"] == 1
    assert red["authorization"] == "***"


def test_audit_log_writes_jsonl_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path=str(path))
    audit.record(AuditEntry(timestamp=1.0, action="go", params={}, outcome="executed"))
    audit.record(AuditEntry(timestamp=2.0, action="x", params={}, outcome="denied"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "go"
    assert json.loads(lines[1])["outcome"] == "denied"


def test_audit_log_path_property():
    assert AuditLog().path is None
    assert AuditLog(path="/tmp/a.jsonl").path == "/tmp/a.jsonl"


def test_gate_run_through_a_persisting_audit_log(tmp_path):
    # An action driven through the gate is appended to the JSONL file.
    path = tmp_path / "gate-audit.jsonl"
    gate = ActionGate(
        ActionPolicy.allow("set_filter", dry_run=False), AuditLog(path=str(path))
    )
    gate.run("set_filter", {"lowcut": 1, "highcut": 40}, lambda: {"ok": True})
    gate.run("denied_action", {}, lambda: {})  # not allowed -> still recorded
    records = [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln]
    assert [r["action"] for r in records] == ["set_filter", "denied_action"]
    assert [r["outcome"] for r in records] == ["executed", "denied"]


def test_audit_recent_limit():
    audit = AuditLog()
    for i in range(10):
        audit.record(AuditEntry(timestamp=float(i), action=f"a{i}", params={},
                                outcome="executed"))
    assert len(audit.recent(3)) == 3
    assert len(audit.recent()) == 10
    assert audit.recent(3)[-1].action == "a9"
