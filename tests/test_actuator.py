"""The actuator side: ServerActions facade, ActuatorTools, CombinedToolset.

A duck-typed ``FakeClient`` stands in for the WebSocket client so these run
without a socket. We check that reads bypass the gate, that mutations are
gated, that client-side validation (filter bounds, register range, presets)
rejects bad input, that the LLM-facing tools dispatch and coerce arguments,
and that a combined toolset routes calls to the right side.
"""

import pytest

from pieeg_agent.agent.actuator_tools import SAFE_ACTIONS, ActuatorTools
from pieeg_agent.agent.tools import CombinedToolset, NeuralTools
from pieeg_agent.server.actions import ServerActions
from pieeg_agent.server.gate import ActionGate, ActionPolicy
from tests.test_tools import FakeCascade, mk_state


class FakeClient:
    """Duck-types the bits of ServerControlClient that ServerActions uses."""

    def __init__(self):
        self.welcome = {"status": "connected", "channels": 16, "mock": True}
        self.requests: list[tuple] = []   # (cmd, reply_key, params)
        self.sent: list[tuple] = []       # (cmd, params)

    def request(self, cmd, *, reply_key, timeout=None, **params):
        self.requests.append((cmd, reply_key, params))
        return {reply_key: {"ok": True, "cmd": cmd}}

    def send(self, cmd, **params):
        self.sent.append((cmd, params))


def _exec_actions(*allowed):
    """A ServerActions wired to a FakeClient with an executing gate."""
    client = FakeClient()
    gate = ActionGate(
        ActionPolicy.allow(*allowed, dry_run=False, cooldown_s=0.0)
    )
    return ServerActions(client, gate), client


# ── reads are not gated ─────────────────────────────────────────────────────


def test_server_info_reads_welcome_without_gate():
    client = FakeClient()
    actions = ServerActions(client, ActionGate(ActionPolicy()))  # permits nothing
    assert actions.server_info()["channels"] == 16


def test_reads_issue_correct_reply_keys():
    client = FakeClient()
    actions = ServerActions(client, ActionGate(ActionPolicy()))
    actions.read_registers()
    actions.list_webhooks()
    actions.osc_status()
    actions.lsl_status()
    cmds = [(c, k) for (c, k, _p) in client.requests]
    assert ("reg_read", "reg_config") in cmds
    assert ("webhook_list", "webhook_rules") in cmds
    assert ("osc_status", "osc_status") in cmds
    assert ("lsl_status", "lsl_status") in cmds


# ── gated mutations ─────────────────────────────────────────────────────────


def test_set_filter_executes_as_fire_and_forget():
    actions, client = _exec_actions("set_filter")
    out = actions.set_filter(enabled=True, lowcut=2.0, highcut=30.0)
    assert out["outcome"] == "executed"
    # set_filter has no reply: it goes out via send(), not request().
    assert client.sent == [("set_filter",
                            {"enabled": True, "lowcut": 2.0, "highcut": 30.0})]
    assert client.requests == []


def test_set_filter_rejects_out_of_range_bounds():
    actions, client = _exec_actions("set_filter")
    out = actions.set_filter(enabled=True, lowcut=10.0, highcut=200.0)
    assert out["outcome"] == "error"
    assert "range" in out["reason"]
    assert client.sent == []  # nothing was sent to the device


def test_recording_roundtrips_record_status():
    actions, client = _exec_actions("start_record", "stop_record")
    assert actions.start_recording()["outcome"] == "executed"
    assert actions.stop_recording()["outcome"] == "executed"
    cmds = [(c, k) for (c, k, _p) in client.requests]
    assert cmds == [("start_record", "record_status"),
                    ("stop_record", "record_status")]


def test_register_preset_validates_name():
    actions, client = _exec_actions("reg_preset")
    good = actions.apply_register_preset("test_signal")
    assert good["outcome"] == "executed"
    bad = actions.apply_register_preset("bogus")
    assert bad["outcome"] == "error"
    assert "preset" in bad["reason"]


def test_write_registers_enforces_chnset_range():
    actions, client = _exec_actions("reg_write")
    ok = actions.write_registers({"0x05": "0x00", "0x0C": "0x05"})
    assert ok["outcome"] == "executed"
    # 0x0D is outside the allowed CHnSET range (0x05-0x0C).
    bad = actions.write_registers({"0x0D": "0x00"})
    assert bad["outcome"] == "error"
    assert "not allowed" in bad["reason"]


def test_osc_passes_config_through():
    actions, client = _exec_actions("osc_start")
    actions.start_osc({"host": "127.0.0.1", "port": 9001})
    cmd, key, params = client.requests[-1]
    assert cmd == "osc_start" and key == "osc_status"
    assert params["config"] == {"host": "127.0.0.1", "port": 9001}


def test_dry_run_policy_blocks_real_send():
    client = FakeClient()
    actions = ServerActions(
        client, ActionGate(ActionPolicy.allow("set_filter", dry_run=True))
    )
    out = actions.set_filter(enabled=True)
    assert out["outcome"] == "dry_run"
    assert client.sent == []  # never reached the device


# ── ActuatorTools (LLM-facing dispatch) ─────────────────────────────────────


def _tools(*allowed):
    actions, client = _exec_actions(*allowed)
    return ActuatorTools(actions), client


def test_actuator_tool_names_and_specs():
    tools, _ = _tools()
    names = tools.names()
    assert "server_status" in names
    assert "set_filter" in names
    assert "apply_register_preset" in names
    # Raw register writes are deliberately NOT exposed to the model.
    assert "write_registers" not in names
    assert "reg_write" not in names
    # Specs are advertised for every tool.
    assert len(tools.specs()) == len(names)


def test_actuator_server_status_is_a_read():
    tools, _ = _tools()
    out = tools.call("server_status")
    assert out["channels"] == 16


def test_actuator_set_filter_dispatch():
    tools, client = _tools("set_filter")
    out = tools.call("set_filter", {"enabled": True, "lowcut": 1, "highcut": 40})
    assert out["outcome"] == "executed"
    assert client.sent[0][0] == "set_filter"


def test_actuator_preset_dispatch():
    tools, client = _tools("reg_preset")
    out = tools.call("apply_register_preset", {"preset": "normal"})
    assert out["outcome"] == "executed"
    assert client.requests[-1][0] == "reg_preset"


def test_actuator_start_osc_assembles_config_from_flat_args():
    tools, client = _tools("osc_start")
    tools.call("start_osc", {"host": "10.0.0.5", "port": 9000, "channel": 3})
    params = client.requests[-1][2]
    assert params["config"] == {"host": "10.0.0.5", "port": 9000, "channel": 3}


def test_actuator_unknown_tool_returns_error_not_raise():
    tools, _ = _tools()
    out = tools.call("nope")
    assert "error" in out and "available" in out


def test_safe_actions_excludes_raw_register_write():
    assert "reg_write" not in SAFE_ACTIONS
    assert "set_filter" in SAFE_ACTIONS
    assert "reg_preset" in SAFE_ACTIONS


# ── CombinedToolset routing ─────────────────────────────────────────────────


def test_combined_toolset_merges_and_routes():
    senses = NeuralTools(FakeCascade(state=mk_state()))
    actuator, client = _tools("set_filter")
    combined = CombinedToolset(senses, actuator)

    # Union of both name sets, specs concatenated.
    assert "get_neural_state" in combined.names()
    assert "set_filter" in combined.names()
    assert len(combined.specs()) == len(senses.specs()) + len(actuator.specs())

    # A sense call routes to NeuralTools.
    state = combined.call("get_neural_state")
    assert "focus" in state
    # An action call routes to ActuatorTools.
    act = combined.call("set_filter", {"enabled": True})
    assert act["outcome"] == "executed"


def test_combined_toolset_unknown_tool():
    senses = NeuralTools(FakeCascade(state=mk_state()))
    actuator, _ = _tools()
    combined = CombinedToolset(senses, actuator)
    out = combined.call("does_not_exist")
    assert "error" in out
