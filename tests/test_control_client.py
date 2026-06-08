"""The WebSocket control client against a loopback fake PiEEG-server.

Rather than mock the transport, we stand up a tiny real ``websockets.sync``
server on an ephemeral port that mimics the two protocol quirks that matter:
it sends a welcome on connect, streams **data frames** the client must drop,
and **broadcasts replies keyed by a top-level field** (not request-correlated).
This exercises the reader thread, the seq-based demux and the timeout path for
real.
"""

import json
import threading

import pytest

pytest.importorskip("websockets")

from websockets.sync.server import serve  # noqa: E402

from pieeg_agent.server.client import (  # noqa: E402
    ServerControlClient,
    ServerControlError,
    _is_data_frame,
    _loads,
)

WELCOME = {"status": "connected", "sample_rate": 250, "channels": 16, "mock": True}


class FakeServer:
    """A loopback control plane: welcome, a data frame per command, scripted replies."""

    def __init__(self, scripts: dict):
        # scripts: cmd -> reply dict (indexed) | None (fire-and-forget, no reply)
        #          a cmd absent from scripts gets NO reply (timeout case).
        self._scripts = scripts
        self.received: list[dict] = []
        self._server = serve(self._handler, "localhost", 0)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def url(self) -> str:
        return f"ws://localhost:{self.port}"

    def _handler(self, ws) -> None:
        ws.send(json.dumps(WELCOME))
        for raw in ws:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            self.received.append(msg)
            cmd = msg.get("cmd")
            # Always stream a data frame first — the client must drop it and
            # must NOT treat it as a reply.
            ws.send(json.dumps({"t": 0.0, "n": 1, "channels": [1.0, 2.0, 3.0]}))
            if cmd in self._scripts:
                reply = self._scripts[cmd]
                if reply is not None:
                    ws.send(json.dumps(reply))

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture
def server():
    srv = FakeServer({
        "start_record": {"record_status": {"recording": True}},
        "stop_record": {"record_status": {"recording": False}},
        "reg_read": {"reg_config": {"status": "ok", "regs": {}}},
        "webhook_list": {"webhook_rules": []},
        "two_keys": {"alpha": 1, "beta": 2},
        "set_filter": None,  # fire-and-forget, no reply
        # "silent" intentionally absent -> never replies (timeout case)
    })
    yield srv
    srv.stop()


# ── pure helpers ───────────────────────────────────────────────────────────


def test_is_data_frame_detection():
    assert _is_data_frame({"t": 0.0, "n": 1, "channels": [1.0, 2.0]})
    assert not _is_data_frame({"record_status": {"recording": True}})
    assert not _is_data_frame({"channels": "not-a-list"})


def test_loads_tolerates_garbage():
    assert _loads("not json") is None
    assert _loads(json.dumps({"a": 1})) == {"a": 1}


# ── connect + welcome ───────────────────────────────────────────────────────


def test_connect_reads_welcome(server):
    with ServerControlClient(server.url) as client:
        assert client.connected
        assert client.welcome["sample_rate"] == 250
        assert client.welcome["mock"] is True


# ── request / reply demux ───────────────────────────────────────────────────


def test_request_returns_keyed_reply(server):
    with ServerControlClient(server.url) as client:
        reply = client.request("start_record", reply_key="record_status")
        assert reply["record_status"]["recording"] is True


def test_request_indexes_all_top_level_keys(server):
    with ServerControlClient(server.url) as client:
        # The reply carries two keys; either can be awaited.
        reply = client.request("two_keys", reply_key="beta")
        assert reply["alpha"] == 1 and reply["beta"] == 2


def test_data_frames_are_dropped_not_returned(server):
    with ServerControlClient(server.url, reply_timeout=1.0) as client:
        client.request("reg_read", reply_key="reg_config")
        stats = client.stats()
        # At least one data frame was seen and dropped during the exchange.
        assert stats["frames"] >= 1


def test_request_times_out_when_no_reply(server):
    with ServerControlClient(server.url, reply_timeout=0.4) as client:
        with pytest.raises(ServerControlError):
            client.request("silent", reply_key="never_comes", timeout=0.4)


def test_fire_and_forget_send_does_not_raise(server):
    with ServerControlClient(server.url) as client:
        client.send("set_filter", enabled=True, lowcut=1.0, highcut=40.0)
        # Give the server a moment to record it, via a round-trip on another cmd.
        client.request("reg_read", reply_key="reg_config")
        cmds = [m.get("cmd") for m in server.received]
        assert "set_filter" in cmds


def test_request_before_connect_raises():
    client = ServerControlClient("ws://localhost:1")  # not connected
    with pytest.raises(ServerControlError):
        client.send("start_record")
