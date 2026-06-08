"""Synchronous WebSocket client for the PiEEG-server control plane.

PiEEG-server exposes its control plane on ``ws://<host>:1616``. The protocol
(verified against ``pieeg_server/server.py``) has two properties that shape
this client:

1. **Commands are flat JSON** — ``{"cmd": "<name>", ...params}``.
2. **Replies are *not* request-correlated.** The server broadcasts to every
   connected client, and the only discriminator is the *presence of a
   top-level key* in the message (``record_status``, ``reg_config``,
   ``osc_status``, …). Meanwhile it streams EEG **data frames**
   (``{"t", "n", "channels"}``) at 250–500 Hz.

So this client runs a background reader thread that drops data frames and
indexes every other message by its top-level keys with a monotonic sequence
number. :meth:`request` sends a command and blocks until a *newer* message
carrying the expected reply key arrives (or a timeout elapses). Commands with
no reply (``set_filter``) are sent fire-and-forget.

The client is deliberately synchronous (a worker thread, not asyncio) to match
the rest of the agent; it uses ``websockets.sync`` under the hood.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any
from urllib.parse import urlencode

logger = logging.getLogger("pieeg.agent.server")

# Top-level keys that identify an EEG data frame (dropped, never a reply).
_FRAME_KEYS = ("channels",)


class ServerControlError(RuntimeError):
    """A control-plane failure: cannot connect, send, or no reply in time."""


class ServerControlClient:
    """Talks to PiEEG-server's WebSocket control plane, synchronously.

    Typical use::

        with ServerControlClient("ws://localhost:1616") as c:
            info = c.welcome
            reply = c.request("start_record", reply_key="record_status")
    """

    def __init__(
        self,
        url: str = "ws://localhost:1616",
        *,
        token: str | None = None,
        open_timeout: float = 5.0,
        reply_timeout: float = 5.0,
    ):
        self._url = url
        self._token = token
        self._open_timeout = open_timeout
        self._reply_timeout = reply_timeout

        self._ws = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

        self._cond = threading.Condition()
        self._seq = 0
        self._inbox: dict[str, tuple[int, dict]] = {}
        self._send_lock = threading.Lock()

        self._welcome: dict | None = None
        self._frames_seen = 0
        self._messages_seen = 0

    # ── lifecycle ────────────────────────────────────────────────────────
    def connect(self) -> dict:
        """Open the connection, read the welcome message, start the reader.

        Returns the server's welcome dict. Raises :class:`ServerControlError`
        if the connection or handshake fails.
        """
        try:
            from websockets.sync.client import connect
        except Exception as exc:  # pragma: no cover - optional dep missing
            raise ServerControlError(
                "The 'websockets' package is required for server control "
                "(pip install \"pieeg-agent[server]\")."
            ) from exc

        url = self._url
        if self._token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode({'token': self._token})}"

        try:
            self._ws = connect(url, open_timeout=self._open_timeout)
        except Exception as exc:
            raise ServerControlError(
                f"Could not connect to PiEEG-server at {self._url}: {exc}"
            ) from exc

        # The server's first message is the welcome snapshot.
        try:
            raw = self._ws.recv(timeout=self._open_timeout)
        except Exception as exc:
            self._safe_close()
            raise ServerControlError(
                f"No welcome from {self._url}: {exc}"
            ) from exc
        self._welcome = _loads(raw)

        self._stop.clear()
        self._reader = threading.Thread(
            target=self._run, name="pieeg-server-rx", daemon=True
        )
        self._reader.start()
        return self._welcome or {}

    def close(self) -> None:
        self._stop.set()
        self._safe_close()
        if self._reader:
            self._reader.join(timeout=2.0)
            self._reader = None

    def __enter__(self) -> "ServerControlClient":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── properties ───────────────────────────────────────────────────────
    @property
    def welcome(self) -> dict:
        return dict(self._welcome or {})

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._stop.is_set()

    def stats(self) -> dict:
        return {
            "messages": self._messages_seen,
            "frames": self._frames_seen,
            "url": self._url,
        }

    # ── messaging ────────────────────────────────────────────────────────
    def send(self, cmd: str, **params: Any) -> None:
        """Send a command without waiting for any reply (fire-and-forget)."""
        if self._ws is None:
            raise ServerControlError("Not connected.")
        payload = {"cmd": cmd, **params}
        try:
            with self._send_lock:
                self._ws.send(json.dumps(payload))
        except Exception as exc:
            raise ServerControlError(f"Failed to send {cmd!r}: {exc}") from exc

    def request(
        self,
        cmd: str,
        *,
        reply_key: str,
        timeout: float | None = None,
        **params: Any,
    ) -> dict:
        """Send ``cmd`` and wait for a fresh message carrying ``reply_key``.

        Because replies are broadcast and not correlated, "fresh" means a
        message indexed *after* this call's send. Returns the full message
        dict. Raises :class:`ServerControlError` on timeout.
        """
        wait = self._reply_timeout if timeout is None else timeout
        with self._cond:
            since = self._seq
            self.send(cmd, **params)
            deadline_ok = self._cond.wait_for(
                lambda: self._inbox.get(reply_key, (0, None))[0] > since,
                timeout=wait,
            )
            if not deadline_ok:
                raise ServerControlError(
                    f"No {reply_key!r} reply to {cmd!r} within {wait:.1f}s."
                )
            return self._inbox[reply_key][1]

    # ── reader thread ────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:  # ConnectionClosed and friends
                break
            msg = _loads(raw)
            if msg is None:
                continue
            self._messages_seen += 1
            if _is_data_frame(msg):
                self._frames_seen += 1
                continue
            with self._cond:
                self._seq += 1
                for key in msg:
                    self._inbox[key] = (self._seq, msg)
                self._cond.notify_all()

    def _safe_close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # pragma: no cover - best-effort
                pass
            self._ws = None


def _loads(raw) -> dict | None:
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return msg if isinstance(msg, dict) else None


def _is_data_frame(msg: dict) -> bool:
    return any(k in msg for k in _FRAME_KEYS) and isinstance(
        msg.get("channels"), list
    )
