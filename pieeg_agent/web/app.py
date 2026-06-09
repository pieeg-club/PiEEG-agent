"""FastAPI transport over :class:`~pieeg_agent.web.engine.WebEngine`.

A deliberately dumb pipe: REST for one-shot reads, three WebSockets for the
live surfaces (brain telemetry, streaming chat, guided pattern training), and
static hosting for the built React front-end. All brain logic lives in the
engine; this module only marshals JSON and keeps blocking work off the event
loop.

FastAPI is an optional dependency (the ``web`` extra). It is imported lazily by
:func:`create_app` so ``import pieeg_agent.web`` stays cheap and SDK-free; a
missing install yields one actionable error instead of an ImportError deep in a
request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import anyio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .engine import WebEngine

# Sentinel marking a drained sync generator pulled through the threadpool.
_SENTINEL = object()


def _safe_next(it):
    try:
        return next(it)
    except StopIteration:
        return _SENTINEL


def create_app(
    engine: WebEngine,
    *,
    static_dir: str | Path | None = None,
    live_interval: float = 0.25,
):
    """Build the FastAPI application bound to ``engine``.

    ``static_dir`` (the built front-end) is mounted at ``/`` when present;
    ``live_interval`` is the cadence of the ``/ws/live`` telemetry push.
    """
    app = FastAPI(title="PiEEG Agent", version="1.0")

    # Local single-user tool: the Vite dev server (port 5173) talks to this
    # backend cross-origin. No cookies/credentials are used, so allow any
    # origin rather than hard-coding dev ports.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _aiter_in_threadpool(gen: Iterator[dict]):
        """Pump a blocking sync generator from a worker thread.

        Each ``next`` runs off the event loop, so a slow LLM token stream never
        stalls other connections. The generator is always closed, releasing any
        lock it holds even if the client disconnects mid-stream.
        """
        it = iter(gen)
        try:
            while True:
                item = await run_in_threadpool(_safe_next, it)
                if item is _SENTINEL:
                    break
                yield item
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()

    # ── REST: one-shot reads ─────────────────────────────────────────────
    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/api/info")
    async def info() -> dict:
        return engine.info()

    @app.get("/api/state")
    async def state() -> dict:
        return await run_in_threadpool(engine.snapshot)

    @app.get("/api/patterns")
    async def patterns() -> dict:
        return await run_in_threadpool(engine.list_patterns)

    @app.get("/api/patterns/{name}")
    async def explain(name: str):
        result = await run_in_threadpool(engine.explain_pattern, name)
        status = 404 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    @app.delete("/api/patterns/{name}")
    async def forget(name: str):
        result = await run_in_threadpool(engine.forget_pattern, name)
        status = 404 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    # ── WS: live brain telemetry (server-push) ───────────────────────────
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                snap = await run_in_threadpool(engine.snapshot)
                await ws.send_json(snap)
                await anyio.sleep(live_interval)
        except WebSocketDisconnect:
            return

    # ── WS: streaming chat ───────────────────────────────────────────────
    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("reset"):
                    await run_in_threadpool(engine.reset_chat)
                    await ws.send_json({"type": "reset"})
                    continue
                text = (msg.get("message") or "").strip()
                if not text:
                    await ws.send_json(
                        {"type": "error", "detail": "empty message"}
                    )
                    continue
                async for event in _aiter_in_threadpool(engine.chat_stream(text)):
                    await ws.send_json(event)
        except WebSocketDisconnect:
            return

    # ── WS: guided pattern training ──────────────────────────────────────
    @app.websocket("/ws/train")
    async def ws_train(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                cmd = await ws.receive_json()
                action = cmd.get("action")
                if action == "begin":
                    result = await run_in_threadpool(
                        engine.train_begin, str(cmd.get("name") or "")
                    )
                elif action == "record":
                    result = await run_in_threadpool(
                        engine.train_record,
                        str(cmd.get("label") or ""),
                        float(cmd.get("seconds") or 4.0),
                    )
                elif action == "finish":
                    result = await run_in_threadpool(
                        engine.train_finish, float(cmd.get("threshold") or 0.6)
                    )
                elif action == "cancel":
                    result = await run_in_threadpool(engine.train_cancel)
                else:
                    result = {"error": f"unknown action {action!r}"}
                await ws.send_json({"action": action, "result": result})
        except WebSocketDisconnect:
            return

    # ── static front-end (mounted last so it can't shadow the API) ───────
    static_path = Path(static_dir) if static_dir else None
    if static_path and static_path.is_dir():
        app.mount(
            "/", StaticFiles(directory=str(static_path), html=True), name="static"
        )
    else:
        @app.get("/")
        async def root() -> dict:
            return {
                "service": "pieeg-agent",
                "ui": "not built",
                "hint": "Build the front-end (web/ → npm run build) or run the "
                "Vite dev server; the API is live at /api and /ws.",
            }

    return app
