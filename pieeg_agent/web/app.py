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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .engine import WebEngine
from ..llm._http import LLMHTTPError

# Sentinel marking a drained sync generator pulled through the threadpool.
_SENTINEL = object()


def _safe_next(it) -> object:
    try:
        return next(it)
    except StopIteration:
        return _SENTINEL


def _format_llm_error(exc: Exception) -> str:
    """Format an LLM error into a user-friendly message."""
    if isinstance(exc, LLMHTTPError):
        if exc.status == 429:
            return (
                "Rate limit exceeded. The API is receiving too many requests. "
                "Please wait a moment and try again."
            )
        elif exc.status == 401:
            return "Authentication failed. Please check your API key."
        elif exc.status == 400:
            return f"Bad request: {exc}"
        elif exc.status >= 500:
            return "The LLM service is experiencing issues. Please try again later."
        else:
            return f"LLM error: {exc}"
    return f"Unexpected error: {exc}"


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

    # ── Notebooks ─────────────────────────────────────────────────────────
    @app.get("/api/notebook")
    async def get_notebook(path: str):
        """Read a Jupyter notebook structure and outputs."""
        result = await run_in_threadpool(engine.read_notebook, path)
        status = 404 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    @app.get("/api/notebooks")
    async def list_notebooks(path: str = ".", recursive: bool = False):
        """List all Jupyter notebooks in a directory."""
        result = await run_in_threadpool(engine.list_notebooks, path, recursive)
        status = 404 if "error" in result else 200
        return JSONResponse(result, status_code=status)

    # ── LSL streams discovery ─────────────────────────────────────────────
    @app.get("/api/streams")
    async def streams(wait: float = 2.0) -> dict:
        """Discover LSL streams on the network."""
        return await run_in_threadpool(engine.list_streams, wait)

    # ── server control ────────────────────────────────────────────────────
    @app.get("/api/server/status")
    async def server_status() -> dict:
        """Get PiEEG-server status if connected."""
        return await run_in_threadpool(engine.server_status)

    @app.post("/api/server/filter")
    async def server_filter(request: Request) -> dict:
        """Set server-side band-pass filter."""
        data = await request.json()
        return await run_in_threadpool(
            engine.server_filter,
            data.get("enabled", True),
            data.get("lowcut", 1.0),
            data.get("highcut", 40.0),
        )

    @app.post("/api/server/record")
    async def server_record(request: Request) -> dict:
        """Start or stop server-side recording."""
        data = await request.json()
        action = data.get("action", "start")
        return await run_in_threadpool(engine.server_record, action)

    @app.post("/api/server/osc")
    async def server_osc(request: Request) -> dict:
        """Start or stop OSC output."""
        data = await request.json()
        action = data.get("action", "start")
        config = data.get("config")
        return await run_in_threadpool(engine.server_osc, action, config)

    @app.post("/api/server/lsl")
    async def server_lsl(request: Request) -> dict:
        """Start or stop LSL output."""
        data = await request.json()
        action = data.get("action", "start")
        config = data.get("config")
        return await run_in_threadpool(engine.server_lsl, action, config)

    @app.post("/api/server/register-preset")
    async def server_register_preset(request: Request) -> dict:
        """Apply an ADS1299 register preset."""
        data = await request.json()
        preset = data.get("preset", "normal")
        return await run_in_threadpool(engine.server_register_preset, preset)

    @app.get("/api/server/webhooks")
    async def server_webhooks() -> dict:
        """List server webhook rules."""
        return await run_in_threadpool(engine.server_webhooks)

    @app.get("/api/llm/models")
    async def llm_models() -> dict:
        """The committed OpenRouter model catalog (snapshot, always current).

        Replaces the old hard-coded model lists: the front-end builds its model
        picker from this, so deprecated IDs never linger in the UI. Refreshed
        by the pre-commit hook via ``scripts/update_models.py``.
        """
        from ..llm.catalog import load_catalog

        return load_catalog()

    @app.post("/api/llm/config")
    async def update_llm_config(request: Request) -> dict:
        """Update LLM configuration (runtime only, not persisted)."""
        # This endpoint allows runtime configuration changes.
        # For permanent config, users should set environment variables.
        data = await request.json()
        provider = data.get("provider")
        model = data.get("model")
        api_key = data.get("api_key")
        
        # Validation: provider is required
        if not provider:
            return JSONResponse(
                {"detail": "provider is required"}, 
                status_code=400
            )
        
        # Note: This is a runtime-only change. For production use,
        # implement persistent configuration storage or use env vars.
        return {
            "status": "configuration received",
            "note": "Runtime config not yet implemented. Set PIEEG_LLM_PROVIDER, PIEEG_LLM_MODEL, and API keys via environment variables and restart.",
            "provider": provider,
            "model": model or "default",
        }

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
                try:
                    async for event in _aiter_in_threadpool(engine.chat_stream(text)):
                        await ws.send_json(event)
                except Exception as exc:
                    # Handle LLM errors gracefully without crashing the websocket
                    error_msg = _format_llm_error(exc)
                    await ws.send_json({
                        "type": "error",
                        "detail": error_msg
                    })
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
