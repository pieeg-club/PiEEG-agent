"""Graphical web UI layer for PiEEG-agent.

A ChatGPT/Gemini-style front-end over the *same* copilot, perception cascade
and pattern engine the CLI drives. :class:`WebEngine` is the brain-facing
façade (no web deps); :func:`create_app` builds the FastAPI transport and is
imported lazily so ``import pieeg_agent.web`` works without the ``web`` extra
installed.
"""

from __future__ import annotations

from .engine import WebEngine, event_to_dict

__all__ = ["WebEngine", "event_to_dict", "create_app"]


def create_app(*args, **kwargs):
    """Build the FastAPI app (see :func:`pieeg_agent.web.app.create_app`).

    Imported here lazily so the optional FastAPI dependency is only required
    when a server is actually started; a missing install yields one actionable
    error instead of an ImportError deep in a request.
    """
    try:
        from .app import create_app as _create_app
    except ImportError as exc:
        raise RuntimeError(
            "The web UI needs FastAPI/uvicorn. Install the extra:\n"
            "  pip install 'pieeg-agent[web]'"
        ) from exc

    return _create_app(*args, **kwargs)
