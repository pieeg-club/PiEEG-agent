"""Tiny JSON-over-HTTP helper shared by the wire adapters.

The adapters deliberately speak the providers' REST APIs directly with the
standard library instead of pulling a vendor SDK. The wire formats (Anthropic
Messages, OpenAI chat completions) are small and stable, and avoiding the SDKs
keeps the dependency surface tiny and the provider-agnostic boundary honest.

This module is intentionally minimal: one blocking POST, JSON in and out, with
errors normalised to :class:`LLMHTTPError` so callers needn't know about
``urllib`` internals.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Iterator


class LLMHTTPError(RuntimeError):
    """An HTTP-level failure talking to an LLM backend.

    ``status`` is the HTTP code when available (0 for transport errors) and
    ``body`` is the decoded response text, which usually carries the
    provider's structured error message.
    """

    def __init__(self, message: str, *, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    *,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """POST ``payload`` as JSON and return the decoded JSON response.

    Raises :class:`LLMHTTPError` on non-2xx responses or transport failures,
    surfacing the provider's error body so the caller can show something
    actionable (bad key, unknown model, rate limit, …).
    
    Rate limits (HTTP 429) are retried with exponential backoff up to
    ``max_retries`` times. Other errors fail immediately.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("content-type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            # Success - break out of retry loop
            break
        except urllib.error.HTTPError as exc:  # 4xx / 5xx with a body
            detail = _read_error_body(exc)
            last_error = LLMHTTPError(
                f"HTTP {exc.code} from {url}: {_short(detail)}",
                status=exc.code,
                body=detail,
            )
            # Retry only on 429 (rate limit) or 503 (service unavailable)
            if exc.code in (429, 503) and attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, ...
                delay = retry_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:  # DNS, refused, timeout, TLS, …
            raise LLMHTTPError(
                f"Could not reach {url}: {exc.reason}", status=0
            ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMHTTPError(
            f"Malformed JSON from {url}: {_short(raw)}", body=raw
        ) from exc


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")
    except Exception:  # pragma: no cover - body already consumed/unreadable
        return str(exc.reason)


def post_sse(
    url: str,
    payload: dict,
    headers: dict[str, str],
    *,
    timeout: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Iterator[dict]:
    """POST ``payload`` and yield each server-sent ``data:`` event as a dict.

    Both the Anthropic and OpenAI streaming wire formats are line-oriented SSE:
    one JSON object per ``data:`` line, comment lines starting with ``:``, and
    an optional ``[DONE]`` sentinel. We ignore ``event:`` lines and rely on the
    JSON payload itself (each adapter knows how to read its own chunks).

    Errors are normalised to :class:`LLMHTTPError`, matching :func:`post_json`,
    so streaming and blocking callers handle failures the same way.
    
    Rate limits (HTTP 429) are retried with exponential backoff up to
    ``max_retries`` times. Other errors fail immediately.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("content-type", "application/json")
    req.add_header("accept", "text/event-stream")
    for key, value in headers.items():
        req.add_header(key, value)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            # Success - break out of retry loop
            break
        except urllib.error.HTTPError as exc:  # 4xx / 5xx with a body
            detail = _read_error_body(exc)
            last_error = LLMHTTPError(
                f"HTTP {exc.code} from {url}: {_short(detail)}",
                status=exc.code,
                body=detail,
            )
            # Retry only on 429 (rate limit) or 503 (service unavailable)
            if exc.code in (429, 503) and attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, ...
                delay = retry_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:  # DNS, refused, timeout, TLS, …
            raise LLMHTTPError(
                f"Could not reach {url}: {exc.reason}", status=0
            ) from exc

    with resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:  # keep-alive / partial — skip
                continue


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"
