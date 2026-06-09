"""Pattern persistence — trained detectors that survive a restart.

A trained pattern (Phase B) is just data: a feature recipe, a small set of
linear weights, its cross-validated score and some metadata. This store keeps
each one as a single JSON file under ``~/.pieeg-agent/patterns`` (override with
``$PIEEG_PATTERN_DIR``), mirroring how the community browser experiences keep
models in ``localStorage`` — small, inspectable, easy to share.

The store is intentionally dumb: it serialises and deserialises dicts and lists
names. The meaning of the payload lives with the model in
:mod:`pieeg_agent.decode.classifier`.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_SAFE = re.compile(r"[^a-z0-9_-]+")


def default_pattern_dir() -> Path:
    """Where patterns live by default (``$PIEEG_PATTERN_DIR`` or under home)."""
    env = os.environ.get("PIEEG_PATTERN_DIR")
    if env:
        return Path(env)
    return Path.home() / ".pieeg-agent" / "patterns"


def default_session_dir() -> Path:
    """Where session recordings live (``$PIEEG_SESSION_DIR`` or under home)."""
    env = os.environ.get("PIEEG_SESSION_DIR")
    if env:
        return Path(env)
    return Path.home() / ".pieeg-agent" / "sessions"


def slugify(name: str) -> str:
    """Filesystem-safe slug for a human pattern name."""
    slug = _SAFE.sub("-", name.strip().lower()).strip("-")
    return slug or "pattern"


class PatternStore:
    """A directory of named pattern JSON files."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_pattern_dir()

    # ── paths ────────────────────────────────────────────────────────────
    def _path(self, name: str) -> Path:
        return self.root / f"{slugify(name)}.json"

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    # ── read ─────────────────────────────────────────────────────────────
    def load(self, name: str) -> dict | None:
        """Load a pattern payload by name, or ``None`` if it is not stored."""
        path = self._path(name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list(self) -> list[str]:
        """The slugs of all stored patterns, sorted."""
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def list_meta(self) -> list[dict]:
        """A compact summary per stored pattern for listing tools."""
        out: list[dict] = []
        for slug in self.list():
            payload = self.load(slug) or {}
            out.append(
                {
                    "name": payload.get("name", slug),
                    "slug": slug,
                    "labels": payload.get("labels", []),
                    "score": payload.get("score"),
                    "metric": payload.get("metric"),
                    "n_reps": payload.get("n_reps"),
                    "saved_at": payload.get("saved_at"),
                }
            )
        return out

    # ── write ────────────────────────────────────────────────────────────
    def save(self, name: str, payload: dict) -> Path:
        """Persist a pattern payload, stamping name/slug/time. Returns the path."""
        self.root.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        record.setdefault("name", name)
        record["slug"] = slugify(name)
        record["saved_at"] = time.time()
        path = self._path(name)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path

    def delete(self, name: str) -> bool:
        """Remove a stored pattern; returns ``True`` if a file was deleted."""
        path = self._path(name)
        if path.is_file():
            path.unlink()
            return True
        return False


class SessionStore:
    """A directory of named session-summary JSON files.

    Mirrors :class:`PatternStore` but holds the lab-notebook summaries produced
    by :class:`pieeg_agent.decode.session.SessionRecording` — small, inspectable
    records the agent can list, re-open and compare across a sitting.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_session_dir()

    def _path(self, name: str) -> Path:
        return self.root / f"{slugify(name)}.json"

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def load(self, name: str) -> dict | None:
        """Load a session summary by name, or ``None`` if it is not stored."""
        path = self._path(name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list(self) -> list[str]:
        """The slugs of all stored sessions, sorted."""
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def list_meta(self) -> list[dict]:
        """A compact summary per stored session for listing tools."""
        out: list[dict] = []
        for slug in self.list():
            payload = self.load(slug) or {}
            out.append(
                {
                    "name": payload.get("label", slug),
                    "slug": slug,
                    "duration_s": payload.get("duration_s"),
                    "n_frames": payload.get("n_frames"),
                    "dominant_band": payload.get("dominant_band"),
                    "saved_at": payload.get("saved_at"),
                }
            )
        return out

    def save(self, name: str, payload: dict) -> Path:
        """Persist a session summary, stamping label/slug/time. Returns the path."""
        self.root.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        record.setdefault("label", name)
        record["slug"] = slugify(name)
        record["saved_at"] = time.time()
        path = self._path(name)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path

    def delete(self, name: str) -> bool:
        """Remove a stored session; returns ``True`` if a file was deleted."""
        path = self._path(name)
        if path.is_file():
            path.unlink()
            return True
        return False
