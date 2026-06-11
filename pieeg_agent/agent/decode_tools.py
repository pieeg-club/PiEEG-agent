"""Pattern-recognition tools — the agent's higher-order senses.

These sit alongside the always-on :class:`~pieeg_agent.agent.tools.NeuralTools`
and expose the :mod:`pieeg_agent.decode` capabilities to the model:

* ``find_artifacts``    — recent blinks / jaw clenches / motion transients.
* ``analyze_spectrum``  — IAF, 1/f slope, theta/beta, entropy, alpha asymmetry.
* ``start_pattern_training`` / ``record_segment`` / ``finish_pattern_training``
  / ``cancel_pattern_training`` — teach a new detector by example (rest vs the
  thing you want recognised), a few reps, then fit with an honest score.
* ``list_patterns`` / ``detect_patterns`` / ``explain_pattern`` /
  ``forget_pattern`` — inspect, watch live, justify and delete trained patterns.
* ``connectivity`` — cross-channel amplitude coupling for a band, right now.
* ``record_session`` / ``list_sessions`` / ``analyze_session`` /
  ``compare_sessions`` / ``forget_session`` — the agent's lab notebook: capture
  a labelled window, re-open its summary, and contrast two with Cohen's d.

Unlike :class:`NeuralTools`, this set *does* hold state: it owns the live
:class:`~pieeg_agent.decode.patterns.PatternBank` (scored every frame) and the
in-progress training session. It taps the cascade through :meth:`on_frame`,
extracting the shared feature vector once per frame and fanning it out to both
the live bank and any open training segment. Reads stay read-only; the only
mutations are deliberate (teaching or forgetting a pattern), and persistence
goes through the :class:`~pieeg_agent.decode.store.PatternStore`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np

from ..decode import (
    ACTIVE,
    REST,
    PatternBank,
    PatternStore,
    PatternTrainer,
    SessionRecorder,
    SessionStore,
    TrainingError,
    analyze_spectrum,
    band_power_connectivity,
    compare_summaries,
    extract_frame,
    layout_for,
)
from ..perceive.cascade import PerceptionCascade
from .tools import Tool, _spec

logger = logging.getLogger("pieeg.agent.decode_tools")

# A few seconds is long enough to gather frames, short enough not to bore the
# user mid-conversation; clamp record requests into a sane band.
_MIN_SEGMENT_S = 1.0
_MAX_SEGMENT_S = 20.0
_MAX_REPS = 8            # prevent runaway training loops
_MIN_REPS = 2            # minimum required for meaningful CV

# A labelled session is a longer lab window; allow up to a couple of minutes.
_MIN_SESSION_S = 2.0
_MAX_SESSION_S = 120.0

# Rolling band-power history for on-demand connectivity. on_frame fires at the
# cascade's feature rate (~8 Hz), so ~32 s of context is plenty for a stable
# correlation without holding much memory.
_HISTORY_FRAMES = 256



class DecodeTools:
    """Pattern-recognition tools and the live pattern engine for one cascade."""

    def __init__(
        self,
        cascade: PerceptionCascade,
        *,
        store: PatternStore | None = None,
        session_store: SessionStore | None = None,
        on_detection=None,
        sleep=time.sleep,
    ):
        self._cascade = cascade
        self._store = store or PatternStore()
        self._sessions = session_store or SessionStore()
        self._bank = PatternBank(on_detection=on_detection)
        self._sleep = sleep

        self._layout = None
        self._latest_features = None
        self._latest_ts = 0.0
        self._trainer: PatternTrainer | None = None
        self._recorder = SessionRecorder()
        self._power_hist: deque = deque(maxlen=_HISTORY_FRAMES)
        self._lock = threading.Lock()

        self._tools: dict[str, Tool] = {}
        self._bank.load_all(self._store)   # restore previously taught patterns
        self._register_all()

    # ── cascade tap (register as the cascade's on_frame) ─────────────────
    def on_frame(self, bp, quality, data, ts) -> None:
        """Extract features once per frame; feed the live bank and any capture."""
        if self._layout is None:
            self._layout = layout_for(self._cascade.channel_labels(), bp.n_channels)
        try:
            feats = extract_frame(bp.per_channel, data)
        except Exception:  # pragma: no cover - a bad frame must not kill the cascade
            return
        # latest_state() is only needed while a session is recording; fetch it
        # outside the lock to keep the hot path short.
        state = self._cascade.latest_state() if self._recorder.is_open else None
        with self._lock:
            self._latest_features = feats
            self._latest_ts = bp.timestamp
            self._power_hist.append(
                (bp.timestamp, np.asarray(bp.per_channel, dtype=np.float64).copy())
            )
            if self._trainer is not None:
                self._trainer.add(feats)
            if self._recorder.is_open:
                self._recorder.add(bp, quality, state)
        self._bank.score_features(feats, bp.timestamp)

    @property
    def bank(self) -> PatternBank:
        return self._bank


    # ── registry surface (mirrors NeuralTools) ──────────────────────────
    def specs(self):
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "available": self.names()}
        try:
            return tool.handler(arguments or {})
        except Exception as exc:  # a tool must never kill the loop
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── registration ────────────────────────────────────────────────────
    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_all(self) -> None:
        self._add(Tool(
            _spec(
                "find_artifacts",
                "Recent time-domain transients detected in the EEG: eye blinks "
                "(including double-blinks), jaw clenches / muscle bursts, and "
                "gross motion. Each has a timestamp, duration and 0..1 "
                "confidence. Use this for 'did I blink', 'am I clenching', or "
                "to explain a sudden quality drop.",
                {
                    "limit": {
                        "type": "integer",
                        "description": "Max transients to return (default 10).",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional filter: blink, blink_double, "
                        "jaw_clench or motion.",
                    },
                },
            ),
            self._find_artifacts,
        ))
        self._add(Tool(
            _spec(
                "analyze_spectrum",
                "Deep spectral analysis of the current signal: individual "
                "alpha-peak frequency (IAF), aperiodic 1/f slope, theta/beta "
                "ratio, spectral entropy and (across channels) frontal alpha "
                "asymmetry. Pass a channel index for one electrode; omit it for "
                "the channel average.",
                {
                    "channel": {
                        "type": "integer",
                        "description": "Channel index for a single electrode "
                        "(omit for the channel average).",
                        "minimum": 0,
                    }
                },
            ),
            self._analyze_spectrum,
        ))
        self._add(Tool(
            _spec(
                "start_pattern_training",
                "Begin teaching a new pattern by example. Give it a name (e.g. "
                "'eyes-closed', 'mental-math', 'left-hand'). After this, guide "
                "the user through a few reps: record a 'rest' segment, then an "
                "'active' segment where they do the thing — repeat 3-5 times — "
                "then call finish_pattern_training. Starting a session replaces "
                "any unfinished one.",
                {"name": {"type": "string", "description": "Pattern name."}},
                required=["name"],
            ),
            self._start_training,
        ))
        self._add(Tool(
            _spec(
                "record_segment",
                "Record ONE labelled segment of the current training session "
                "from the live signal. label='rest' for the baseline, "
                "label='active' while the user performs the pattern. Blocks for "
                "'seconds' (default 4) while it gathers frames. Tell the user "
                "what to do and wait for them to confirm they are ready BEFORE "
                "calling this — it samples the live signal immediately. Each "
                "active segment is one rep. Call this AT MOST ONCE PER TURN: a "
                "second call in the same turn is refused so the user has time to "
                "switch states. Always read the 'action' field in the response — "
                "it tells you whether to keep going or to call "
                "finish_pattern_training now.",
                {
                    "label": {
                        "type": "string",
                        "enum": [REST, ACTIVE],
                        "description": "'rest' (baseline) or 'active' (doing it).",
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Seconds to record (1-20, default 4).",
                        "minimum": _MIN_SEGMENT_S,
                        "maximum": _MAX_SEGMENT_S,
                    },
                },
                required=["label"],
            ),
            self._record_segment,
        ))
        self._add(Tool(
            _spec(
                "finish_pattern_training",
                "Fit the detector from the recorded segments, cross-validate it "
                "(leave-one-rep-out balanced accuracy), save it and start "
                "scoring it live. Returns the honest score, which channels carry "
                "it and the top separating features. Needs at least 2 rest and 2 "
                "active segments.",
                {
                    "threshold": {
                        "type": "number",
                        "description": "Activation threshold 0..1 (default 0.6).",
                        "minimum": 0.05,
                        "maximum": 0.95,
                    }
                },
            ),
            self._finish_training,
        ))
        self._add(Tool(
            _spec(
                "cancel_pattern_training",
                "Discard the in-progress training session without saving.",
            ),
            self._cancel_training,
        ))
        self._add(Tool(
            _spec(
                "list_patterns",
                "List the trained patterns the agent knows, with their "
                "cross-validated balanced accuracy and whether they are loaded "
                "live. Use this for 'what can you recognise'.",
            ),
            self._list_patterns,
        ))
        self._add(Tool(
            _spec(
                "detect_patterns",
                "Current live activation (0..1 smoothed probability and on/off) "
                "of every loaded pattern, right now. Use this for 'what am I "
                "doing' or 'is the eyes-closed pattern firing'.",
            ),
            self._detect_patterns,
        ))
        self._add(Tool(
            _spec(
                "explain_pattern",
                "Explain a trained pattern: its cross-validated score, the "
                "per-channel importance the detector relies on, and the top "
                "rest-vs-active features that separate it. Use this to justify "
                "or debug a pattern.",
                {"name": {"type": "string", "description": "Pattern name."}},
                required=["name"],
            ),
            self._explain_pattern,
        ))
        self._add(Tool(
            _spec(
                "forget_pattern",
                "Delete a trained pattern from the live bank and from disk. "
                "This cannot be undone.",
                {"name": {"type": "string", "description": "Pattern name."}},
                required=["name"],
            ),
            self._forget_pattern,
        ))

        # ── connectivity ────────────────────────────────────────────────
        self._add(Tool(
            _spec(
                "connectivity",
                "Cross-channel functional connectivity right now: how strongly "
                "the electrodes' band-power envelopes move together (amplitude "
                "coupling, Pearson r over a short window). Returns mean "
                "connectivity, the strongest channel pairs, a per-channel "
                "coupling score and the full matrix. This is amplitude coupling, "
                "not phase coherence, and within-session only.",
                {
                    "band": {
                        "type": "string",
                        "description": "Band to couple on: Delta, Theta, Alpha, "
                        "Beta or Gamma (default Alpha).",
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Seconds of recent signal to use (2-30, "
                        "default 8).",
                        "minimum": 2.0,
                        "maximum": 30.0,
                    },
                },
            ),
            self._connectivity,
        ))

        # ── sessions (the lab notebook) ─────────────────────────────────
        self._add(Tool(
            _spec(
                "record_session",
                "Record a labelled session window from the live signal and save "
                "a summary (band means/spread, focus/relax/engagement, signal "
                "quality, artifacts and connectivity). Give it a label like "
                "'eyes-closed-rest' or 'mental-math'. Tell the user what to do "
                "BEFORE calling. Blocks for 'seconds' (default 20) while it "
                "gathers frames.",
                {
                    "label": {
                        "type": "string",
                        "description": "Human label for this session window.",
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Seconds to record (2-120, default 20).",
                        "minimum": _MIN_SESSION_S,
                        "maximum": _MAX_SESSION_S,
                    },
                },
                required=["label"],
            ),
            self._record_session,
        ))
        self._add(Tool(
            _spec(
                "list_sessions",
                "List the saved session recordings with their duration, frame "
                "count and dominant band. Use this for 'what sessions have I "
                "recorded' before analysing or comparing.",
            ),
            self._list_sessions,
        ))
        self._add(Tool(
            _spec(
                "analyze_session",
                "Re-open a saved session's full summary by label: per-band "
                "mean/spread, focus/relax/engagement, signal quality, artifact "
                "counts, per-channel band powers and connectivity.",
                {"label": {"type": "string", "description": "Session label."}},
                required=["label"],
            ),
            self._analyze_session,
        ))
        self._add(Tool(
            _spec(
                "compare_sessions",
                "Contrast two saved sessions with a within-session Cohen's d per "
                "feature (band powers, focus/relax/engagement, signal quality), "
                "ranked by effect size. Use this for 'what changed between my "
                "rest and my focus block'.",
                {
                    "a": {"type": "string", "description": "First session label."},
                    "b": {"type": "string", "description": "Second session label."},
                },
                required=["a", "b"],
            ),
            self._compare_sessions,
        ))
        self._add(Tool(
            _spec(
                "forget_session",
                "Delete a saved session recording from disk. Cannot be undone.",
                {"label": {"type": "string", "description": "Session label."}},
                required=["label"],
            ),
            self._forget_session,
        ))

    # ── handlers ─────────────────────────────────────────────────────────
    def _find_artifacts(self, args: dict) -> dict:
        limit = _as_int(args.get("limit"), 10, lo=1, hi=100)
        want = args.get("type")
        events = self._cascade.recent_artifacts(limit if not want else 100)
        rows = [e.to_dict() for e in events]
        if want:
            rows = [r for r in rows if r["type"] == want]
        rows = rows[-limit:]
        return {"count": len(rows), "artifacts": rows}

    def _analyze_spectrum(self, args: dict) -> dict:
        bp = self._cascade.latest_band_powers()
        if bp is None:
            return {
                "status": "no_data",
                "detail": "No spectrum yet — the cascade is still filling its "
                "first analysis window.",
            }
        channel = args.get("channel")
        ch = int(channel) if isinstance(channel, (int, float)) else None
        return analyze_spectrum(bp, self._cascade.channel_labels(), channel=ch)

    # ── training handlers ────────────────────────────────────────────────
    def _start_training(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip()
        if not name:
            return {"error": "a pattern name is required"}
        if self._layout is None:
            return {
                "status": "no_signal",
                "detail": "No frames yet — start the stream/monitor so the "
                "agent can see the signal before training.",
            }
        with self._lock:
            self._trainer = PatternTrainer(name, self._layout)
        return {
            "status": "training_started",
            "name": name,
            "next": "Tell the user to relax, then call record_segment "
            "label='rest'; then record_segment label='active' while they do "
            "the pattern. Repeat 3-5 times, then finish_pattern_training.",
        }

    def _record_segment(self, args: dict) -> dict:
        with self._lock:
            trainer = self._trainer
        if trainer is None:
            return {"error": "no training session — call start_pattern_training first"}
        
        # Check if we've hit the max reps limit
        counts_before = trainer.counts()
        if counts_before["reps"] >= _MAX_REPS:
            return {
                "error": f"Maximum {_MAX_REPS} reps reached. Call finish_pattern_training now.",
                "totals": counts_before,
                "ready": True,
                "must_finish": True,
            }
        
        label = str(args.get("label") or "").strip().lower()
        if label not in (REST, ACTIVE):
            return {"error": f"label must be {REST!r} or {ACTIVE!r}"}
        seconds = _as_float(args.get("seconds"), 4.0, lo=_MIN_SEGMENT_S, hi=_MAX_SEGMENT_S)

        with self._lock:
            trainer.open_segment(label)
        before = trainer.counts()
        
        # Log recording start for user feedback
        logger.info(f"Recording {label} segment for {seconds}s (rep {before['reps'] + 1})...")
        
        self._sleep(seconds)            # frames stream in via on_frame meanwhile
        
        with self._lock:
            captured = trainer.close_segment()
        counts = trainer.counts()
        
        logger.info(f"Recorded {captured} frames, totals: {counts['rest']} rest, {counts['active']} active ({counts['reps']} reps)")
        
        result = {
            "status": "segment_recorded",
            "label": label,
            "captured_frames": captured,
            "totals": counts,
            "ready": trainer.ready,
        }
        
        if captured == 0:
            result["warning"] = (
                "No frames captured — is the stream running? Each segment needs "
                "the live cascade feeding frames."
            )
            return result
        
        # Strong completion signals to guide the LLM
        reps = counts["reps"]
        if reps >= _MAX_REPS:
            result["must_finish"] = True
            result["action"] = f"STOP RECORDING. Maximum {_MAX_REPS} reps reached. Call finish_pattern_training NOW."
        elif reps >= _MIN_REPS and trainer.ready:
            if reps >= 4:
                result["action"] = "Training is ready! You have enough data. Call finish_pattern_training now (or record 1-2 more reps for marginal improvement)."
            else:
                result["action"] = "Training is ready with minimum data. Recommend 1-2 more reps, then call finish_pattern_training."
        elif label == ACTIVE and counts["active"] >= 2:
            result["action"] = f"Record 1 rest segment, then 1 active segment. {_MIN_REPS - reps} more active reps needed minimum."
        else:
            result["action"] = "Continue alternating: 1 rest segment, then 1 active segment."
        
        return result

    def _finish_training(self, args: dict) -> dict:
        with self._lock:
            trainer = self._trainer
        if trainer is None:
            return {"error": "no training session to finish"}
        threshold = _as_float(args.get("threshold"), 0.6, lo=0.05, hi=0.95)
        try:
            pattern = trainer.fit(threshold=threshold)
        except TrainingError as exc:
            return {"error": str(exc), "totals": trainer.counts()}
        with self._lock:
            self._bank.add(pattern)
            self._trainer = None
        path = self._store.save(pattern.name, pattern.to_dict())
        cv = pattern.cv
        return {
            "status": "trained",
            "name": pattern.name,
            "saved_to": str(path),
            "balanced_accuracy": cv.get("balanced_accuracy"),
            "n_folds": cv.get("n_folds"),
            "channel_importance": pattern.classifier.channel_importance()[:5],
            "top_features": pattern.ranking.get("top_features", [])[:5],
            "caveat": pattern.ranking.get("caveat", ""),
        }

    def _cancel_training(self, args: dict) -> dict:
        with self._lock:
            had = self._trainer is not None
            self._trainer = None
        return {"status": "cancelled" if had else "nothing_to_cancel"}

    # ── inspection handlers ──────────────────────────────────────────────
    def _list_patterns(self, args: dict) -> dict:
        live = set(self._bank.names())
        rows = []
        for meta in self._store.list_meta():
            rows.append({
                "name": meta["name"],
                "balanced_accuracy": meta.get("score"),
                "metric": meta.get("metric"),
                "loaded": meta["name"] in live,
            })
        return {"count": len(rows), "patterns": rows}

    def _detect_patterns(self, args: dict) -> dict:
        snap = self._bank.snapshot()
        active = [s for s in snap if s["active"]]
        return {"patterns": snap, "active": [s["name"] for s in active]}

    def _explain_pattern(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip()
        pat = self._bank.get(name)
        if pat is None:
            data = self._store.load(name)
            if data is None:
                return {"error": f"unknown pattern {name!r}", "known": self._bank.names()}
            from ..decode import TrainedPattern
            pat = TrainedPattern.from_dict(data)
        return {
            "name": pat.name,
            "cross_validation": pat.cv,
            "channel_importance": pat.classifier.channel_importance(),
            "ranking": pat.ranking,
            "threshold": pat.threshold,
        }

    def _forget_pattern(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip()
        removed_live = self._bank.remove(name)
        removed_disk = self._store.delete(name)
        if not (removed_live or removed_disk):
            return {"error": f"unknown pattern {name!r}"}
        return {"status": "forgotten", "name": name}

    # ── connectivity handler ─────────────────────────────────────────────
    def _connectivity(self, args: dict) -> dict:
        band = str(args.get("band") or "Alpha")
        seconds = _as_float(args.get("seconds"), 8.0, lo=2.0, hi=30.0)
        with self._lock:
            hist = list(self._power_hist)
        if not hist:
            return {
                "status": "no_data",
                "detail": "No frames yet — start the stream/monitor first.",
            }
        t_end = hist[-1][0]
        frames = [pc for (t, pc) in hist if t >= t_end - seconds]
        if len(frames) < 8:
            return {
                "status": "insufficient",
                "detail": "Need a few seconds of signal for connectivity.",
                "n_frames": len(frames),
            }
        arr = np.stack(frames, axis=0)  # [T, n_ch, n_bands]
        return band_power_connectivity(arr, self._cascade.channel_labels(), band=band)

    # ── session handlers (the lab notebook) ──────────────────────────────
    def _record_session(self, args: dict) -> dict:
        label = str(args.get("label") or "").strip()
        if not label:
            return {"error": "a session label is required"}
        if self._recorder.is_open:
            return {"error": "a session recording is already in progress"}
        seconds = _as_float(args.get("seconds"), 20.0, lo=_MIN_SESSION_S, hi=_MAX_SESSION_S)
        labels = self._cascade.channel_labels()
        with self._lock:
            self._recorder.open(label, channel_labels=labels)
        self._sleep(seconds)            # frames stream in via on_frame meanwhile
        with self._lock:
            recording = self._recorder.close()
        if recording.n_frames == 0:
            return {
                "status": "no_frames",
                "label": label,
                "warning": "No frames captured — is the stream running?",
            }
        summary = recording.summary()
        # Fold in the artifacts that happened inside the recorded window.
        counts: dict[str, int] = {}
        for art in self._cascade.recent_artifacts(300):
            if art.timestamp >= recording.started_at:
                counts[art.type] = counts.get(art.type, 0) + 1
        summary["artifacts"] = counts
        path = self._sessions.save(label, summary)
        return {
            "status": "recorded",
            "label": label,
            "n_frames": recording.n_frames,
            "duration_s": summary["duration_s"],
            "dominant_band": summary["dominant_band"],
            "indices": summary["indices"],
            "signal_quality": summary["signal_quality"],
            "connectivity": summary["connectivity"],
            "artifacts": counts,
            "saved_to": str(path),
        }

    def _list_sessions(self, args: dict) -> dict:
        rows = self._sessions.list_meta()
        return {"count": len(rows), "sessions": rows}

    def _analyze_session(self, args: dict) -> dict:
        label = str(args.get("label") or "").strip()
        data = self._sessions.load(label)
        if data is None:
            return {
                "error": f"unknown session {label!r}",
                "known": [m["name"] for m in self._sessions.list_meta()],
            }
        return data

    def _compare_sessions(self, args: dict) -> dict:
        a = str(args.get("a") or "").strip()
        b = str(args.get("b") or "").strip()
        if not a or not b:
            return {"error": "two session labels are required: 'a' and 'b'"}
        da = self._sessions.load(a)
        db = self._sessions.load(b)
        missing = [name for name, data in ((a, da), (b, db)) if data is None]
        if missing:
            return {
                "error": f"unknown session(s): {', '.join(missing)}",
                "known": [m["name"] for m in self._sessions.list_meta()],
            }
        return compare_summaries(da, db)

    def _forget_session(self, args: dict) -> dict:
        label = str(args.get("label") or "").strip()
        if not self._sessions.delete(label):
            return {"error": f"unknown session {label!r}"}
        return {"status": "forgotten", "label": label}


def _as_float(value, default: float, *, lo: float, hi: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _as_int(value, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))
