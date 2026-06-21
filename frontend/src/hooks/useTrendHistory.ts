import { useCallback, useEffect, useRef, useState } from "react";
import type { Snapshot, TrendPoint } from "../types";

// How long we keep history (seconds). 2h at ~1 Hz ≈ 7200 points — cheap to
// hold in memory and small enough to persist to localStorage.
const MAX_AGE_S = 2 * 60 * 60;
const MAX_POINTS = 7400;
// Don't record faster than this — the backend state rate is ~1 Hz while the
// live socket pushes ~4 Hz, so we throttle to one point per second.
const MIN_INTERVAL_S = 1.0;
// Persist at most this often to avoid hammering localStorage.
const PERSIST_EVERY_MS = 15_000;
const STORE_KEY = "pieeg:trends:v1";

const r3 = (v: number) => Math.round(v * 1000) / 1000;

function bandVal(rel: Record<string, number> | undefined, name: string): number {
  if (!rel) return 0;
  // Band keys arrive capitalised ("Alpha") from the backend; be tolerant.
  const v = rel[name] ?? rel[name[0].toUpperCase() + name.slice(1)] ?? rel[name.toLowerCase()];
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

function load(): TrendPoint[] {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const cutoff = Date.now() / 1000 - MAX_AGE_S;
    return parsed.filter(
      (p): p is TrendPoint => p && typeof p.t === "number" && p.t >= cutoff,
    );
  } catch {
    return [];
  }
}

/**
 * Accumulates the live snapshot into a long-term, downsampled trend buffer so
 * the UI can show how focus / relax / engagement and the frequency bands drift
 * over minutes-to-hours rather than just the last few seconds.
 *
 * The buffer is throttled to ~1 Hz, capped to the last two hours, and mirrored
 * to localStorage so a page refresh doesn't wipe a long working session.
 */
export function useTrendHistory(snapshot: Snapshot | null) {
  const [points, setPoints] = useState<TrendPoint[]>(() => load());
  const lastTsRef = useRef(0);
  const lastPersistRef = useRef(0);

  useEffect(() => {
    const state = snapshot?.state;
    if (!state || state.status === "no_data" || state.warming_up) return;
    if (state.focus == null || state.relax == null || state.engagement == null) return;

    // Prefer the backend sample time; fall back to the wall clock.
    const t = state.timestamp ?? snapshot?.bands?.timestamp ?? Date.now() / 1000;
    if (t - lastTsRef.current < MIN_INTERVAL_S) return;
    lastTsRef.current = t;

    const rel = snapshot?.bands?.relative ?? state.rel_bands;
    const point: TrendPoint = {
      t: r3(t),
      focus: r3(state.focus),
      relax: r3(state.relax),
      engagement: r3(state.engagement),
      quality: r3(state.signal_quality ?? 0),
      delta: r3(bandVal(rel, "delta")),
      theta: r3(bandVal(rel, "theta")),
      alpha: r3(bandVal(rel, "alpha")),
      beta: r3(bandVal(rel, "beta")),
      gamma: r3(bandVal(rel, "gamma")),
    };

    setPoints((prev) => {
      const cutoff = t - MAX_AGE_S;
      const next = prev.length && prev[prev.length - 1].t >= t ? prev.slice() : [...prev, point];
      // Drop anything older than the window or beyond the hard cap.
      let trimmed = next.filter((p) => p.t >= cutoff);
      if (trimmed.length > MAX_POINTS) trimmed = trimmed.slice(trimmed.length - MAX_POINTS);

      const now = Date.now();
      if (now - lastPersistRef.current > PERSIST_EVERY_MS) {
        lastPersistRef.current = now;
        try {
          localStorage.setItem(STORE_KEY, JSON.stringify(trimmed));
        } catch {
          /* storage full / unavailable — keep running in-memory only */
        }
      }
      return trimmed;
    });
  }, [snapshot]);

  // Flush to storage when the tab is hidden or closed so the latest session
  // isn't lost between persist intervals.
  useEffect(() => {
    const flush = () => {
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify(points));
      } catch {
        /* ignore */
      }
    };
    window.addEventListener("beforeunload", flush);
    document.addEventListener("visibilitychange", flush);
    return () => {
      window.removeEventListener("beforeunload", flush);
      document.removeEventListener("visibilitychange", flush);
    };
  }, [points]);

  const clear = useCallback(() => {
    setPoints([]);
    lastTsRef.current = 0;
    try {
      localStorage.removeItem(STORE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  return { points, clear };
}
