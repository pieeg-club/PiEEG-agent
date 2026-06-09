// Small presentation helpers shared by the brain-panel cards.

export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export const pct = (v?: number | null) =>
  v == null || Number.isNaN(v) ? "—" : `${Math.round(v * 100)}%`;

export const cap = (s?: string) => (s ? s[0].toUpperCase() + s.slice(1) : "—");

// Wall-clock seconds → compact "now / 3s / 4m / 2h".
export function ago(ts?: number): string {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 1) return "now";
  if (s < 60) return `${Math.floor(s)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h`;
}

const BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"];

// Sort band entries into the canonical low→high order, unknown keys last.
export function orderBands(rel: Record<string, number>): [string, number][] {
  return Object.entries(rel).sort((a, b) => {
    const ia = BAND_ORDER.indexOf(a[0].toLowerCase());
    const ib = BAND_ORDER.indexOf(b[0].toLowerCase());
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}

// Balanced-accuracy → traffic-light class (mirrors the CLI's honesty band).
export function scoreTone(score?: number | null): string {
  if (score == null) return "";
  if (score < 0.7) return "score-weak";
  if (score < 0.85) return "score-ok";
  return "score-good";
}
