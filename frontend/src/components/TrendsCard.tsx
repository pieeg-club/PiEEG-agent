import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TrendPoint } from "../types";
import { clamp01 } from "../util";

// ── static configuration ──────────────────────────────────────────────────

const WINDOWS = [
  { label: "5m",  s:  5 * 60 },
  { label: "15m", s: 15 * 60 },
  { label: "30m", s: 30 * 60 },
  { label: "1h",  s: 60 * 60 },
  { label: "2h",  s: 120 * 60 },
] as const;

const MAX_BUCKETS = 240;

// Internal canvas padding in CSS px.
const PAD_L = 30;
const PAD_R = 10;
const PAD_T = 8;
const PAD_B = 18;

type SeriesKey = keyof Omit<TrendPoint, "t">;

interface Series {
  key: SeriesKey;
  color: string; // hex — CSS vars don't resolve in Canvas2D
  label: string;
}

const INDEX_SERIES: Series[] = [
  { key: "focus",      color: "#7c9cff", label: "Focus" },
  { key: "relax",      color: "#5eead4", label: "Relax" },
  { key: "engagement", color: "#f0a36b", label: "Engagement" },
  { key: "quality",    color: "#8b93a7", label: "Signal" },
];

const BAND_SERIES: Series[] = [
  { key: "delta", color: "#6c8cff", label: "δ" },
  { key: "theta", color: "#56c4d8", label: "θ" },
  { key: "alpha", color: "#5eead4", label: "α" },
  { key: "beta",  color: "#f0a36b", label: "β" },
  { key: "gamma", color: "#f97e72", label: "γ" },
];

// ── downsampling ──────────────────────────────────────────────────────────

const AVG_KEYS: SeriesKey[] = [
  "focus", "relax", "engagement", "quality",
  "delta", "theta", "alpha", "beta", "gamma",
];

type Bucket = TrendPoint;

function bucketize(points: TrendPoint[], start: number, end: number): Bucket[] {
  const span = end - start;
  if (span <= 0 || points.length === 0) return [];
  type Slot = { sum: Record<string, number>; n: number; tsum: number };
  const slots: (Slot | undefined)[] = new Array(MAX_BUCKETS);
  for (const p of points) {
    if (p.t < start || p.t > end) continue;
    let idx = Math.floor(((p.t - start) / span) * MAX_BUCKETS);
    if (idx >= MAX_BUCKETS) idx = MAX_BUCKETS - 1;
    if (idx < 0) idx = 0;
    let slot = slots[idx];
    if (!slot) {
      const sum: Record<string, number> = {};
      for (const k of AVG_KEYS) sum[k] = 0;
      slot = { sum, n: 0, tsum: 0 };
      slots[idx] = slot;
    }
    for (const k of AVG_KEYS) slot.sum[k] += p[k];
    slot.n++;
    slot.tsum += p.t;
  }
  const out: Bucket[] = [];
  for (const slot of slots) {
    if (!slot) continue;
    const b: Record<string, number> = { t: slot.tsum / slot.n };
    for (const k of AVG_KEYS) b[k] = slot.sum[k] / slot.n;
    out.push(b as Bucket);
  }
  return out;
}

const mean = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

function trendOf(buckets: Bucket[], key: SeriesKey) {
  if (buckets.length < 4) return { dir: "flat" as const };
  const n = buckets.length;
  const size = Math.max(1, Math.round(n / 5));
  const head = mean(buckets.slice(0, size).map((b) => b[key]));
  const tail = mean(buckets.slice(-size).map((b) => b[key]));
  const delta = tail - head;
  if (delta > 0.05) return { dir: "up" as const };
  if (delta < -0.05) return { dir: "down" as const };
  return { dir: "flat" as const };
}

function relLabel(deltaS: number): string {
  if (deltaS <= 2) return "now";
  if (deltaS < 60) return `-${Math.round(deltaS)}s`;
  const m = deltaS / 60;
  if (m < 60) return `-${Math.round(m)}m`;
  const h = deltaS / 3600;
  return `-${h < 10 ? h.toFixed(1) : Math.round(h)}h`;
}

function fmtDuration(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

// ── canvas draw ───────────────────────────────────────────────────────────

interface DrawData {
  buckets: Bucket[];
  start: number;
  end: number;
  series: Series[];
  hidden: Set<SeriesKey>;
  hoverT: number | null;
  w: number;
  h: number;
}

function drawChart(ctx: CanvasRenderingContext2D, d: DrawData, dpr: number): void {
  const { buckets, start, end, series, hidden, hoverT, w, h } = d;
  const span = Math.max(1, end - start);
  const plotW = w - PAD_L - PAD_R;
  const plotH = h - PAD_T - PAD_B;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (plotW <= 0 || plotH <= 0) return;

  const xOf = (t: number) => PAD_L + ((t - start) / span) * plotW;
  const yOf = (v: number) => PAD_T + (1 - clamp01(v)) * plotH;

  // Grid + y labels
  ctx.font = `8px "Geist Mono",Consolas,monospace`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (const g of [0, 0.5, 1]) {
    const y = yOf(g);
    ctx.strokeStyle = "rgba(35,42,57,0.9)";
    ctx.lineWidth = 0.75;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(PAD_L, y);
    ctx.lineTo(w - PAD_R, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#5b6478";
    ctx.fillText(`${Math.round(g * 100)}`, PAD_L - 3, y);
  }

  // X labels
  ctx.fillStyle = "#5b6478";
  ctx.textBaseline = "bottom";
  for (let i = 0; i <= 3; i++) {
    const t = start + (span * i) / 3;
    ctx.textAlign = i === 0 ? "left" : i === 3 ? "right" : "center";
    ctx.fillText(relLabel(end - t), xOf(t), h - 2);
  }

  if (buckets.length < 2) return;

  // Series lines
  for (const s of series) {
    if (hidden.has(s.key)) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash([]);
    ctx.beginPath();
    let first = true;
    for (const b of buckets) {
      const x = xOf(b.t);
      const y = yOf(b[s.key]);
      if (first) { ctx.moveTo(x, y); first = false; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Hover crosshair + dots
  if (hoverT !== null) {
    let nearest = buckets[0];
    let bestD = Math.abs(nearest.t - hoverT);
    for (const b of buckets) {
      const d = Math.abs(b.t - hoverT);
      if (d < bestD) { bestD = d; nearest = b; }
    }
    const cx = xOf(nearest.t);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, PAD_T);
    ctx.lineTo(cx, PAD_T + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const s of series) {
      if (hidden.has(s.key)) continue;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(cx, yOf(nearest[s.key]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ── LineChart ─────────────────────────────────────────────────────────────

interface LineChartProps {
  buckets: Bucket[];
  start: number;
  end: number;
  series: Series[];
  hidden: Set<SeriesKey>;
  hoverT: number | null;
  onHover: (t: number | null) => void;
}

const LineChart = memo(function LineChart({
  buckets, start, end, series, hidden, hoverT, onHover,
}: LineChartProps) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const needsRender = useRef(true);
  const rafRef      = useRef<number>(0);
  const dataRef     = useRef<DrawData>({
    buckets, start, end, series, hidden, hoverT, w: 0, h: 0,
  });

  // Sync props into ref + mark dirty after every render
  useEffect(() => {
    dataRef.current = { ...dataRef.current, buckets, start, end, series, hidden, hoverT };
    needsRender.current = true;
  });

  // RAF loop — set up once on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const applySize = () => {
      const dpr = window.devicePixelRatio || 1;
      const w   = canvas.offsetWidth;
      const h   = canvas.offsetHeight;
      if (w === 0 || h === 0) return;
      canvas.width  = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      dataRef.current = { ...dataRef.current, w, h };
      needsRender.current = true;
    };

    applySize();
    const ro = new ResizeObserver(applySize);
    ro.observe(canvas);

    const render = () => {
      rafRef.current = requestAnimationFrame(render);
      if (!needsRender.current) return;
      const d = dataRef.current;
      if (d.w === 0 || d.h === 0) return;
      needsRender.current = false;
      drawChart(ctx, d, window.devicePixelRatio || 1);
    };
    rafRef.current = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, []);

  const hoverBucket = useMemo(() => {
    if (hoverT == null || buckets.length === 0) return null;
    return buckets.reduce((best, b) =>
      Math.abs(b.t - hoverT) < Math.abs(best.t - hoverT) ? b : best,
    );
  }, [hoverT, buckets]);

  const handleMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const { start: s, end: en } = dataRef.current;
    onHover(s + frac * (en - s));
  }, [onHover]);

  const tipLeft = hoverBucket != null
    ? PAD_L + ((hoverBucket.t - start) / Math.max(1, end - start)) * (dataRef.current.w - PAD_L - PAD_R)
    : 0;

  return (
    <div className="tc-chart-wrap">
      <canvas
        ref={canvasRef}
        className="tc-canvas"
        onMouseMove={handleMove}
        onMouseLeave={() => onHover(null)}
      />
      {hoverBucket && (
        <div className="tc-tip" style={{ left: tipLeft }}>
          <div className="tc-tip-time">{relLabel(end - hoverBucket.t)}</div>
          {series.filter((s) => !hidden.has(s.key)).map((s) => (
            <div className="tc-tip-row" key={s.key}>
              <span className="tc-tip-dot" style={{ background: s.color }} />
              <span className="tc-tip-lab">{s.label}</span>
              <span className="tc-tip-val">{Math.round(hoverBucket[s.key] * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

// ── Legend ────────────────────────────────────────────────────────────────

function Legend({
  series, hidden, onToggle,
}: { series: Series[]; hidden: Set<SeriesKey>; onToggle: (k: SeriesKey) => void }) {
  return (
    <div className="tc-legend">
      {series.map((s) => (
        <button
          key={s.key}
          className={"tc-legend-item" + (hidden.has(s.key) ? " off" : "")}
          onClick={() => onToggle(s.key)}
          title={s.label}
        >
          <span className="tc-legend-dot" style={{ background: s.color }} />
          {s.label}
        </button>
      ))}
    </div>
  );
}

// ── Insight row ───────────────────────────────────────────────────────────

const ARROW: Record<"up" | "down" | "flat", string> = { up: "▲", down: "▼", flat: "▬" };

function InsightRow({ buckets }: { buckets: Bucket[] }) {
  const rows: { label: string; key: SeriesKey; color: string }[] = [
    { label: "Focus",  key: "focus",      color: "#7c9cff" },
    { label: "Relax",  key: "relax",      color: "#5eead4" },
    { label: "Engage", key: "engagement", color: "#f0a36b" },
  ];
  return (
    <div className="tc-insight">
      {rows.map((r) => {
        const t   = trendOf(buckets, r.key);
        const avg = mean(buckets.map((b) => b[r.key]));
        return (
          <div className="tc-insight-cell" key={r.key}>
            <span className="tc-insight-dot" style={{ background: r.color }} />
            <span className="tc-insight-lab">{r.label}</span>
            <span className="tc-insight-avg">{Math.round(avg * 100)}%</span>
            <span className={"tc-insight-dir dir-" + t.dir}>{ARROW[t.dir]}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── TrendsCard ────────────────────────────────────────────────────────────

export const TrendsCard = memo(function TrendsCard({
  points,
  onClear,
}: {
  points: TrendPoint[];
  onClear: () => void;
}) {
  const [windowS, setWindowS] = useState<number>(WINDOWS[1].s);
  const [hidden,  setHidden]  = useState<Set<SeriesKey>>(new Set());
  const [hoverT,  setHoverT]  = useState<number | null>(null);

  const toggle = useCallback((key: SeriesKey) =>
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    }), []);

  const now     = points.length ? points[points.length - 1].t : Date.now() / 1000;
  const start   = now - windowS;
  const buckets = useMemo(() => bucketize(points, start, now), [points, start, now]);

  const trackedFor = points.length ? points[points.length - 1].t - points[0].t : 0;
  const enough     = buckets.length >= 2;

  return (
    <section className="card tc-card">
      {/* ── header ── */}
      <div className="card-title">
        Neural Trends
        {trackedFor > 0 && (
          <span className="tag">{fmtDuration(trackedFor)}</span>
        )}
        <div className="tc-windows" style={{ marginLeft: "auto" }}>
          {WINDOWS.map((w) => (
            <button
              key={w.s}
              className={"tc-win" + (windowS === w.s ? " active" : "")}
              onClick={() => setWindowS(w.s)}
            >
              {w.label}
            </button>
          ))}
        </div>
        {trackedFor > 0 && (
          <button className="tc-clear-btn" onClick={onClear} title="Clear history">
            ✕
          </button>
        )}
      </div>

      {/* ── content ── */}
      {!enough ? (
        <div className="tc-waiting">
          <span className="tc-waiting-icon">⏳</span>
          Collecting history… check back in a minute.
        </div>
      ) : (
        <>
          <InsightRow buckets={buckets} />

          <div className="tc-section">
            <div className="tc-sec-head">
              <span className="tc-sec-label">Cognitive</span>
              <Legend series={INDEX_SERIES} hidden={hidden} onToggle={toggle} />
            </div>
            <LineChart
              buckets={buckets} start={start} end={now}
              series={INDEX_SERIES} hidden={hidden}
              hoverT={hoverT} onHover={setHoverT}
            />
          </div>

          <div className="tc-section">
            <div className="tc-sec-head">
              <span className="tc-sec-label">Bands</span>
              <Legend series={BAND_SERIES} hidden={hidden} onToggle={toggle} />
            </div>
            <LineChart
              buckets={buckets} start={start} end={now}
              series={BAND_SERIES} hidden={hidden}
              hoverT={hoverT} onHover={setHoverT}
            />
          </div>

          <div className="tc-foot">
            Self-normalised within-session — drift relative to your own baseline.
          </div>
        </>
      )}
    </section>
  );
});
