import { memo } from "react";
import type { Artifacts, Bands, NeuralState, Quality } from "../types";
import { ago, cap, clamp01, orderBands, pct } from "../util";

function Skeleton({ width = "100%", height = "10px" }: { width?: string; height?: string }) {
  return <div className="skeleton" style={{ width, height }} />;
}

function Meter({ label, value, tone }: { label: string; value?: number; tone?: string }) {
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span>{pct(value)}</span>
      </div>
      <div className="meter-track">
        <div
          className={"meter-fill " + (tone || "")}
          style={{ width: `${clamp01(value || 0) * 100}%` }}
        />
      </div>
    </div>
  );
}

// The ~1 Hz smoothed state: focus / relax / engagement, dominant band and a
// signal-quality readout. Mirrors get_neural_state.
export const StateCard = memo(function StateCard({ state }: { state?: NeuralState }) {
  const nodata = !state || state.status === "no_data";
  return (
    <section className="card state-card">
      <div className="card-title">
        Live Neural State
        {state?.warming_up && <span className="tag">warming up</span>}
        {state?.signal_quality != null && (
          <span className="tag" style={{ marginLeft: "auto" }}>
            {pct(state.signal_quality)} quality
          </span>
        )}
      </div>
      {nodata ? (
        <div className="skeleton-container">
          <Skeleton height="36px" />
          <Skeleton height="36px" />
          <Skeleton height="36px" />
          <div style={{ display: "flex", gap: "12px", marginTop: "12px" }}>
            <Skeleton width="50%" height="24px" />
            <Skeleton width="50%" height="24px" />
          </div>
        </div>
      ) : (
        <>
          <Meter label="Focus" value={state.focus} tone="t-focus" />
          <Meter label="Relax" value={state.relax} tone="t-relax" />
          <Meter label="Engagement" value={state.engagement} tone="t-engage" />
          <div className="state-footer">
            <span className="state-stat">
              <span className="stat-label">Dominant Band</span>
              <span className="stat-value">{cap(state.dominant_band)}</span>
            </span>
            {state.n_channels != null && (
              <span className="state-stat">
                <span className="stat-label">Channels</span>
                <span className="stat-value">{state.n_channels}</span>
              </span>
            )}
          </div>
          {state.bad_channels && state.bad_channels.length > 0 ? (
            <div className="warn-line">⚠ Check: {state.bad_channels.join(", ")}</div>
          ) : (
            <div className="warn-line ok-line">✓ All channels clean</div>
          )}
        </>
      )}
    </section>
  );
});

// Relative band powers averaged across channels. Mirrors get_band_powers.
export const BandBars = memo(function BandBars({ bands }: { bands?: Bands }) {
  const rel = bands?.relative;
  const nodata = !rel || bands?.status === "no_data";
  return (
    <section className="card">
      <div className="card-title">Frequency bands</div>
      {nodata ? (
        <div className="skeleton-container">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} height="24px" />
          ))}
        </div>
      ) : (
        <div className="bands">
          {orderBands(rel!).map(([name, val]) => (
            <div className="band-row" key={name}>
              <span className="band-name">{cap(name)}</span>
              <div className="band-track">
                <div
                  className={"band-fill band-" + name.toLowerCase()}
                  style={{ width: `${clamp01(val) * 100}%` }}
                />
              </div>
              <span className="band-val">{pct(val)}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
});

const Q_TONE: Record<string, string> = {
  good: "q-good",
  flat: "q-flat",
  rail: "q-rail",
  noisy: "q-noisy",
  line: "q-line",
};

// Per-channel signal-quality verdicts. Mirrors get_channel_quality.
export const QualityGrid = memo(function QualityGrid({ quality }: { quality?: Quality }) {
  const chans = quality?.channels;
  return (
    <section className="card">
      <div className="card-title">
        Channels
        {quality?.overall != null && <span className="tag">{pct(quality.overall)}</span>}
      </div>
      {!chans || !chans.length ? (
        <div className="skeleton-container">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(70px, 1fr))", gap: "8px" }}>
            {[...Array(8)].map((_, i) => (
              <Skeleton key={i} height="36px" />
            ))}
          </div>
        </div>
      ) : (
        <div className="chan-grid">
          {chans.map((c) => (
            <div
              className={"chan " + (Q_TONE[c.status] || "")}
              key={c.index}
              title={`${c.label}: ${c.status} · ${c.rms_uv}µV · line×${c.line_ratio} · score ${pct(
                c.score
              )}`}
            >
              <span className="chan-label">{c.label}</span>
              <span className="chan-status">{c.status}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
});

const ART_TONE: Record<string, string> = {
  blink: "a-blink",
  blink_double: "a-blink",
  jaw_clench: "a-jaw",
  motion: "a-motion",
};

// Recent transients (blink / jaw / motion). Mirrors find_artifacts.
export const ArtifactFeed = memo(function ArtifactFeed({ artifacts }: { artifacts?: Artifacts }) {
  const rows = (artifacts?.artifacts || []).slice().reverse();
  return (
    <section className="card">
      <div className="card-title">Artifacts</div>
      {!rows.length ? (
        <div className="muted">Clean — no recent transients.</div>
      ) : (
        <ul className="feed">
          {rows.map((a, i) => (
            <li key={`${a.type}-${a.channel}-${a.timestamp}-${i}`} className={a.severity === "warn" ? "warn" : ""}>
              <span className={"feed-dot " + (ART_TONE[a.type] || "")} />
              <span className="feed-type">{a.type.replace(/_/g, " ")}</span>
              <span className="feed-meta">
                {a.channel} · {Math.round(a.duration_ms)}ms
              </span>
              <span className="feed-age">{ago(a.timestamp)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
});
