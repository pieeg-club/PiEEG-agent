import type { Artifacts, Bands, NeuralState, Quality } from "../types";
import { ago, cap, clamp01, orderBands, pct } from "../util";

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
export function StateCard({ state }: { state?: NeuralState }) {
  const nodata = !state || state.status === "no_data";
  return (
    <section className="card">
      <div className="card-title">
        Live state
        {state?.warming_up && <span className="tag">warming up</span>}
      </div>
      {nodata ? (
        <div className="muted">Waiting for the first analysis window…</div>
      ) : (
        <>
          <Meter label="Focus" value={state.focus} tone="t-focus" />
          <Meter label="Relax" value={state.relax} tone="t-relax" />
          <Meter label="Engagement" value={state.engagement} tone="t-engage" />
          <div className="state-footer">
            <span>
              dominant <b>{cap(state.dominant_band)}</b>
            </span>
            <span>
              signal <b>{pct(state.signal_quality)}</b>
            </span>
          </div>
          {state.bad_channels && state.bad_channels.length > 0 && (
            <div className="warn-line">check: {state.bad_channels.join(", ")}</div>
          )}
        </>
      )}
    </section>
  );
}

// Relative band powers averaged across channels. Mirrors get_band_powers.
export function BandBars({ bands }: { bands?: Bands }) {
  const rel = bands?.relative;
  const nodata = !rel || bands?.status === "no_data";
  return (
    <section className="card">
      <div className="card-title">Frequency bands</div>
      {nodata ? (
        <div className="muted">No spectrum yet…</div>
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
}

const Q_TONE: Record<string, string> = {
  good: "q-good",
  flat: "q-flat",
  rail: "q-rail",
  noisy: "q-noisy",
  line: "q-line",
};

// Per-channel signal-quality verdicts. Mirrors get_channel_quality.
export function QualityGrid({ quality }: { quality?: Quality }) {
  const chans = quality?.channels;
  return (
    <section className="card">
      <div className="card-title">
        Channels
        {quality?.overall != null && <span className="tag">{pct(quality.overall)}</span>}
      </div>
      {!chans || !chans.length ? (
        <div className="muted">No channels yet…</div>
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
}

const ART_TONE: Record<string, string> = {
  blink: "a-blink",
  blink_double: "a-blink",
  jaw_clench: "a-jaw",
  motion: "a-motion",
};

// Recent transients (blink / jaw / motion). Mirrors find_artifacts.
export function ArtifactFeed({ artifacts }: { artifacts?: Artifacts }) {
  const rows = (artifacts?.artifacts || []).slice().reverse();
  return (
    <section className="card">
      <div className="card-title">Artifacts</div>
      {!rows.length ? (
        <div className="muted">Clean — no recent transients.</div>
      ) : (
        <ul className="feed">
          {rows.map((a, i) => (
            <li key={i} className={a.severity === "warn" ? "warn" : ""}>
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
}
