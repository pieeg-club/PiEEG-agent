import type { Patterns } from "../types";
import { clamp01, pct } from "../util";

// Live activation of every loaded pattern (smoothed probability + on/off),
// plus the entry points to train a new one, explain one, or forget one.
// Mirrors detect_patterns; explain/forget go through the REST API.
export function PatternTicker({
  patterns,
  onTrain,
  onExplain,
  onForget,
}: {
  patterns?: Patterns;
  onTrain: () => void;
  onExplain: (name: string) => void;
  onForget: (name: string) => void;
}) {
  const rows = patterns?.patterns || [];
  return (
    <section className="card">
      <div className="card-title">
        Patterns
        <button className="btn-sm" onClick={onTrain}>
          + Train
        </button>
      </div>
      {!rows.length ? (
        <div className="muted">No patterns yet. Train one to start decoding.</div>
      ) : (
        <div className="patterns">
          {rows.map((p) => (
            <div className={"pattern-row " + (p.active ? "active" : "")} key={p.name}>
              <button
                className="pattern-name"
                onClick={() => onExplain(p.name)}
                title="Explain this pattern"
              >
                {p.name}
              </button>
              <div className="band-track">
                <div
                  className="band-fill pattern-fill"
                  style={{ width: `${clamp01(p.probability) * 100}%` }}
                />
              </div>
              <span className="band-val">{pct(p.probability)}</span>
              <button
                className="x"
                onClick={() => onForget(p.name)}
                title="Forget this pattern"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
