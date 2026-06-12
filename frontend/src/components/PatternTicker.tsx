import { memo, useCallback } from "react";
import type { Patterns } from "../types";
import { clamp01, pct } from "../util";

// Live activation of every loaded pattern (smoothed probability + on/off),
// plus the entry points to train a new one, explain one, or forget one.
// Mirrors detect_patterns; explain/forget go through the REST API.
export const PatternTicker = memo(function PatternTicker({
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
  
  const getHealthIcon = useCallback((status: string) => {
    switch (status) {
      case 'healthy': return '✓';
      case 'degraded': return '⚠';
      case 'needs_retrain': return '🔄';
      default: return '';
    }
  }, []);
  
  const getHealthClass = useCallback((status: string) => {
    switch (status) {
      case 'healthy': return 'health-healthy';
      case 'degraded': return 'health-degraded';
      case 'needs_retrain': return 'health-needs-retrain';
      default: return '';
    }
  }, []);
  
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
                {p.health && (
                  <span className={`health-icon ${getHealthClass(p.health.status)}`}
                        title={`Confidence: ${pct(p.health.confidence)} (${p.health.status})`}>
                    {getHealthIcon(p.health.status)}
                  </span>
                )}
              </button>
              <div className="pattern-tracks">
                {/* Probability bar */}
                <div className="band-track">
                  <div
                    className="band-fill pattern-fill"
                    style={{ width: `${clamp01(p.probability) * 100}%` }}
                  />
                </div>
                {/* Confidence bar (if health data available) */}
                {p.health && (
                  <div className="band-track confidence-track" title={`Confidence: ${pct(p.health.confidence)}`}>
                    <div
                      className={`band-fill confidence-fill ${getHealthClass(p.health.status)}`}
                      style={{ width: `${clamp01(p.health.confidence) * 100}%` }}
                    />
                  </div>
                )}
              </div>
              <span className="band-val">
                {pct(p.probability)}
                {p.health && p.health.status !== 'healthy' && (
                  <span className={`health-badge ${getHealthClass(p.health.status)}`}>
                    {pct(p.health.confidence)}
                  </span>
                )}
              </span>
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
});
