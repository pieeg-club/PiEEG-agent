import { useEffect } from "react";
import type { ChannelImportance, PatternExplain } from "../types";
import { clamp01, pct, scoreTone } from "../util";

function num(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}

// A read-only explanation of a trained pattern: its cross-validated score, the
// per-channel importance the detector leans on, and the top rest-vs-active
// features. Data comes from GET /api/patterns/{name} (explain_pattern).
export function PatternModal({
  name,
  data,
  onClose,
}: {
  name: string;
  data: PatternExplain | null;
  onClose: () => void;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);
  const cv = data?.cross_validation as Record<string, unknown> | undefined;
  const bacc = num(cv?.balanced_accuracy);
  const folds = num(cv?.n_folds);
  const importance = (data?.channel_importance || []) as ChannelImportance[];
  const top = (data?.ranking?.top_features || []) as {
    feature: string;
    channel?: string;
    cohens_d: number;
  }[];
  const caveat = data?.ranking?.caveat;

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>{name}</h2>
          <button className="x" onClick={onClose}>
            ×
          </button>
        </div>

        {!data ? (
          <div className="muted">Loading…</div>
        ) : data.error ? (
          <div className="muted">{data.error}</div>
        ) : (
          <div className="modal-body">
            <div className="explain-top">
              <div className={"score-badge " + scoreTone(bacc)}>
                <span className="score-num">{pct(bacc)}</span>
                <span className="score-label">balanced acc.</span>
              </div>
              <div className="explain-facts">
                {folds != null && <div>{folds}-fold leave-one-rep-out CV</div>}
                {data.threshold != null && <div>fires above {pct(data.threshold)}</div>}
                {bacc != null && bacc < 0.7 && (
                  <div className="warn-line">weak separation — add more reps</div>
                )}
              </div>
            </div>

            {importance.length > 0 && (
              <div className="explain-section">
                <div className="section-title">Channel importance</div>
                {importance.map((c) => (
                  <div className="band-row" key={c.channel}>
                    <span className="band-name">{c.channel}</span>
                    <div className="band-track">
                      <div
                        className="band-fill pattern-fill"
                        style={{ width: `${clamp01(c.importance) * 100}%` }}
                      />
                    </div>
                    <span className="band-val">{pct(c.importance)}</span>
                  </div>
                ))}
              </div>
            )}

            {top.length > 0 && (
              <div className="explain-section">
                <div className="section-title">Top separating features</div>
                <ul className="feat-list">
                  {top.slice(0, 6).map((f, i) => (
                    <li key={i}>
                      <span>
                        {f.channel ? `${f.channel} · ` : ""}
                        {f.feature}
                      </span>
                      <span className="feat-d">d={f.cohens_d}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {caveat && <div className="caveat">{caveat}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
