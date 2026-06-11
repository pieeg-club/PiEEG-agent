import type { Connectivity } from "../types";

interface Props {
  conn: Connectivity;
}

export function ConnectivityCard({ conn }: Props) {
  if (conn.status === "no_data" || conn.status === "insufficient") {
    return (
      <div className="card">
        <div className="card-header">Connectivity</div>
        <div className="card-body">
          <p className="muted">{conn.detail || "Warming up…"}</p>
        </div>
      </div>
    );
  }

  const pairs = conn.strongest_pairs || [];
  const mean = conn.mean_connectivity ?? 0;
  const band = conn.band || "Alpha";

  // Mini 8x8 heatmap if matrix is available (for 8ch PiEEG)
  const matrix = conn.matrix || [];
  const labels = conn.labels || [];
  const showMatrix = matrix.length <= 8 && matrix.length > 0;

  return (
    <div className="card">
      <div className="card-header">Connectivity · {band}</div>
      <div className="card-body">
        {/* Mean connectivity bar */}
        <div className="conn-mean">
          <span className="label">Mean r:</span>
          <span className="value">{mean.toFixed(3)}</span>
        </div>

        {/* Strongest pair */}
        {pairs.length > 0 && (
          <div className="conn-pair">
            <span className="label">Top:</span>
            <span className="value">
              {pairs[0].a}–{pairs[0].b} ({pairs[0].r >= 0 ? "+" : ""}
              {pairs[0].r.toFixed(2)})
            </span>
          </div>
        )}

        {/* Mini heatmap */}
        {showMatrix && (
          <div className="conn-heatmap">
            <div className="heatmap-grid" style={{ gridTemplateColumns: `repeat(${matrix.length}, 1fr)` }}>
              {matrix.map((row, i) =>
                row.map((val, j) => {
                  const norm = Math.abs(val);
                  const bgAlpha = Math.min(norm, 1.0);
                  const bg = val >= 0 ? `rgba(59, 130, 246, ${bgAlpha})` : `rgba(239, 68, 68, ${bgAlpha})`;
                  return (
                    <div
                      key={`${i}-${j}`}
                      className="heatmap-cell"
                      style={{ backgroundColor: bg }}
                      title={`${labels[i]}–${labels[j]}: ${val.toFixed(2)}`}
                    />
                  );
                })
              )}
            </div>
            <div className="heatmap-labels">
              {labels.map((ch) => (
                <span key={ch} className="heatmap-label">
                  {ch}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
