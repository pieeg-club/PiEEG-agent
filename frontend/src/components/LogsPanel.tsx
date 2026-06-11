import { useEffect, useRef, useState } from "react";
import type { LogEntry, LogLevel } from "../hooks/useLogsCapture";

const LEVEL_COLORS: Record<LogLevel, string> = {
  llm: "var(--accent)",
  tool: "var(--accent-2)",
  system: "var(--purple)",
  error: "var(--bad)",
  websocket: "var(--muted)",
};

const LEVEL_ICONS: Record<LogLevel, string> = {
  llm: "🧠",
  tool: "⚙️",
  system: "⚡",
  error: "⚠️",
  websocket: "🔌",
};

function formatTimestamp(ts: number): string {
  const date = new Date(ts);
  const h = date.getHours().toString().padStart(2, "0");
  const m = date.getMinutes().toString().padStart(2, "0");
  const s = date.getSeconds().toString().padStart(2, "0");
  const ms = date.getMilliseconds().toString().padStart(3, "0");
  return `${h}:${m}:${s}.${ms}`;
}

function JsonViewer({ data }: { data: any }) {
  if (!data) return null;
  
  const formatted = JSON.stringify(data, null, 2);
  
  return (
    <pre className="log-json">
      {formatted}
    </pre>
  );
}

function LogEntryItem({
  log,
  onToggle,
}: {
  log: LogEntry;
  onToggle: (id: number) => void;
}) {
  const hasData = log.data && Object.keys(log.data).length > 0;
  const levelColor = LEVEL_COLORS[log.level];
  const levelIcon = LEVEL_ICONS[log.level];

  return (
    <div className="log-entry" data-level={log.level}>
      <div className="log-header" onClick={() => hasData && onToggle(log.id)}>
        <div className="log-indicator" style={{ background: levelColor }} />
        <span className="log-icon">{levelIcon}</span>
        <span className="log-time">{formatTimestamp(log.timestamp)}</span>
        <span className="log-level" style={{ color: levelColor }}>
          {log.level.toUpperCase()}
        </span>
        <span className="log-message">{log.message}</span>
        {hasData && (
          <span className="log-caret">{log.expanded ? "▾" : "▸"}</span>
        )}
      </div>
      {log.expanded && hasData && (
        <div className="log-data">
          <JsonViewer data={log.data} />
        </div>
      )}
    </div>
  );
}

export function LogsPanel({
  logs,
  enabled,
  filter,
  onToggleEnabled,
  onFilterChange,
  onToggleExpanded,
  onClear,
  onClose,
}: {
  logs: LogEntry[];
  enabled: boolean;
  filter: LogLevel | "all";
  onToggleEnabled: (enabled: boolean) => void;
  onFilterChange: (filter: LogLevel | "all") => void;
  onToggleExpanded: (id: number) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
    setAutoScroll(isAtBottom);
  };

  const filterOptions: Array<{ value: LogLevel | "all"; label: string }> = [
    { value: "all", label: "All" },
    { value: "llm", label: "LLM" },
    { value: "tool", label: "Tools" },
    { value: "system", label: "System" },
    { value: "websocket", label: "WebSocket" },
    { value: "error", label: "Errors" },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="logs-panel" onClick={(e) => e.stopPropagation()}>
        <div className="logs-header">
          <div className="logs-title">
            <span className="logs-icon">📋</span>
            System Logs
            <span className="logs-count">{logs.length}</span>
          </div>
          <div className="logs-controls">
            <label className="logs-toggle">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => onToggleEnabled(e.target.checked)}
              />
              <span>Capture</span>
            </label>
            
            <select
              className="logs-filter"
              value={filter}
              onChange={(e) => onFilterChange(e.target.value as LogLevel | "all")}
            >
              {filterOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>

            <button className="logs-btn" onClick={onClear} title="Clear logs">
              🗑️ Clear
            </button>
            
            <button className="logs-btn" onClick={onClose} title="Close">
              ✕
            </button>
          </div>
        </div>

        <div className="logs-scroll" ref={scrollRef} onScroll={handleScroll}>
          {logs.length === 0 ? (
            <div className="logs-empty">
              <div className="logs-empty-icon">📭</div>
              <div className="logs-empty-text">No logs captured yet</div>
              <div className="logs-empty-hint">
                Interact with the system to see events appear here
              </div>
            </div>
          ) : (
            logs.map((log) => (
              <LogEntryItem
                key={log.id}
                log={log}
                onToggle={onToggleExpanded}
              />
            ))
          )}
        </div>

        {!autoScroll && (
          <button
            className="logs-scroll-btn"
            onClick={() => {
              if (scrollRef.current) {
                scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
                setAutoScroll(true);
              }
            }}
          >
            ↓ Scroll to bottom
          </button>
        )}

        <div className="logs-footer">
          <div className="logs-legend">
            {Object.entries(LEVEL_ICONS).map(([level, icon]) => (
              <div key={level} className="logs-legend-item">
                <span>{icon}</span>
                <span style={{ color: LEVEL_COLORS[level as LogLevel] }}>
                  {level}
                </span>
              </div>
            ))}
          </div>
          <div className="logs-hint">
            Click on entries with data to expand/collapse details
          </div>
        </div>
      </div>
    </div>
  );
}
