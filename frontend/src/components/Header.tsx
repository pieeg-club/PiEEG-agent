import type { Info } from "../types";

// Top bar: brand, the live session facts (stream / channels / rate / model)
// and a connection dot reflecting the /ws/live socket.
export function Header({
  info,
  connected,
  onSettings,
  onSystem,
  onLogs,
  onHistory,
  sidebarOpen,
}: {
  info: Info | null;
  connected: boolean;
  onSettings: () => void;
  onSystem: () => void;
  onLogs: () => void;
  onHistory?: () => void;
  sidebarOpen?: boolean;
}) {
  return (
    <header className="app-header">
      <div className="brand">
        {onHistory && (
          <button className="menu-btn" onClick={onHistory} title={sidebarOpen ? "Close sidebar" : "Open sidebar"}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              {sidebarOpen ? (
                <>
                  <rect x="3" y="3" width="7" height="18" rx="1"></rect>
                  <path d="M14 4l-4 8 4 8"></path>
                </>
              ) : (
                <>
                  <rect x="3" y="3" width="7" height="18" rx="1"></rect>
                  <path d="M14 4l4 8-4 8"></path>
                </>
              )}
            </svg>
          </button>
        )}
        <span className="brand-name">PiEEG</span>
        <span className="brand-sub">Agent</span>
      </div>
      <div className="header-meta">
        {info?.stream && <span className="chip">{info.stream}</span>}
        {info?.channels != null && <span className="chip">{info.channels} ch</span>}
        {info?.rate != null && <span className="chip">{info.rate} Hz</span>}
        <button className="chip chip-clickable" onClick={onLogs} title="System logs & debug">
          Logs
          <span className="chip-icon">📋</span>
        </button>
        <button className="chip chip-clickable" onClick={onSystem} title="System control">
          System
          <span className="chip-icon">🎛</span>
        </button>
        {info?.provider && (
          <button className="chip chip-accent chip-clickable" onClick={onSettings}>
            {info.provider}
            {info.model ? ` · ${info.model}` : ""}
            <span className="chip-icon">⚙</span>
          </button>
        )}
        {info?.control && (
          <span className="chip chip-warn" title="The copilot has gated control tools">
            control
          </span>
        )}
        <span
          className={"live-dot " + (connected ? "on" : "off")}
          title={connected ? "live telemetry connected" : "reconnecting…"}
        />
      </div>
    </header>
  );
}
