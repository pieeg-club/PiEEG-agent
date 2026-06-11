import type { Info } from "../types";

// Top bar: brand, the live session facts (stream / channels / rate / model)
// and a connection dot reflecting the /ws/live socket.
export function Header({
  info,
  connected,
  onSettings,
  onSystem,
}: {
  info: Info | null;
  connected: boolean;
  onSettings: () => void;
  onSystem: () => void;
}) {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark">◉</span>
        <span className="brand-name">PiEEG</span>
        <span className="brand-sub">Agent</span>
      </div>
      <div className="header-meta">
        {info?.stream && <span className="chip">{info.stream}</span>}
        {info?.channels != null && <span className="chip">{info.channels} ch</span>}
        {info?.rate != null && <span className="chip">{info.rate} Hz</span>}
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
