import type { Info } from "../types";

// Top bar: brand, the live session facts (stream / channels / rate / model)
// and a connection dot reflecting the /ws/live socket.
export function Header({ info, connected }: { info: Info | null; connected: boolean }) {
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
        {info?.provider && (
          <span className="chip chip-accent">
            {info.provider}
            {info.model ? ` · ${info.model}` : ""}
          </span>
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
