import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "./Toast";

interface Stream {
  name: string;
  type: string;
  channels: number;
  rate: number;
  source_id: string;
}

interface ServerStatus {
  sample_rate?: number;
  num_channels?: number;
  filter?: any;
  recording?: boolean;
  error?: string;
}

export function SystemControl({ onClose }: { onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<"streams" | "server">("streams");
  const [streams, setStreams] = useState<Stream[] | null>(null);
  const [discovering, setDiscovering] = useState(false);
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [filterEnabled, setFilterEnabled] = useState(true);
  const [lowcut, setLowcut] = useState(1.0);
  const [highcut, setHighcut] = useState(40.0);

  useEffect(() => {
    if (activeTab === "server") {
      loadServerStatus();
    }
  }, [activeTab]);

  const discoverStreams = async () => {
    setDiscovering(true);
    try {
      const result = await api.streams(2.0);
      if (result.error) {
        toast.error(result.error);
        setStreams([]);
      } else {
        setStreams(result.streams || []);
        toast.success(`Found ${result.streams?.length || 0} stream(s)`);
      }
    } catch (err) {
      toast.error("Failed to discover streams");
      setStreams([]);
    } finally {
      setDiscovering(false);
    }
  };

  const loadServerStatus = async () => {
    try {
      const status = await api.serverStatus();
      setServerStatus(status);
    } catch (err) {
      setServerStatus({ error: "Not connected" });
    }
  };

  const applyFilter = async () => {
    try {
      const result = await api.serverFilter(filterEnabled, lowcut, highcut);
      if (result.error) {
        toast.error(result.error);
      } else {
        toast.success("Filter updated");
        loadServerStatus();
      }
    } catch (err) {
      toast.error("Failed to update filter");
    }
  };

  const toggleRecording = async (action: "start" | "stop") => {
    try {
      const result = await api.serverRecord(action);
      if (result.error) {
        toast.error(result.error);
      } else {
        toast.success(`Recording ${action}ed`);
        loadServerStatus();
      }
    } catch (err) {
      toast.error(`Failed to ${action} recording`);
    }
  };

  const applyPreset = async (preset: string) => {
    try {
      const result = await api.serverRegisterPreset(preset);
      if (result.error) {
        toast.error(result.error);
      } else {
        toast.success(`Preset "${preset}" applied`);
      }
    } catch (err) {
      toast.error("Failed to apply preset");
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal system-control" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>System Control</h2>
          <button onClick={onClose} className="close" aria-label="Close">
            ×
          </button>
        </header>

        <div className="tabs">
          <button
            className={activeTab === "streams" ? "active" : ""}
            onClick={() => setActiveTab("streams")}
          >
            LSL Streams
          </button>
          <button
            className={activeTab === "server" ? "active" : ""}
            onClick={() => setActiveTab("server")}
          >
            Server Control
          </button>
        </div>

        <div className="tab-content">
          {activeTab === "streams" && (
            <div className="streams-panel">
              <div className="panel-header">
                <p>Discover Lab Streaming Layer (LSL) streams on the network.</p>
                <button onClick={discoverStreams} disabled={discovering}>
                  {discovering ? "Discovering..." : "Discover Streams"}
                </button>
              </div>

              {streams && streams.length === 0 && (
                <p className="empty-state">
                  No streams found. Make sure PiEEG-server is running with --lsl.
                </p>
              )}

              {streams && streams.length > 0 && (
                <table className="streams-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Channels</th>
                      <th>Rate</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {streams.map((s, i) => (
                      <tr key={i}>
                        <td>{s.name}</td>
                        <td>{s.type}</td>
                        <td>{s.channels}</td>
                        <td>{s.rate > 0 ? `${s.rate} Hz` : "irregular"}</td>
                        <td className="source">{s.source_id}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {activeTab === "server" && (
            <div className="server-panel">
              {serverStatus?.error ? (
                <div className="error-state">
                  <p>{serverStatus.error}</p>
                  <p className="hint">
                    Start the web server with <code>--allow-actions</code> to enable
                    device control.
                  </p>
                </div>
              ) : (
                <>
                  <section className="control-section">
                    <h3>Server Status</h3>
                    {serverStatus && (
                      <dl className="status-grid">
                        <dt>Sample Rate:</dt>
                        <dd>{serverStatus.sample_rate || "—"} Hz</dd>
                        <dt>Channels:</dt>
                        <dd>{serverStatus.num_channels || "—"}</dd>
                        <dt>Recording:</dt>
                        <dd>{serverStatus.recording ? "Active" : "Stopped"}</dd>
                      </dl>
                    )}
                    <button onClick={loadServerStatus} className="secondary">
                      Refresh Status
                    </button>
                  </section>

                  <section className="control-section">
                    <h3>Band-Pass Filter</h3>
                    <div className="filter-controls">
                      <label>
                        <input
                          type="checkbox"
                          checked={filterEnabled}
                          onChange={(e) => setFilterEnabled(e.target.checked)}
                        />
                        <span>Enabled</span>
                      </label>
                      <label>
                        Lowcut (Hz):
                        <input
                          type="number"
                          value={lowcut}
                          onChange={(e) => setLowcut(parseFloat(e.target.value))}
                          min="0.1"
                          max="125"
                          step="0.1"
                        />
                      </label>
                      <label>
                        Highcut (Hz):
                        <input
                          type="number"
                          value={highcut}
                          onChange={(e) => setHighcut(parseFloat(e.target.value))}
                          min="0.1"
                          max="125"
                          step="0.1"
                        />
                      </label>
                      <button onClick={applyFilter}>Apply Filter</button>
                    </div>
                  </section>

                  <section className="control-section">
                    <h3>Recording</h3>
                    <div className="button-group">
                      <button onClick={() => toggleRecording("start")}>
                        Start Recording
                      </button>
                      <button onClick={() => toggleRecording("stop")} className="secondary">
                        Stop Recording
                      </button>
                    </div>
                  </section>

                  <section className="control-section">
                    <h3>ADS1299 Register Presets</h3>
                    <div className="button-group">
                      <button onClick={() => applyPreset("normal")}>Normal</button>
                      <button onClick={() => applyPreset("internal_short")}>
                        Internal Short
                      </button>
                      <button onClick={() => applyPreset("test_signal")}>
                        Test Signal
                      </button>
                      <button onClick={() => applyPreset("temp_sensor")}>
                        Temp Sensor
                      </button>
                    </div>
                  </section>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
