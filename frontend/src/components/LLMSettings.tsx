import { useEffect, useState } from "react";
import type { Info } from "../types";
import { ChromeAI } from "../util/chromeai";

const PROVIDERS = [
  { id: "anthropic", name: "Anthropic", requiresKey: true },
  { id: "openai", name: "OpenAI", requiresKey: true },
  { id: "groq", name: "Groq", requiresKey: true },
  { id: "together", name: "Together AI", requiresKey: true },
  { id: "ollama", name: "Ollama (Local)", requiresKey: false },
  { id: "lmstudio", name: "LM Studio (Local)", requiresKey: false },
  { id: "echo", name: "Echo (Debug)", requiresKey: false },
];

const MODELS: Record<string, string[]> = {
  anthropic: [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
  ],
  openai: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
  groq: ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
  ollama: ["llama3.2", "llama3.1", "mistral", "codellama"],
  lmstudio: ["custom-model"],
  together: ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
  echo: ["debug"],
};

interface LLMSettingsProps {
  info: Info | null;
  onClose: () => void;
}

export function LLMSettings({ info, onClose }: LLMSettingsProps) {
  const [provider, setProvider] = useState(info?.provider || "anthropic");
  const [model, setModel] = useState(info?.model || "");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [chromeAIStatus, setChromeAIStatus] = useState<{
    available: boolean;
    status: string;
    message: string;
  } | null>(null);

  const selectedProvider = PROVIDERS.find((p) => p.id === provider);
  const availableModels = MODELS[provider] || [];

  // Check Chrome AI availability on mount
  useEffect(() => {
    ChromeAI.getStatus().then(setChromeAIStatus);
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);

    try {
      const res = await fetch("/api/llm/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          model: model || undefined,
          api_key: apiKey || undefined,
        }),
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Failed to save configuration");
      }

      setMessage({ type: "success", text: "Configuration saved. Restart required." });
      setTimeout(() => {
        onClose();
      }, 2000);
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "Failed to save",
      });
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal llm-settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>LLM Configuration</h2>
          <button className="x" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="modal-body">
          <div className="settings-section">
            <label className="settings-label">
              Provider
              <span className="settings-hint">Choose your LLM backend</span>
            </label>
            <div className="provider-grid">
              {PROVIDERS.map((p) => (
                <button
                  key={p.id}
                  className={`provider-card ${provider === p.id ? "active" : ""}`}
                  onClick={() => {
                    setProvider(p.id);
                    setModel("");
                  }}
                >
                  <span className="provider-name">{p.name}</span>
                  {!p.requiresKey && <span className="provider-badge">No key</span>}
                </button>
              ))}
            </div>
          </div>

          <div className="settings-section">
            <label className="settings-label">
              Model
              <span className="settings-hint">
                {availableModels.length > 0 ? "Select or enter custom model ID" : "Enter model ID"}
              </span>
            </label>
            {availableModels.length > 0 ? (
              <select
                className="settings-input"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                <option value="">Default for provider</option>
                {availableModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                className="settings-input"
                placeholder="Leave empty for provider default"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            )}
          </div>

          {chromeAIStatus && (
            <div className="settings-section">
              <label className="settings-label">
                Chrome AI Fallback
                <span className="settings-hint">
                  Offline fallback when backend is rate-limited
                </span>
              </label>
              <div className={`chrome-ai-status ${chromeAIStatus.available ? "available" : "unavailable"}`}>
                <div className="status-indicator">
                  <span className={`status-dot ${chromeAIStatus.status}`} />
                  <span className="status-text">{chromeAIStatus.message}</span>
                </div>
                {!chromeAIStatus.available && chromeAIStatus.status === "unavailable" && (
                  <div className="settings-note">
                    💡 Enable Chrome AI: Open <code>chrome://flags/#prompt-api-for-gemini-nano</code>,
                    set to "Enabled", and restart Chrome. Requires Chrome 127+ on supported devices.
                  </div>
                )}
                {chromeAIStatus.status === "download-needed" && (
                  <div className="settings-note">
                    ✨ Model will download automatically on first fallback use (~1.5 GB)
                  </div>
                )}
                {chromeAIStatus.available && chromeAIStatus.status === "ready" && (
                  <div className="settings-note success">
                    ✅ Chrome AI will automatically respond when backend is rate-limited
                  </div>
                )}
              </div>
            </div>
          )}

          {selectedProvider?.requiresKey && (
            <div className="settings-section">
              <label className="settings-label">
                API Key
                <span className="settings-hint">
                  Set via environment variable for persistent config
                </span>
              </label>
              <input
                type="password"
                className="settings-input"
                placeholder={`${provider.toUpperCase()}_API_KEY or ANTHROPIC_API_KEY`}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <div className="settings-note">
                💡 For permanent configuration, set environment variables:
                <code>PIEEG_LLM_PROVIDER</code>, <code>PIEEG_LLM_MODEL</code>, and{" "}
                <code>{provider.toUpperCase()}_API_KEY</code>
              </div>
            </div>
          )}

          {message && (
            <div className={`settings-message ${message.type}`}>{message.text}</div>
          )}

          <div className="modal-actions">
            <button className="btn-secondary" onClick={onClose} disabled={saving}>
              Cancel
            </button>
            <button className="btn-primary" onClick={handleSave} disabled={saving}>
              {saving ? "Saving..." : "Save Configuration"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
