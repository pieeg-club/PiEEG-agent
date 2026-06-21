import { useEffect, useState } from "react";
import useTTS from "../hooks/useTTS";
import type { Info } from "../types";

// Fallback providers list if backend doesn't send availableProviders
const FALLBACK_PROVIDERS = [
  { id: "anthropic", name: "Anthropic", requiresKey: true, defaultModel: "claude-sonnet-4-6" },
  { id: "openai", name: "OpenAI", requiresKey: true, defaultModel: "gpt-5.4-mini" },
  { id: "groq", name: "Groq", requiresKey: true, defaultModel: "llama-3.3-70b-versatile" },
  { id: "together", name: "Together AI", requiresKey: true, defaultModel: "meta-llama/Llama-3.3-70B-Instruct-Turbo" },
  { id: "ollama", name: "Ollama (Local)", requiresKey: false, defaultModel: "llama3.1" },
  { id: "lmstudio", name: "LM Studio (Local)", requiresKey: false, defaultModel: "local-model" },
  { id: "echo", name: "Echo (Debug)", requiresKey: false, defaultModel: "echo-debug-v1" },
];

// Common models per provider - used for quick model selection dropdown
const COMMON_MODELS: Record<string, string[]> = {
  anthropic: [
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "claude-haiku-4-5",
    "claude-fable-5",
    // Legacy models for backwards compat
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
  ],
  openai: [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    // Legacy models for backwards compat
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
  ],
  groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "mixtral-8x7b-32768"],
  ollama: ["llama3.2", "llama3.1", "mistral", "codellama"],
  lmstudio: ["custom-model"],
  together: ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.1-8B-Instruct-Turbo"],
  echo: ["debug"],
};

interface LLMSettingsProps {
  info: Info | null;
  onClose: () => void;
  onSaved?: () => void;
}

function TTSControl() {
  const tts = useTTS();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <input
          type="checkbox"
          checked={tts.enabled}
          onChange={(e) => tts.setEnabled(e.target.checked)}
        />
        <span>{tts.isSupported ? (tts.enabled ? "Enabled" : "Disabled") : "Not supported"}</span>
      </label>
      {!tts.isSupported && <div style={{ color: "#888" }}>Browser does not support SpeechSynthesis.</div>}
    </div>
  );
}

export function LLMSettings({ info, onClose, onSaved }: LLMSettingsProps) {
  const providers = info?.availableProviders || FALLBACK_PROVIDERS;
  const [provider, setProvider] = useState(info?.provider || "anthropic");
  const [model, setModel] = useState(info?.model || "");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const selectedProvider = providers.find((p) => p.id === provider);
  const availableModels = COMMON_MODELS[provider] || [];

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
      onSaved?.();
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
              {providers.map((p) => (
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

          <div className="settings-section">
            <label className="settings-label">
              Text-to-speech
              <span className="settings-hint">Play assistant replies aloud in the browser</span>
            </label>
            <TTSControl />
          </div>

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
