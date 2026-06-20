import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import useTTS from "../hooks/useTTS";
import type { CatalogModel, Info, ModelCatalog } from "../types";

const PROVIDERS = [
  { id: "openrouter", name: "OpenRouter", requiresKey: true, recommended: true },
  { id: "anthropic", name: "Anthropic", requiresKey: true },
  { id: "openai", name: "OpenAI", requiresKey: true },
  { id: "groq", name: "Groq", requiresKey: true },
  { id: "together", name: "Together AI", requiresKey: true },
  { id: "ollama", name: "Ollama (Local)", requiresKey: false },
  { id: "lmstudio", name: "LM Studio (Local)", requiresKey: false },
  { id: "echo", name: "Echo (Debug)", requiresKey: false },
];

// Per-token prices → friendly "$X/M tokens"; null when unknown/free.
function formatPrice(perToken?: string | null): string | null {
  if (perToken == null) return null;
  const n = Number(perToken);
  if (!isFinite(n)) return null;
  if (n === 0) return "free";
  const perM = n * 1_000_000;
  return `$${perM >= 1 ? perM.toFixed(2) : perM.toPrecision(2)}/M`;
}

function formatContext(n?: number | null): string | null {
  if (!n) return null;
  return n >= 1000 ? `${Math.round(n / 1000)}K ctx` : `${n} ctx`;
}

function vendorOf(id: string): string {
  const i = id.indexOf("/");
  return i === -1 ? "other" : id.slice(0, i);
}

interface ModelPickerProps {
  models: CatalogModel[];
  value: string;
  onChange: (id: string) => void;
}

// Searchable, grouped combobox over the live OpenRouter catalog. Defaults to
// showing only tool-capable models because the agent relies on tool calls.
function ModelPicker({ models, value, onChange }: ModelPickerProps) {
  const [query, setQuery] = useState("");
  const [toolsOnly, setToolsOnly] = useState(true);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = models.filter((m) => {
      if (toolsOnly && !m.supports_tools) return false;
      if (!q) return true;
      return m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q);
    });
    const byVendor = new Map<string, CatalogModel[]>();
    for (const m of filtered) {
      const v = vendorOf(m.id);
      (byVendor.get(v) ?? byVendor.set(v, []).get(v)!).push(m);
    }
    return [...byVendor.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [models, query, toolsOnly]);

  const total = groups.reduce((n, [, list]) => n + list.length, 0);

  return (
    <div className="model-picker">
      <div className="model-picker-bar">
        <input
          type="text"
          className="settings-input"
          placeholder="Search models (e.g. claude, gpt, gemini)…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="model-picker-toggle">
          <input
            type="checkbox"
            checked={toolsOnly}
            onChange={(e) => setToolsOnly(e.target.checked)}
          />
          Tools only
        </label>
      </div>

      {value && (
        <div className="model-picker-selected">
          Selected: <code>{value}</code>
        </div>
      )}

      <div className="model-picker-list">
        {total === 0 && <div className="model-picker-empty">No matching models.</div>}
        {groups.map(([vendor, list]) => (
          <div key={vendor} className="model-picker-group">
            <div className="model-picker-group-head">{vendor}</div>
            {list.map((m) => {
              const ctx = formatContext(m.context_length);
              const inPrice = formatPrice(m.prompt_price);
              const outPrice = formatPrice(m.completion_price);
              return (
                <button
                  key={m.id}
                  type="button"
                  className={`model-picker-row ${value === m.id ? "active" : ""}`}
                  onClick={() => onChange(m.id)}
                  title={m.id}
                >
                  <span className="model-picker-name">{m.name}</span>
                  <span className="model-picker-badges">
                    {ctx && <span className="model-badge">{ctx}</span>}
                    {inPrice && <span className="model-badge">in {inPrice}</span>}
                    {outPrice && <span className="model-badge">out {outPrice}</span>}
                    {!m.supports_tools && <span className="model-badge warn">no tools</span>}
                  </span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

interface LLMSettingsProps {
  info: Info | null;
  onClose: () => void;
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

export function LLMSettings({ info, onClose }: LLMSettingsProps) {
  const [provider, setProvider] = useState(info?.provider || "openrouter");
  const [model, setModel] = useState(info?.model || "");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .models()
      .then((c) => alive && setCatalog(c))
      .catch(() => alive && setCatalog({ source: "", fetched_at: null, count: 0, models: [] }));
    return () => {
      alive = false;
    };
  }, []);

  const selectedProvider = PROVIDERS.find((p) => p.id === provider);
  const useCatalog = provider === "openrouter" && (catalog?.models.length ?? 0) > 0;

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

      setMessage({ type: "success", text: "Configuration saved and applied!" });
      setTimeout(() => {
        onClose();
        window.location.reload();  // Refresh to show new provider/model in header
      }, 1500);
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
                  {p.recommended && <span className="provider-badge rec">Recommended</span>}
                  {!p.requiresKey && <span className="provider-badge">No key</span>}
                </button>
              ))}
            </div>
          </div>

          <div className="settings-section">
            <label className="settings-label">
              Model
              <span className="settings-hint">
                {useCatalog
                  ? "Search the live OpenRouter catalog"
                  : "Enter a model ID (leave empty for the provider default)"}
              </span>
            </label>
            {useCatalog ? (
              <>
                <ModelPicker models={catalog!.models} value={model} onChange={setModel} />
                {catalog?.fetched_at && (
                  <div className="settings-note">
                    Catalog: {catalog.count} models · updated{" "}
                    {new Date(catalog.fetched_at).toLocaleDateString()}
                  </div>
                )}
              </>
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
