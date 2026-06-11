import type { Info, PatternExplain, PatternList, Snapshot } from "./types";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { accept: "application/json" } });
  if (!res.ok && res.status !== 404) {
    throw new Error(`${path} → ${res.status}`);
  }
  return (await res.json()) as T;
}

async function delJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { method: "DELETE" });
  return (await res.json()) as T;
}

async function postJSON<T>(path: string, body?: any): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return (await res.json()) as T;
}

// One-shot REST reads. The live surfaces use WebSockets instead (see hooks/).
export const api = {
  info: () => getJSON<Info>("/api/info"),
  state: () => getJSON<Snapshot>("/api/state"),
  patterns: () => getJSON<PatternList>("/api/patterns"),
  explain: (name: string) =>
    getJSON<PatternExplain>(`/api/patterns/${encodeURIComponent(name)}`),
  forget: (name: string) =>
    delJSON<{ status?: string; error?: string }>(
      `/api/patterns/${encodeURIComponent(name)}`
    ),
  
  // LSL streams discovery
  streams: (wait: number = 2.0) =>
    getJSON<{ streams?: any[]; error?: string }>(`/api/streams?wait=${wait}`),
  
  // Server control
  serverStatus: () =>
    getJSON<any>("/api/server/status"),
  serverFilter: (enabled: boolean, lowcut: number, highcut: number) =>
    postJSON<any>("/api/server/filter", { enabled, lowcut, highcut }),
  serverRecord: (action: "start" | "stop") =>
    postJSON<any>("/api/server/record", { action }),
  serverOsc: (action: "start" | "stop", config?: any) =>
    postJSON<any>("/api/server/osc", { action, config }),
  serverLsl: (action: "start" | "stop", config?: any) =>
    postJSON<any>("/api/server/lsl", { action, config }),
  serverRegisterPreset: (preset: string) =>
    postJSON<any>("/api/server/register-preset", { preset }),
  serverWebhooks: () =>
    getJSON<any>("/api/server/webhooks"),
};
