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
};
