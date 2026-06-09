import { useCallback, useEffect, useRef, useState } from "react";
import { openSocket, type SocketHandle } from "../socket";
import type { TrainReply, TrainResult } from "../types";

// Drives /ws/train as a simple request/response channel. The backend replies
// with exactly one {action, result} per command, in order, so a FIFO queue of
// resolvers is enough to turn each send into an awaitable promise. `record`
// blocks server-side for its duration while live frames stream in.
export function useTrainSocket() {
  const [connected, setConnected] = useState(false);
  const sockRef = useRef<SocketHandle | null>(null);
  const pending = useRef<Array<(r: TrainResult) => void>>([]);

  useEffect(() => {
    const sock = openSocket("/ws/train", {
      onMessage: (data) => {
        const reply = data as TrainReply;
        const resolve = pending.current.shift();
        resolve?.(reply.result ?? {});
      },
      onOpen: () => setConnected(true),
      onClose: () => {
        setConnected(false);
        const drained = pending.current;
        pending.current = [];
        drained.forEach((p) => p({ error: "disconnected from agent" }));
      },
    });
    sockRef.current = sock;
    return () => sock.close();
  }, []);

  const request = useCallback((payload: Record<string, unknown>) => {
    return new Promise<TrainResult>((resolve) => {
      const ok = sockRef.current?.send(payload);
      if (!ok) {
        resolve({ error: "not connected to agent" });
        return;
      }
      pending.current.push(resolve);
    });
  }, []);

  return {
    connected,
    begin: (name: string) => request({ action: "begin", name }),
    record: (label: string, seconds: number) =>
      request({ action: "record", label, seconds }),
    finish: (threshold: number) => request({ action: "finish", threshold }),
    cancel: () => request({ action: "cancel" }),
  };
}
