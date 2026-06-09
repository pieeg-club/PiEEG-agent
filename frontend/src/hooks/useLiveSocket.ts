import { useEffect, useRef, useState } from "react";
import { openSocket, type SocketHandle } from "../socket";
import type { Snapshot } from "../types";

// Subscribes to /ws/live and exposes the latest brain snapshot plus the
// connection state. The backend pushes a full snapshot a few times a second.
export function useLiveSocket() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const sockRef = useRef<SocketHandle | null>(null);

  useEffect(() => {
    const sock = openSocket("/ws/live", {
      onMessage: (data) => setSnapshot(data as Snapshot),
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
    });
    sockRef.current = sock;
    return () => sock.close();
  }, []);

  return { snapshot, connected };
}
