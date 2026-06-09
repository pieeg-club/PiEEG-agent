// A tiny reconnecting WebSocket wrapper shared by the live/chat/train hooks.
// Uses same-origin relative paths so it works both behind the Vite dev proxy
// (:5173 → :8000) and when FastAPI serves the built UI at the same origin.

export interface SocketHandle {
  send: (obj: unknown) => boolean;
  close: () => void;
}

export interface SocketCallbacks {
  onMessage: (data: any) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}

export function openSocket(path: string, cb: SocketCallbacks): SocketHandle {
  let ws: WebSocket | null = null;
  let closed = false;
  let retry = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const connect = () => {
    ws = new WebSocket(wsUrl(path));
    ws.onopen = () => {
      retry = 0;
      cb.onOpen?.();
    };
    ws.onmessage = (e) => {
      try {
        cb.onMessage(JSON.parse(e.data));
      } catch {
        /* ignore non-JSON frames */
      }
    };
    ws.onclose = () => {
      cb.onClose?.();
      if (!closed) {
        const delay = Math.min(1000 * 2 ** retry, 8000);
        retry += 1;
        timer = setTimeout(connect, delay);
      }
    };
    ws.onerror = () => ws?.close();
  };

  connect();

  return {
    send: (obj: unknown) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
        return true;
      }
      return false;
    },
    close: () => {
      closed = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    },
  };
}
