import { useEffect, useState } from "react";

interface ToastMessage {
  id: number;
  text: string;
  type: "success" | "error" | "info";
}

let toastId = 1;
let listeners: ((msg: ToastMessage) => void)[] = [];

export const toast = {
  success: (text: string) => {
    const msg = { id: toastId++, text, type: "success" as const };
    listeners.forEach((fn) => fn(msg));
  },
  error: (text: string) => {
    const msg = { id: toastId++, text, type: "error" as const };
    listeners.forEach((fn) => fn(msg));
  },
  info: (text: string) => {
    const msg = { id: toastId++, text, type: "info" as const };
    listeners.forEach((fn) => fn(msg));
  },
};

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  useEffect(() => {
    const listener = (msg: ToastMessage) => {
      setToasts((prev) => [...prev, msg]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== msg.id));
      }, 3000);
    };
    listeners.push(listener);
    return () => {
      listeners = listeners.filter((l) => l !== listener);
    };
  }, []);

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          {t.text}
        </div>
      ))}
    </div>
  );
}
