import { useCallback, useEffect, useRef, useState } from "react";
import { openSocket, type SocketHandle } from "../socket";
import type { ChatWireEvent } from "../types";
import type { useLogsCapture } from "./useLogsCapture";
import {
  saveConversation,
  loadConversation,
  generateConversationId,
  getCurrentConversationId,
  setCurrentConversationId,
  clearCurrentConversationId,
} from "../util/conversations";

export type ToolPart = {
  kind: "tool";
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
  status: "running" | "done";
};
export type TextPart = { kind: "text"; text: string };
export type Part = TextPart | ToolPart;

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  parts: Part[];
  done: boolean;
  error?: boolean;
  usage?: { input_tokens: number; output_tokens: number };
  iterations?: number;
}

// Drives /ws/chat: keeps the message thread, maps the streamed token /
// tool_start / tool_result / done events onto the in-progress assistant turn,
// and serialises sends behind a `busy` flag (the backend handles one turn at a
// time). Tool calls render inline as chips, ChatGPT/Gemini style.
// Now includes conversation persistence to localStorage.
export function useChatSocket(logs?: ReturnType<typeof useLogsCapture>) {
  const [conversationId, setConversationId] = useState<string>(() => {
    return getCurrentConversationId() || generateConversationId();
  });
  
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    // Try loading from saved conversation
    const savedId = getCurrentConversationId();
    if (savedId) {
      const conv = loadConversation(savedId);
      if (conv) return conv.messages;
    }
    // Fallback to sessionStorage for migration
    try {
      const saved = sessionStorage.getItem("pieeg-chat-history");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  
  const [connected, setConnected] = useState(false);
  const [busy, setBusyState] = useState(false);

  const sockRef = useRef<SocketHandle | null>(null);
  const idRef = useRef(1);
  const busyRef = useRef(false);
  const logsRef = useRef(logs);

  // Update logs ref when it changes
  useEffect(() => {
    logsRef.current = logs;
  }, [logs]);

  const setBusy = (v: boolean) => {
    busyRef.current = v;
    setBusyState(v);
  };

  // Auto-save conversation to localStorage when messages change
  useEffect(() => {
    if (messages.length > 0) {
      saveConversation(conversationId, messages);
    }
    // Also keep sessionStorage for backwards compat
    try {
      sessionStorage.setItem("pieeg-chat-history", JSON.stringify(messages));
    } catch {
      // Ignore quota errors
    }
  }, [messages, conversationId]);

  // Track current conversation ID
  useEffect(() => {
    setCurrentConversationId(conversationId);
  }, [conversationId]);

  // Apply a mutation to the current (last) assistant message.
  const mutateLast = useCallback(
    (fn: (m: ChatMessage) => ChatMessage) => {
      setMessages((prev) => {
        if (!prev.length) return prev;
        const last = prev[prev.length - 1];
        if (last.role !== "assistant" || last.done) return prev;
        const copy = prev.slice();
        copy[copy.length - 1] = fn({ ...last, parts: last.parts.slice() });
        return copy;
      });
    },
    []
  );

  const onEvent = useCallback(
    (ev: ChatWireEvent) => {
      // Log the event if logging is enabled
      logsRef.current?.onChatEvent(ev);
      
      switch (ev.type) {
        case "token":
          mutateLast((m) => {
            const last = m.parts[m.parts.length - 1];
            if (last && last.kind === "text") {
              m.parts[m.parts.length - 1] = {
                kind: "text",
                text: last.text + ev.text,
              };
            } else {
              m.parts.push({ kind: "text", text: ev.text });
            }
            return m;
          });
          break;
        case "tool_start":
          mutateLast((m) => {
            m.parts.push({
              kind: "tool",
              name: ev.name,
              args: ev.arguments,
              status: "running",
            });
            return m;
          });
          break;
        case "tool_result":
          mutateLast((m) => {
            for (let i = m.parts.length - 1; i >= 0; i--) {
              const p = m.parts[i];
              if (p.kind === "tool" && p.name === ev.name && p.status === "running") {
                m.parts[i] = { ...p, result: ev.result, status: "done" };
                break;
              }
            }
            return m;
          });
          break;
        case "done":
          mutateLast((m) => ({
            ...m,
            done: true,
            usage: ev.usage,
            iterations: ev.iterations,
          }));
          setBusy(false);
          break;
        case "error":
          mutateLast((m) => {
            m.parts.push({ kind: "text", text: `\n\n⚠️ ${ev.detail}` });
            return { ...m, done: true, error: true };
          });
          setBusy(false);
          break;
        case "reset":
          break;
      }
    },
    [mutateLast]
  );

  useEffect(() => {
    const sock = openSocket("/ws/chat", {
      onMessage: (data) => onEvent(data as ChatWireEvent),
      onOpen: () => {
        setConnected(true);
        logsRef.current?.onWebSocketEvent("open", "/ws/chat");
      },
      onClose: () => {
        setConnected(false);
        logsRef.current?.onWebSocketEvent("close", "/ws/chat");
      },
    });
    sockRef.current = sock;
    return () => sock.close();
  }, [onEvent]);

  const send = useCallback((text: string) => {
    const t = text.trim();
    if (!t || busyRef.current) return;
    const ok = sockRef.current?.send({ message: t });
    
    // Log user message
    logsRef.current?.onUserMessage(t);
    
    setMessages((prev) => [
      ...prev,
      { id: idRef.current++, role: "user", parts: [{ kind: "text", text: t }], done: true },
      { id: idRef.current++, role: "assistant", parts: [], done: false },
    ]);
    if (ok) {
      setBusy(true);
    } else {
      // Socket not open yet — surface it instead of hanging on "busy".
      mutateLast((m) => {
        m.parts.push({ kind: "text", text: "⚠️ not connected to the agent" });
        return { ...m, done: true, error: true };
      });
    }
  }, [mutateLast]);

  const reset = useCallback(() => {
    sockRef.current?.send({ reset: true });
    setMessages([]);
    setBusy(false);
    try {
      sessionStorage.removeItem("pieeg-chat-history");
    } catch {
      // Ignore
    }
  }, []);

  const loadConv = useCallback((id: string) => {
    const conv = loadConversation(id);
    if (conv) {
      setConversationId(id);
      setMessages(conv.messages);
      // Reset agent context when switching conversations
      sockRef.current?.send({ reset: true });
      setBusy(false);
    }
  }, []);

  const newConversation = useCallback(() => {
    const newId = generateConversationId();
    setConversationId(newId);
    setMessages([]);
    // Reset agent context
    sockRef.current?.send({ reset: true });
    setBusy(false);
    clearCurrentConversationId();
    setCurrentConversationId(newId);
  }, []);

  return { 
    messages, 
    connected, 
    busy, 
    send, 
    reset, 
    conversationId,
    loadConversation: loadConv,
    newConversation,
  };
}
