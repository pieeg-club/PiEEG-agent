import { useCallback, useRef, useState } from "react";
import type { ChatWireEvent } from "../types";

export type LogLevel = "llm" | "tool" | "system" | "error" | "websocket";

export interface LogEntry {
  id: number;
  timestamp: number;
  level: LogLevel;
  message: string;
  data?: any;
  expanded?: boolean;
}

interface LogEntryBuilder {
  id: number;
  timestamp: number;
  level: LogLevel;
  message: string;
  data?: any;
  toolName?: string;
  isToolStart?: boolean;
}

// Hook to capture all chat events, tool calls, and system events for debugging
export function useLogsCapture() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [enabled, setEnabled] = useState(true);
  const [filter, setFilter] = useState<LogLevel | "all">("all");
  const idRef = useRef(1);
  const currentRequestRef = useRef<{ userMessage?: string; tools?: string[] } | null>(null);

  const addLog = useCallback(
    (level: LogLevel, message: string, data?: any) => {
      if (!enabled) return;
      
      const entry: LogEntry = {
        id: idRef.current++,
        timestamp: Date.now(),
        level,
        message,
        data,
        expanded: false,
      };

      setLogs((prev) => [...prev, entry]);
    },
    [enabled]
  );

  const onChatEvent = useCallback(
    (ev: ChatWireEvent, userMessage?: string) => {
      switch (ev.type) {
        case "token":
          // Don't log every token to avoid spam
          break;

        case "tool_start":
          addLog("tool", `Tool Started: ${ev.name}`, {
            name: ev.name,
            arguments: ev.arguments,
            status: "running",
          });
          
          // Track tools for the current request
          if (currentRequestRef.current) {
            if (!currentRequestRef.current.tools) {
              currentRequestRef.current.tools = [];
            }
            currentRequestRef.current.tools.push(ev.name);
          }
          break;

        case "tool_result":
          addLog("tool", `Tool Result: ${ev.name}`, {
            name: ev.name,
            result: ev.result,
            status: "done",
          });
          break;

        case "done": {
          const contextData: any = {
            usage: ev.usage,
            iterations: ev.iterations,
            tool_calls: ev.tool_calls,
          };

          // Include user message if this is the completion of a request
          if (currentRequestRef.current?.userMessage) {
            contextData.user_message = currentRequestRef.current.userMessage;
            contextData.tools_used = currentRequestRef.current.tools || [];
          }

          addLog("llm", "LLM Request Complete", contextData);
          
          // Clear current request
          currentRequestRef.current = null;
          break;
        }

        case "error":
          addLog("error", `Chat Error: ${ev.detail}`, { detail: ev.detail });
          currentRequestRef.current = null;
          break;

        case "reset":
          addLog("system", "Chat Reset", {});
          currentRequestRef.current = null;
          break;
      }
    },
    [addLog]
  );

  const onUserMessage = useCallback((message: string) => {
    // Track the start of a new request
    currentRequestRef.current = {
      userMessage: message,
      tools: [],
    };
    
    addLog("llm", "User Query Sent", { 
      message,
      note: "Watch for tool calls and completion below"
    });
  }, [addLog]);

  const onWebSocketEvent = useCallback(
    (event: "open" | "close" | "error", path: string, detail?: any) => {
      addLog("websocket", `WebSocket ${event}: ${path}`, detail);
    },
    [addLog]
  );

  const toggleExpanded = useCallback((id: number) => {
    setLogs((prev) =>
      prev.map((log) =>
        log.id === id ? { ...log, expanded: !log.expanded } : log
      )
    );
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    idRef.current = 1;
  }, []);

  const filteredLogs = filter === "all" 
    ? logs 
    : logs.filter((log) => log.level === filter);

  return {
    logs: filteredLogs,
    allLogs: logs,
    enabled,
    filter,
    setEnabled,
    setFilter,
    addLog,
    onChatEvent,
    onUserMessage,
    onWebSocketEvent,
    toggleExpanded,
    clearLogs,
  };
}
