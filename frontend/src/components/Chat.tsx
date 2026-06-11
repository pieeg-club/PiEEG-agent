import { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import type { ChatMessage, Part, ToolPart } from "../hooks/useChatSocket";
import { toast } from "./Toast";

const SUGGESTIONS = [
  "What is my brain doing right now?",
  "How is my signal quality?",
  "Which channels look bad?",
  "Let's train a new pattern together",
];

function ToolChip({ part }: { part: ToolPart }) {
  const [open, setOpen] = useState(false);
  const body = JSON.stringify(
    { arguments: part.args, result: part.result },
    null,
    2
  );
  return (
    <div className={"tool-chip " + part.status}>
      <button className="tool-head" onClick={() => setOpen((o) => !o)}>
        <span className="tool-dot" />
        <span className="tool-name">{part.name}</span>
        <span className="tool-state">{part.status === "running" ? "running…" : "done"}</span>
        <span className="tool-caret">{open ? "▾" : "▸"}</span>
      </button>
      {open && <pre className="tool-body">{body}</pre>}
    </div>
  );
}

function renderPart(p: Part, i: number, msgId: string) {
  if (p.kind === "text") {
    const html = marked.parse(p.text, { async: false }) as string;
    return (
      <div className="text" key={`${msgId}-text-${i}`} dangerouslySetInnerHTML={{ __html: html }} />
    );
  }
  return <ToolChip part={p} key={`${msgId}-tool-${i}-${p.name}`} />;
}

// A clickable suggestion the agent surfaces mid-flow (e.g. guided training).
// Sending the message text drives the exact same copilot turn a typed reply
// would, so the buttons are pure convenience — the agent stays in control.
type QuickReply = { label: string; message: string; tone?: "rest" | "active" | "finish" | "cancel" };

// Derive the live pattern-training context purely from the streamed tool
// results already in the thread (no extra backend protocol). Each training
// turn records at most one segment, so we replay the whole conversation to
// find: is a session open, what was recorded last, and is it ready to finish.
function deriveTrainingActions(messages: ChatMessage[]): QuickReply[] {
  let active = false;
  let lastLabel: string | null = null;
  let reps = 0;
  let ready = false;
  let mustFinish = false;

  for (const m of messages) {
    for (const p of m.parts) {
      if (p.kind !== "tool") continue;
      const r = (p.result ?? {}) as Record<string, unknown>;
      const status = r.status as string | undefined;
      if (p.name === "start_pattern_training") {
        if (status === "training_started") {
          active = true;
          lastLabel = null;
          reps = 0;
          ready = false;
          mustFinish = false;
        }
      } else if (p.name === "record_segment") {
        if (status === "segment_recorded") {
          lastLabel = (r.label as string) ?? lastLabel;
          const totals = r.totals as { reps?: number } | undefined;
          reps = totals?.reps ?? reps;
          ready = Boolean(r.ready);
          mustFinish = Boolean(r.must_finish);
        }
      } else if (p.name === "finish_pattern_training" || p.name === "cancel_pattern_training") {
        active = false;
      }
    }
  }

  if (!active) return [];

  if (mustFinish) {
    return [
      { label: "Finish & train detector", message: "Finish the training now and fit the detector.", tone: "finish" },
      { label: "Cancel", message: "Cancel the pattern training.", tone: "cancel" },
    ];
  }

  const nextLabel = lastLabel === "rest" ? "active" : "rest";
  const actions: QuickReply[] =
    nextLabel === "rest"
      ? [{ label: "Record rest — I'm relaxed", message: "I'm relaxed and ready — record the rest segment now.", tone: "rest" }]
      : [{ label: "Record active — I'm doing it", message: "I'm doing it now — record the active segment.", tone: "active" }];

  if (ready || reps >= 2) {
    actions.push({ label: "Finish & train", message: "Finish the training now and fit the detector.", tone: "finish" });
  }
  actions.push({ label: "Cancel", message: "Cancel the pattern training.", tone: "cancel" });
  return actions;
}


function Bubble({ m }: { m: ChatMessage }) {
  const empty = m.parts.length === 0;
  
  const copyMessage = () => {
    const textParts = m.parts.filter((p) => p.kind === "text").map((p) => (p as { text: string }).text);
    const text = textParts.join("\n");
    navigator.clipboard.writeText(text).then(() => {
      toast.success("Message copied");
    });
  };

  return (
    <div className={"msg " + m.role}>
      <div className="avatar">{m.role === "user" ? "You" : "Agent"}</div>
      <div className={"bubble" + (m.error ? " error" : "")}>
        {m.parts.map((p, i) => renderPart(p, i, m.id))}
        {m.role === "assistant" && empty && !m.done && (
          <span className="typing">
            <i />
            <i />
            <i />
          </span>
        )}
        {m.role === "assistant" && !empty && !m.done && <span className="cursor" />}
        {m.usage && (
          <div className="usage">
            {m.usage.input_tokens}→{m.usage.output_tokens} tok
            {m.iterations ? ` · ${m.iterations} step${m.iterations > 1 ? "s" : ""}` : ""}
          </div>
        )}
        {m.role === "assistant" && m.done && !empty && (
          <button className="copy-btn" onClick={copyMessage} title="Copy message">
            📋
          </button>
        )}
      </div>
    </div>
  );
}

// The conversation surface: a streaming thread plus a composer. Enter sends,
// Shift+Enter inserts a newline. Empty-state shows clickable starter prompts.
export function Chat({
  messages,
  busy,
  connected,
  onSend,
  onReset,
}: {
  messages: ChatMessage[];
  busy: boolean;
  connected: boolean;
  onSend: (text: string) => void;
  onReset: () => void;
}) {
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };

  const quickReplies = deriveTrainingActions(messages);
  const lastDone = messages.length > 0 && messages[messages.length - 1].done;
  const showQuickReplies = quickReplies.length > 0 && !busy && lastDone && connected;

  return (
    <div className="chat">
      <div className="chat-thread">
        {messages.length === 0 ? (
          <div className="welcome">
            <h1>Talk to your brain</h1>
            <p>
              A live EEG copilot. Ask about your signal, frequency bands and
              channel quality — or teach it a new pattern by example.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion" onClick={() => onSend(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <Bubble m={m} key={m.id} />)
        )}
        <div ref={endRef} />
      </div>

      {showQuickReplies && (
        <div className="quick-replies">
          {quickReplies.map((q) => (
            <button
              key={q.label}
              className={"quick-reply" + (q.tone ? " " + q.tone : "")}
              onClick={() => onSend(q.message)}
            >
              {q.label}
            </button>
          ))}
        </div>
      )}

      <div className="composer">
        <textarea
          value={text}
          placeholder="Ask about your brain — focus, signal quality, train a pattern…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
        />
        <button className="send" disabled={busy || !text.trim()} onClick={submit}>
          {busy ? "…" : "Send"}
        </button>
      </div>

      <div className="chat-bar">
        <button className="link" onClick={() => { onReset(); toast.info("Chat reset"); }}>
          Reset conversation
        </button>
        <span className="conn">
          <span className={"mini-dot " + (connected ? "on" : "off")} />
          {connected ? "connected" : "reconnecting…"}
        </span>
      </div>
    </div>
  );
}
