import { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import type { ChatMessage, Part, ToolPart } from "../hooks/useChatSocket";
import { toast } from "./Toast";
import { NotebookViewer } from "./NotebookViewer";
import useTTS from "../hooks/useTTS";

const ALL_SUGGESTIONS = [
  // Signal Quality & Monitoring
  "What is my brain doing right now?",
  "How is my signal quality?",
  "Which channels look bad?",
  "Show me my current frequency bands",
  "Is there any artifact contamination?",
  "Which channel has the best signal?",
  
  // Training & Pattern Recognition
  "Let's train a new pattern together",
  "Show me all trained patterns",
  "Can I train a focus detection pattern?",
  "Help me train a relaxation state",
  "Train a pattern to detect eye blinks",
  
  // Brain States & Analysis
  "Am I in a focused state right now?",
  "What's my alpha/beta ratio?",
  "Analyze my mental state trends",
  "Is my theta activity elevated?",
  
  // Connectivity & Network Analysis
  "Show me brain connectivity patterns",
  "Which regions are most connected?",
  "What's the coherence between channels?",
  
  // Troubleshooting & Help
  "Why is my signal noisy?",
  "How can I improve signal quality?",
  "What do the different brain waves mean?",
  "Explain the cascade monitor",
  
  // Advanced Features
  "Export my current session data",
  "Compare current state to baseline",
];

// Fisher-Yates shuffle
function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

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

function NotebookResultChip({ part }: { part: ToolPart }) {
  const [showNotebook, setShowNotebook] = useState(false);
  const result = part.result as { path?: string; cells?: number; status?: string; error?: string } | undefined;
  
  if (!result || result.error) {
    return <ToolChip part={part} />;
  }

  const path = result.path;
  const cellCount = result.cells || 0;
  const notebookName = path ? path.split(/[/\\]/).pop() : "notebook.ipynb";

  return (
    <>
      <div className="notebook-result-chip">
        <div className="notebook-chip-icon">📓</div>
        <div className="notebook-chip-content">
          <div className="notebook-chip-title">
            {part.name === "create_notebook" ? "Created notebook" : "Executed notebook"}
          </div>
          <div className="notebook-chip-name">{notebookName}</div>
          <div className="notebook-chip-meta">{cellCount} cells</div>
        </div>
        <button
          className="notebook-chip-btn"
          onClick={() => setShowNotebook(true)}
        >
          View
        </button>
      </div>
      {showNotebook && path && (
        <NotebookViewer
          notebookPath={path}
          onClose={() => setShowNotebook(false)}
        />
      )}
    </>
  );
}

function renderPart(p: Part, i: number, msgId: string) {
  if (p.kind === "text") {
    const html = marked.parse(p.text, { async: false }) as string;
    return (
      <div className="text" key={`${msgId}-text-${i}`} dangerouslySetInnerHTML={{ __html: html }} />
    );
  }
  
  // Special rendering for notebook tools
  if (
    p.kind === "tool" &&
    p.status === "done" &&
    (p.name === "create_notebook" || p.name === "run_notebook")
  ) {
    return <NotebookResultChip part={p} key={`${msgId}-tool-${i}-${p.name}`} />;
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
  
  // Check if currently thinking (has tool calls running or waiting for next step)
  const hasRunningTools = m.parts.some(p => p.kind === "tool" && p.status === "running");
  const hasCompletedTools = m.parts.some(p => p.kind === "tool" && p.status === "done");
  const isThinking = !m.done && (hasRunningTools || (hasCompletedTools && !empty));
  
  const copyMessage = () => {
    const textParts = m.parts.filter((p) => p.kind === "text").map((p) => (p as { text: string }).text);
    const text = textParts.join("\n");
    navigator.clipboard.writeText(text).then(() => {
      toast.success("Message copied");
    });
  };

  // TTS controls: provided by global hook in Chat (but Bubble can also call it)
  const tts = useTTS();
  const hasTextParts = m.parts.filter((p) => p.kind === "text").length > 0;
  const messageText = m.parts
    .filter((p) => p.kind === "text")
    .map((p) => (p as { text: string }).text)
    .join("\n");

  const handleSpeak = () => {
    if (!tts.isSupported) {
      toast.info("Text-to-speech not supported in this browser");
      return;
    }
    if (tts.speaking) {
      tts.stop();
      return;
    }
    tts.speak(messageText);
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
        {m.role === "assistant" && isThinking && (
          <div className="thinking">
            <div className="thinking-orb">
              <span /><span /><span /><span />
            </div>
            <span className="thinking-text">Thinking...</span>
          </div>
        )}
        {m.role === "assistant" && !empty && !m.done && !isThinking && <span className="cursor" />}
        {m.usage && (
          <div className="usage">
            {m.usage.input_tokens}→{m.usage.output_tokens} tok
            {m.iterations ? ` · ${m.iterations} step${m.iterations > 1 ? "s" : ""}` : ""}
          </div>
        )}
        {!m.done && m.iterations && m.iterations > 0 && (
          <div className="progress-bar">
            <div className="progress-text">Step {m.iterations} / 10</div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${(m.iterations / 10) * 100}%` }} />
            </div>
          </div>
        )}
        {m.role === "assistant" && m.done && !empty && (
          <div className="msg-toolbar">
            <button className="msg-btn copy" onClick={copyMessage} title="Copy message">
              📋
            </button>
            {hasTextParts && (
              <button
                className={`msg-btn speak ${tts.speaking ? "speaking" : ""}`}
                onClick={handleSpeak}
                title={tts.speaking ? "Stop" : "Speak message"}
              >
                {tts.speaking ? "⏹" : "🔊"}
              </button>
            )}
          </div>
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
  const tts = useTTS();
  const spokenRef = useRef<Set<string>>(new Set());
  const [displayedSuggestions, setDisplayedSuggestions] = useState<string[]>(() => 
    shuffleArray(ALL_SUGGESTIONS).slice(0, 6)
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-speak new assistant messages when TTS is enabled
  useEffect(() => {
    if (!tts.enabled || !tts.isSupported) return;
    for (const m of messages) {
      if (m.role === "assistant" && m.done && m.parts.length > 0 && !spokenRef.current.has(m.id)) {
        const textParts = m.parts.filter((p) => p.kind === "text").map((p) => (p as { text: string }).text);
        const text = textParts.join("\n").trim();
        if (text) {
          tts.speak(text);
          spokenRef.current.add(m.id);
        }
      }
    }
  }, [messages, tts]);

  const submit = () => {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  };

  const quickReplies = deriveTrainingActions(messages);
  const lastDone = messages.length > 0 && messages[messages.length - 1].done;
  const showQuickReplies = quickReplies.length > 0 && !busy && lastDone && connected;

  const shuffleSuggestions = () => {
    setDisplayedSuggestions(shuffleArray(ALL_SUGGESTIONS).slice(0, 6));
  };

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
            <div className="suggestions-wrapper">
              <div className="suggestions">
                {displayedSuggestions.map((s, i) => (
                  <button key={`${s}-${i}`} className="suggestion" onClick={() => onSend(s)}>
                    {s}
                  </button>
                ))}
              </div>
              <button className="shuffle-btn" onClick={shuffleSuggestions} title="Show different suggestions">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M13 2L13 5L10 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M3 14L3 11L6 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12.5 5.5C11.5 4 9.5 3 7.5 3C4.5 3 2 5.5 2 8.5C2 9.5 2.5 10.5 3 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  <path d="M3.5 10.5C4.5 12 6.5 13 8.5 13C11.5 13 14 10.5 14 7.5C14 6.5 13.5 5.5 13 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                <span>More suggestions</span>
              </button>
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
