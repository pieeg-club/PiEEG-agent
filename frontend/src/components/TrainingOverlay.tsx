import { useEffect, useRef, useState } from "react";
import type { TrainResult } from "../types";
import { pct, scoreTone } from "../util";
import { toast } from "./Toast";

const PREP_S = 2; // "get ready" beat before each recorded segment

type Stage = "prep" | "record";
interface Phase {
  stage: Stage;
  label: "rest" | "active";
  rep: number;
  remaining: number;
  total: number;
}

interface Trainer {
  begin: (name: string) => Promise<TrainResult>;
  record: (label: string, seconds: number) => Promise<TrainResult>;
  finish: (threshold: number) => Promise<TrainResult>;
  cancel: () => Promise<TrainResult>;
  connected: boolean;
}

const RING_R = 54;
const RING_C = 2 * Math.PI * RING_R;

function Ring({ phase }: { phase: Phase }) {
  const frac = Math.max(0, Math.min(1, phase.remaining / phase.total));
  const tone = phase.label === "rest" ? "ring-rest" : "ring-active";
  return (
    <div className={"ring " + tone}>
      <svg viewBox="0 0 120 120">
        <circle className="ring-bg" cx="60" cy="60" r={RING_R} />
        <circle
          className="ring-fg"
          cx="60"
          cy="60"
          r={RING_R}
          strokeDasharray={RING_C}
          strokeDashoffset={RING_C * (1 - frac)}
        />
      </svg>
      <div className="ring-num">{Math.ceil(phase.remaining)}</div>
    </div>
  );
}

// Guided pattern training over /ws/train. Walks the user through rest/active
// segments for a few reps (each a server-blocking record while live frames
// stream in), then fits + cross-validates and shows the honest result.
export function TrainingOverlay({
  train,
  onClose,
}: {
  train: Trainer;
  onClose: () => void;
}) {
  const [step, setStep] = useState<"setup" | "guide" | "finishing" | "result">("setup");
  const [name, setName] = useState("");
  const [reps, setReps] = useState(4);
  const [seconds, setSeconds] = useState(4);
  const [error, setError] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase | null>(null);
  const [totals, setTotals] = useState<TrainResult["totals"] | null>(null);
  const [result, setResult] = useState<TrainResult | null>(null);
  const cancelRef = useRef(false);

  const countdown = (stage: Stage, label: "rest" | "active", rep: number, total: number) =>
    new Promise<void>((resolve) => {
      const t0 = Date.now();
      setPhase({ stage, label, rep, remaining: total, total });
      const iv = setInterval(() => {
        const left = total - (Date.now() - t0) / 1000;
        if (left <= 0 || cancelRef.current) {
          clearInterval(iv);
          setPhase((p) => (p ? { ...p, remaining: 0 } : p));
          resolve();
        } else {
          setPhase((p) => (p ? { ...p, remaining: left } : p));
        }
      }, 100);
    });

  const recordPhase = async (label: "rest" | "active", rep: number) => {
    await countdown("prep", label, rep, PREP_S);
    if (cancelRef.current) return;
    // Animate the record window locally while the server blocks for `seconds`.
    const t0 = Date.now();
    setPhase({ stage: "record", label, rep, remaining: seconds, total: seconds });
    const iv = setInterval(() => {
      const left = Math.max(0, seconds - (Date.now() - t0) / 1000);
      setPhase((p) => (p && p.stage === "record" ? { ...p, remaining: left } : p));
    }, 100);
    const res = await train.record(label, seconds);
    clearInterval(iv);
    if (res.totals) setTotals(res.totals);
  };

  const runGuide = async () => {
    for (let r = 0; r < reps; r++) {
      if (cancelRef.current) return;
      await recordPhase("rest", r);
      if (cancelRef.current) return;
      await recordPhase("active", r);
    }
    if (cancelRef.current) return;
    setStep("finishing");
    const res = await train.finish(0.6);
    setResult(res);
    setStep("result");
    if (res.balanced_accuracy != null) {
      const acc = res.balanced_accuracy;
      if (acc >= 0.7) {
        toast.success(`Pattern trained! ${pct(acc)} accuracy`);
      } else {
        toast.info(`Pattern trained with ${pct(acc)} accuracy — add more reps to improve`);
      }
    }
  };

  const begin = async () => {
    setError(null);
    cancelRef.current = false;
    const res = await train.begin(name.trim());
    if (res.error) return setError(res.error);
    if (res.status === "no_signal") return setError(res.detail || "No signal yet — start the stream first.");
    setStep("guide");
    runGuide();
  };

  const abort = async () => {
    cancelRef.current = true;
    await train.cancel();
    onClose();
  };

  const phaseTitle = (p: Phase) => {
    if (p.stage === "prep") return p.label === "rest" ? "Get ready: REST" : "Get ready: ACTIVE";
    return p.label === "rest"
      ? "REST — relax and hold still"
      : "ACTIVE — do the pattern now";
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && (step === "setup" || step === "result")) {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [step, onClose]);

  return (
    <div className="overlay">
      <div className="modal train" onClick={(e) => e.stopPropagation()}>
        {step === "setup" && (
          <>
            <div className="modal-head">
              <h2>Train a pattern</h2>
              <button className="x" onClick={onClose}>
                ×
              </button>
            </div>
            <div className="modal-body">
              <p className="muted">
                Teach the agent a brain or muscle pattern by example. You'll
                alternate <b>rest</b> and <b>active</b> for a few reps, then it
                fits a detector and cross-validates it honestly.
              </p>
              <label className="field">
                <span>Pattern name</span>
                <input
                  autoFocus
                  value={name}
                  placeholder="e.g. eyes-closed, jaw-clench, imagine-left"
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && name.trim() && begin()}
                />
              </label>
              <div className="field-row">
                <label className="field">
                  <span>Reps: {reps}</span>
                  <input
                    type="range"
                    min={2}
                    max={8}
                    value={reps}
                    onChange={(e) => setReps(+e.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Seconds / segment: {seconds}</span>
                  <input
                    type="range"
                    min={2}
                    max={8}
                    value={seconds}
                    onChange={(e) => setSeconds(+e.target.value)}
                  />
                </label>
              </div>
              {error && <div className="warn-line">{error}</div>}
              {!train.connected && <div className="warn-line">connecting to agent…</div>}
            </div>
            <div className="modal-foot">
              <button className="btn ghost" onClick={onClose}>
                Cancel
              </button>
              <button className="btn primary" disabled={!name.trim() || !train.connected} onClick={begin}>
                Begin
              </button>
            </div>
          </>
        )}

        {step === "guide" && phase && (
          <div className="guide">
            <div className="guide-rep">
              Rep {phase.rep + 1} / {reps}
            </div>
            <Ring phase={phase} />
            <div className={"guide-label " + phase.label}>{phaseTitle(phase)}</div>
            {totals && (
              <div className="guide-totals">
                rest {totals.rest} · active {totals.active} · {totals.reps} reps
              </div>
            )}
            <button className="btn ghost" onClick={abort}>
              Cancel
            </button>
          </div>
        )}

        {step === "finishing" && (
          <div className="guide">
            <div className="spinner" />
            <div className="guide-label">Training & cross-validating…</div>
          </div>
        )}

        {step === "result" && result && (
          <>
            <div className="modal-head">
              <h2>{result.error ? "Couldn't train" : `Trained: ${result.name}`}</h2>
              <button className="x" onClick={result.error ? abort : onClose}>
                ×
              </button>
            </div>
            <div className="modal-body">
              {result.error ? (
                <>
                  <div className="warn-line">{result.error}</div>
                  {result.totals && (
                    <div className="muted">
                      captured: rest {result.totals.rest}, active {result.totals.active},{" "}
                      {result.totals.reps} reps
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="explain-top">
                    <div className={"score-badge " + scoreTone(result.balanced_accuracy)}>
                      <span className="score-num">{pct(result.balanced_accuracy)}</span>
                      <span className="score-label">balanced acc.</span>
                    </div>
                    <div className="explain-facts">
                      {result.n_folds != null && <div>{result.n_folds}-fold leave-one-rep-out CV</div>}
                      {result.balanced_accuracy != null && result.balanced_accuracy < 0.7 && (
                        <div className="warn-line">weak — record more reps for a better detector</div>
                      )}
                      <div className="muted">saved · now decoding live</div>
                    </div>
                  </div>
                  {result.channel_importance && result.channel_importance.length > 0 && (
                    <div className="explain-section">
                      <div className="section-title">Relies most on</div>
                      <div className="chips">
                        {result.channel_importance.slice(0, 5).map((c) => (
                          <span className="chip" key={c.channel}>
                            {c.channel} {pct(c.importance)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {result.caveat && <div className="caveat">{result.caveat}</div>}
                </>
              )}
            </div>
            <div className="modal-foot">
              {result.error ? (
                <>
                  <button className="btn ghost" onClick={abort}>
                    Close
                  </button>
                  <button className="btn primary" onClick={() => setStep("setup")}>
                    Try again
                  </button>
                </>
              ) : (
                <button className="btn primary" onClick={onClose}>
                  Done
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
