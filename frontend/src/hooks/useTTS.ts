import { useEffect, useRef, useState } from "react";

const LS_KEY = "pieeg-tts-enabled";

export type TTSHandle = {
  enabled: boolean;
  setEnabled: (v: boolean) => void;
  isSupported: boolean;
  speaking: boolean;
  speak: (text: string, opts?: { lang?: string; rate?: number; pitch?: number; voiceName?: string }) => void;
  stop: () => void;
};

export function useTTS(): TTSHandle {
  const isSupported = typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined";
  const [enabled, setEnabled] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem(LS_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [speaking, setSpeaking] = useState(false);
  const utterRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(LS_KEY, enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }, [enabled]);

  const stop = () => {
    if (!isSupported) return;
    window.speechSynthesis.cancel();
    utterRef.current = null;
    setSpeaking(false);
  };

  const speak = (text: string, opts?: { lang?: string; rate?: number; pitch?: number; voiceName?: string }) => {
    // Allow callers to speak ad-hoc even if the global `enabled` flag is off.
    if (!isSupported) return;
    if (!text || !text.trim()) return;
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      if (opts?.lang) u.lang = opts.lang;
      if (opts?.rate) u.rate = opts.rate;
      if (opts?.pitch) u.pitch = opts.pitch;
      if (opts?.voiceName) {
        const voices = window.speechSynthesis.getVoices() || [];
        const match = voices.find((v) => v.name === opts.voiceName);
        if (match) u.voice = match;
      }
      u.onstart = () => setSpeaking(true);
      u.onend = () => {
        setSpeaking(false);
        utterRef.current = null;
      };
      u.onerror = () => {
        setSpeaking(false);
        utterRef.current = null;
      };
      utterRef.current = u;
      window.speechSynthesis.speak(u);
    } catch (err) {
      // ignore errors
      setSpeaking(false);
      utterRef.current = null;
    }
  };

  return { enabled, setEnabled, isSupported, speaking, speak, stop };
}

export default useTTS;
