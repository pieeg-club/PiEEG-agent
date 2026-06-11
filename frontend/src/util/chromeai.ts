/**
 * Chrome Built-in AI (Gemini Nano) adapter.
 * 
 * Fallback provider when backend LLMs are rate-limited. Runs entirely client-side
 * in Chrome 127+ with prompt API enabled via chrome://flags/#prompt-api-for-gemini-nano
 * 
 * **Detection**: Check `ai?.languageModel` availability
 * **Limits**: ~1024 tokens/response, context window varies by device
 * **Privacy**: All inference happens locally, no data leaves the browser
 * 
 * @see https://developer.chrome.com/docs/ai/built-in-apis
 */

// Chrome AI types (not in @types/chrome yet)
// Uses global 'ai' object (not window.ai - that's obsolete)
declare global {
  const ai: {
    languageModel?: {
      capabilities(): Promise<{
        available: "readily" | "after-download" | "no";
        defaultTemperature?: number;
        defaultTopK?: number;
        maxTopK?: number;
      }>;
      create(options?: {
        temperature?: number;
        topK?: number;
        systemPrompt?: string;
      }): Promise<ChromeAISession>;
    };
  } | undefined;
}

interface ChromeAISession {
  prompt(text: string): Promise<string>;
  promptStreaming(text: string): ReadableStream<string>;
  destroy(): void;
  clone(): Promise<ChromeAISession>;
  tokensLeft: number;
  tokensSoFar: number;
  maxTokens: number;
}

/** System prompt optimized for EEG agent context */
const SYSTEM_PROMPT = `You are an EEG analysis assistant embedded in PiEEG Agent.
The user is monitoring their brain activity in real-time. They may ask about:
- Current signal quality, artifacts, or connectivity patterns
- What mental states their brain signals indicate
- How to improve signal quality or troubleshoot electrodes
- Guidance for training neural pattern detectors

Keep answers concise (1-3 sentences) since you're a fallback with limited context.
Be honest about limitations — you don't have live EEG data access, only what the user tells you.`;

export class ChromeAI {
  private session: ChromeAISession | null = null;
  private initializing = false;

  /** Check if Chrome built-in AI is available */
  static async isAvailable(): Promise<boolean> {
    if (typeof ai === "undefined" || !ai?.languageModel) return false;
    try {
      const caps = await ai.languageModel.capabilities();
      return caps.available === "readily" || caps.available === "after-download";
    } catch {
      return false;
    }
  }

  /** Get availability status with details */
  static async getStatus(): Promise<{
    available: boolean;
    status: "ready" | "download-needed" | "unavailable" | "unknown";
    message: string;
  }> {
    if (typeof ai === "undefined" || !ai?.languageModel) {
      return {
        available: false,
        status: "unavailable",
        message: "Chrome built-in AI not detected. Requires Chrome 127+ with chrome://flags/#prompt-api-for-gemini-nano enabled.",
      };
    }

    try {
      const caps = await ai.languageModel.capabilities();
      if (caps.available === "readily") {
        return {
          available: true,
          status: "ready",
          message: "Chrome AI ready (Gemini Nano running locally)",
        };
      }
      if (caps.available === "after-download") {
        return {
          available: true,
          status: "download-needed",
          message: "Chrome AI available but model needs download (will happen automatically on first use)",
        };
      }
      return {
        available: false,
        status: "unavailable",
        message: "Chrome AI not available on this device",
      };
    } catch (err) {
      return {
        available: false,
        status: "unknown",
        message: `Chrome AI status check failed: ${err}`,
      };
    }
  }

  /** Initialize a session (lazy, happens on first prompt) */
  private async ensureSession(): Promise<ChromeAISession> {
    if (this.session) return this.session;
    if (this.initializing) {
      // Wait for existing init to complete
      while (this.initializing) {
        await new Promise((r) => setTimeout(r, 100));
      }
      if (this.session) return this.session;
    }

    this.initializing = true;
    try {
      if (typeof ai === "undefined" || !ai?.languageModel) {
        throw new Error("Chrome AI not available");
      }
      this.session = await ai.languageModel.create({
        systemPrompt: SYSTEM_PROMPT,
        temperature: 0.7,
        topK: 3,
      });
      return this.session;
    } finally {
      this.initializing = false;
    }
  }

  /** Send a prompt and get the full response */
  async prompt(text: string): Promise<string> {
    const session = await this.ensureSession();
    return await session.prompt(text);
  }

  /** Send a prompt and stream the response token-by-token */
  async *promptStream(text: string): AsyncGenerator<string, void, unknown> {
    const session = await this.ensureSession();
    const stream = session.promptStreaming(text);
    const reader = stream.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        yield value;
      }
    } finally {
      reader.releaseLock();
    }
  }

  /** Get token usage stats */
  getUsage(): { used: number; remaining: number; max: number } | null {
    if (!this.session) return null;
    return {
      used: this.session.tokensSoFar,
      remaining: this.session.tokensLeft,
      max: this.session.maxTokens,
    };
  }

  /** Clean up session */
  destroy() {
    this.session?.destroy();
    this.session = null;
  }
}

/** Singleton instance for app-wide use */
export const chromeAI = new ChromeAI();
