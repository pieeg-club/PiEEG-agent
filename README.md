# PiEEG-agent


[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/github/license/pieeg-club/PiEEG-agent)](LICENSE)
[![LSL](https://img.shields.io/badge/LSL-compatible-green?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEyIDJhMTAgMTAgMCAxIDAgMCwyMCAxMCAxMCAwIDAgMCAwLTIwem0wIDJhOCA4IDAgMSAxIDAgMTYgOCA4IDAgMCAxIDAtMTZ6bTAgM2ExIDEgMCAwIDAtMSAxdjVhMSAxIDAgMCAwIDIgMFY4YTEgMSAwIDAgMC0xLTF6bTAgOWExIDEgMCAxIDAgMCAyIDEgMSAwIDAgMCAwLTJ6IiBmaWxsPSIjZmZmIi8+PC9zdmc+)](https://labstreaminglayer.org/)
[![Platform](https://img.shields.io/badge/platform-PiEEG%20|%20Any%20LSL%20Device-c51a4a)](https://pieeg.com)
[![LLM](https://img.shields.io/badge/LLM-Anthropic%20|%20OpenAI%20|%20Groq%20|%20Ollama%20|%20More-8A2BE2)](https://github.com/pieeg-club/PiEEG-agent)
[![Discord](https://img.shields.io/discord/1059637443548987462?color=5865F2&logo=discord&logoColor=white&label=Discord)](https://discord.gg/neJ45FR6Sv)
[![Docs](https://img.shields.io/badge/docs-pieeg.com-blue?logo=vercel&logoColor=white)](https://docs.pieeg.com)

**Natural language EEG lab notebook.** Train pattern classifiers, analyze connectivity, compare sessions — all by talking to an AI copilot that reads your live brain signals.



<img width="1918" height="962" alt="image" src="https://github.com/user-attachments/assets/2b643271-4904-4d37-a149-7f8c91163528" />




---

## ⚡ Quick Start: Full System with Web Interface

### Complete Setup (PiEEG Hardware + Web UI + Actions)

```bash
# Terminal 1: Start PiEEG server with LSL streaming
pieeg-server --lsl

# Terminal 2: Install agent and launch web interface 
pip install -e ".[server]"  # includes control plane for device actions
pieeg-agent web --allow-actions --execute
```

Open **http://localhost:8000** → Full brain copilot with:
- 🧠 **Live EEG monitoring** — focus/relax/engagement, band powers, quality
- 💬 **Natural language chat** — "am I focused?", "train a relaxation pattern", "show connectivity"
- 🎛️ **Device control** — copilot can adjust filters, start recording, enable OSC output
- 📊 **Pattern training** — record mental states, train classifiers, compare sessions

**What `--allow-actions --execute` means:**
- `--allow-actions`: Copilot can **preview** device actions (filters, recording, OSC)
- `--execute`: Copilot can **actually execute** approved actions (with safety gates)
- Without flags: Read-only mode (state monitoring, pattern training, connectivity analysis)

---

### Development Setup (No Hardware)

```bash
# Terminal 1: Mock EEG stream for testing
pieeg-server --mock --lsl

# Terminal 2: Launch web UI (read-only, no actions)
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...  # or skip for interactive setup
pieeg-agent web
```

Open http://localhost:8000 → Chat with synthetic brain data.

**No hardware?** Mock server generates realistic 8-channel EEG with configurable patterns.

**No API key?** Interactive setup wizard guides you through provider selection:

```
🤖 LLM Provider Setup
==================================================
Choose a provider to power the brain copilot:

  1. Anthropic
  2. OpenAI
  3. Groq
  4. Together AI
  5. Ollama (local, no key needed)
  6. LM Studio (local, no key needed)
  7. Echo (debug, no key needed)

Select provider [1-7]: 1

✓ Selected: Anthropic

🔑 API Key Required
This provider needs an API key stored in $ANTHROPIC_API_KEY
Get yours from:
  → https://console.anthropic.com/settings/keys

Enter your Anthropic API key: ********
✓ API key received

💾 Save Configuration
Would you like to save these settings?
  Location: ~/.pieeg-agent/config.json
  ⚠️  API key will be stored in plaintext (file permissions: 0600)

Save configuration? [Y/n]: y
✓ Configuration saved
```

**Configuration persists**: Once saved, your provider and API key are automatically loaded on subsequent runs. You won't be prompted again unless:
- The saved config is invalid or missing
- You explicitly reset it with `pieeg-agent config reset`
- You override with environment variables or CLI flags

**Manage configuration:**
```bash
pieeg-agent config          # view current settings
pieeg-agent config reset    # delete saved config
```

---

## 🎛️ Device Actions & Safety

### How Actions Work

The copilot can **control PiEEG hardware** through a WebSocket connection to `pieeg-server`. All actions pass through a **safety gate**:

```
┌─────────────┐
│ User request│  "start recording and set filter to 1-40 Hz"
└──────┬──────┘
       ▼
┌─────────────────────────────────────────────────────┐
│ LLM Copilot                                         │
│  • Understands intent                               │
│  • Calls: start_recording(), set_filter(1, 40)     │
└──────┬──────────────────────────────────────────────┘
       ▼
┌─────────────────────────────────────────────────────┐
│ Safety Gate                                         │
│  ✓ Allowlist check (is this action permitted?)     │
│  ✓ Dry-run mode check (preview vs execute?)        │
│  ✓ Cooldown check (not too soon after last call?)  │
│  ✓ Audit log (record attempt + outcome)            │
└──────┬──────────────────────────────────────────────┘
       ▼
┌─────────────────────────────────────────────────────┐
│ PiEEG Server (if approved)                          │
│  • Filter updated: 1-40 Hz                          │
│  • Recording started: data.csv                      │
└─────────────────────────────────────────────────────┘
```

### Action Modes

| Mode | Command | Behavior |
|------|---------|----------|
| **Read-only** | `pieeg-agent web` | No device control, only monitoring/analysis |
| **Preview** | `pieeg-agent web --allow-actions` | Shows what it would do, doesn't execute |
| **Execute** | `pieeg-agent web --allow-actions --execute` | Actually controls device (gated) |

### Available Actions

| Action | What it does | Gated? |
|--------|--------------|--------|
| `server_status` | Read device state (sample rate, channels, filter) | ❌ Read-only |
| `set_filter` | Adjust band-pass filter range | ✅ Yes |
| `start_recording` / `stop_recording` | Server-side CSV recording | ✅ Yes |
| `start_osc` / `stop_osc` | OSC output for external apps (VRChat, Unity) | ✅ Yes |
| `apply_register_preset` | ADS1299 chip preset (normal/test/short) | ✅ Yes |

### Example: Web UI with Actions

```bash
# Full control mode
pieeg-agent web --allow-actions --execute
```

In chat:
```
you > what's my current filter setting?
  (calls: server_status)
copilot > Band-pass is 0.5–45 Hz, 8 channels active at 250 Hz.

you > narrow it to 1–30 Hz for cleaner alpha
  (calls: set_filter)
copilot > Done. Filter updated to 1–30 Hz.

you > start a 5-minute recording
  (calls: start_recording)
copilot > Recording started — saving to data_20260611_143052.csv on the server.
```

**Safety features:**
- **Cooldown**: Can't spam the same action (default: 2-3 seconds between calls)
- **Audit log**: Every attempt logged to `~/.pieeg-agent/audit.jsonl`
- **Dry-run default**: Must explicitly add `--execute` to actually control device

---

## 🧪 Lab Features — What You Can Do

**The challenge**: LLMs reason in natural language at ~1 Hz. EEG arrives at 250–500 Hz × multiple channels = thousands of samples/second. You can't dump raw voltages into a prompt.

**The solution**: A **perception cascade** that progressively trades temporal resolution for semantic density:

```
┌─────────────┬──────────┬────────────────────────┬──────────────┐
│ Tier        │ Rate     │ Representation         │ LLM sees it? │
├─────────────┼──────────┼────────────────────────┼──────────────┤
│ T0 Raw      │ 250 Hz   │ float32 µV samples     │ never        │
│ T1 Features │   8 Hz   │ band powers, quality   │ never        │
│ T2 State    │   1 Hz   │ focus/relax/engagement │ on demand    │
│ T3 Events   │ ~sparse  │ "focus_high" @10:04:32 │ YES – main   │
│ T4 Reason   │ on query │ NL questions + tools   │ —            │
└─────────────┴──────────┴────────────────────────┴──────────────┘
```

**Three architectural rules:**
1. **Ingestion never blocks on the LLM** — high-rate intake runs in a dedicated thread, slow downstream processing never creates backpressure
2. **LLM pulls, never pushed** — the model calls tools to request state; it's never spammed with raw data
3. **Everything the model sees is language-sized** — indices, events, quality verdicts, not voltage arrays

This architecture is proven in production BCI systems (BrainFlow + Lab Streaming Layer) and keeps token costs sane while maintaining scientific validity.

---

## 🧠 System Architecture

### Full Integration with PiEEG Hardware

```
┌────────────────────────────────────────────────────────────────────┐
│  PiEEG Hardware + Server (Raspberry Pi or PC)                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ ADS1299 → pieeg-server                                       │ │
│  │   • 8-ch EEG @ 250 Hz                                        │ │
│  │   • Hardware filtering                                       │ │
│  │   • CSV recording                                            │ │
│  │   • LSL broadcast (EEG data)                                 │ │
│  │   • WebSocket :1616 (control commands)                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────┬───────────┘
         │ LSL stream                                    │ WebSocket
         │ (EEG data)                                    │ (actions)
         ▼                                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  PiEEG-Agent (same machine or remote)                              │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ PERCEIVE: High-rate → semantic pyramid                       │ │
│  │   T0: Raw 250 Hz µV samples → ring buffer (60s)              │ │
│  │   T1: FFT → band powers (δθαβγ) @ 8 Hz                       │ │
│  │   T2: State indices (focus/relax/engagement) @ 1 Hz          │ │
│  │   T3: Events (focus_high, quality_drop) @ sparse             │ │
│  └──────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ DECODE: Lab notebook tools                                   │ │
│  │   • Patterns: train classifiers (L2+group-lasso, LORO-CV)    │ │
│  │   • Connectivity: cross-channel coupling analysis            │ │
│  │   • Sessions: capture + compare with Cohen's d               │ │
│  └──────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ REASON: LLM copilot                                          │ │
│  │   • Reads: neural state, quality, events, patterns           │ │
│  │   • Writes: device actions (if --allow-actions --execute)    │ │
│  │   • Providers: Anthropic, OpenAI, Groq, Ollama, etc.         │ │
│  └──────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ ACT: Safety gate (optional, with --allow-actions)            │ │
│  │   ✓ Allowlist → ✓ Cooldown → ✓ Dry-run check → ✓ Audit      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────────────┘
         │ http://localhost:8000
         ▼
┌────────────────────────────────────────────────────────────────────┐
│  Web Interface (browser)                                           │
│    • Chat with brain copilot                                       │
│    • Live EEG state cards                                          │
│    • Pattern training UI                                           │
│    • System control panel                                          │
└────────────────────────────────────────────────────────────────────┘
```

### Standalone LSL Mode (No PiEEG Server)

```
┌──────────────────────────────────────────┐
│  Any LSL Source                          │
│  (OpenBCI, Muse, EMOTIV, Mock, etc.)     │
└────────┬─────────────────────────────────┘
         │ LSL stream only
         ▼
┌──────────────────────────────────────────┐
│  PiEEG-Agent (read-only mode)            │
│    ✓ PERCEIVE                            │
│    ✓ DECODE                              │
│    ✓ REASON                              │
│    ✗ ACT (no device to control)          │
└──────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation | Benefit |
|-----------|----------------|---------|
| **Ingestion never blocks** | Dedicated LSL thread, ring buffer | No sample loss even if LLM is slow |
| **LLM pulls, never pushed** | Tools request state on demand | Token costs stay sane |
| **Language-sized representations** | Events, not voltage arrays | Models reason about "focus_high", not floats |
| **Scientifically honest metrics** | Within-session normalization, warm-up flags | Users know what numbers mean |
| **Decoupled layers** | Swap providers, run without LLM, test with mock | Debuggable, testable, maintainable |
| **Safe by default** | Dry-run, cooldown, audit log | AI can't spam device commands |

---

## 📖 Usage Examples

### Integration with PiEEG Server

**PiEEG-server** is the hardware interface (reads from ADS1299 chip, applies filters, streams via LSL and WebSocket). **PiEEG-agent** is the AI copilot (reads LSL streams, analyzes brain state, optionally controls the server).

#### Full Integration Example

```bash
# On Raspberry Pi (or same machine):
pieeg-server --lsl --websocket --port 1616

# Agent connects to:
# - LSL stream (for EEG data)
# - WebSocket :1616 (for device control)
pieeg-agent web --allow-actions --execute
```

**What you can do:**
- Chat: "Am I focused?" → reads live EEG state
- Command: "Set filter to 8-30 Hz" → adjusts server filter
- Train: "Record 'meditation' for 60 seconds" → captures labeled session
- Export: "Start OSC to localhost:9000" → streams to Unity/VRChat

**Environment variables** (if server isn't on localhost:1616):
```bash
export PIEEG_WS_URL=ws://192.168.1.100:1616
export PIEEG_WS_TOKEN=your-auth-token  # if server requires auth
```

#### LSL-Only (No Device Control)

Use **any** LSL source — OpenBCI, Muse, EMOTIV, or PiEEG:

```bash
pieeg-agent web  # read-only, no --allow-actions
```

Copilot can:
- Monitor state (focus/relax/engagement)
- Train patterns ("eyes-open" vs "eyes-closed")
- Analyze connectivity (alpha coupling between channels)
- Compare sessions (meditation vs baseline)

Cannot:
- Adjust hardware filters (no device control)
- Start/stop recording on server (use server CLI directly)

---

### 1. Discover & Validate LSL Streams

Works with any LSL source (PiEEG, OpenBCI, Muse, EMOTIV, etc.):

```bash
pieeg-agent streams              # discover all LSL outlets on network
pieeg-agent ingest --seconds 10  # drain stream, print throughput stats
```

Exits with code 0 only if no samples lost and ring never overflows — validates your LSL source.

### 2. Monitor Live State

```bash
pieeg-agent monitor
```

Prints every second:
```
10:04:36 | focus 0.62 relax 0.41 engage 0.55 | d0.03 t0.01 a0.94 b0.01 g0.01 | dom Alpha | Q 1.00 clean
   >> 10:04:48  relax_high  -  relax rose to 0.74
   !! 10:05:02  quality_drop  -  signal quality degraded to 0.42 (Ch2)
```

- **Indices** — `focus`, `relax`, `engage` are 0–1 **relative to this session**, not absolute/clinical
- **Band powers** — `d`=delta, `t`=theta, `a`=alpha, `b`=beta, `g`=gamma (normalised)
- **Events** — `>>` marks state transitions, `!!` marks quality issues
- **`~` prefix** — warm-up period (indices need signal variance to normalise)

### 3. One-Shot Questions

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pieeg-agent ask "am I focused?"
pieeg-agent ask --provider ollama --model llama3.2 "is my signal clean?"
pieeg-agent ask --provider echo "test without an API key"  # debug mode, no LLM
```

Supported providers: `anthropic` (default), `openai`, `groq`, `together`, `ollama`, `lmstudio`, `echo` (debug).

**Debug mode**: Use `--provider echo` to test without API keys or network calls. The echo provider:
- **Simulates LLM responses** based on keyword matching (not real language understanding)
- **Executes real tool calls** — you see actual EEG data from the perception cascade
- **Shows system status** — tool count, perception connection, detected keywords
- Useful for frontend development, integration testing, and validating the perception pipeline

The model calls tools:
- `get_neural_state()` → focus/relax/engagement + dominant band
- `get_channel_quality()` → per-channel verdicts (good/flat/rail/noisy/line)
- `get_recent_events()` → timestamped event log
- `get_band_powers(per_channel=True)` → spatial detail

### 4. Interactive Chat

```bash
pieeg-agent chat
```

Keeps conversation context:
```
you > how's my signal?
  (calls: get_channel_quality)
copilot > All channels read "good", quality score 0.98 — trustworthy signal.

you > am I focused or relaxed?
  (calls: get_neural_state)
copilot > Relaxed. Alpha is dominant (0.87) and focus is low (0.31) — typical 
          eyes-closed resting state.

you > what changed in the last minute?
  (calls: get_recent_events)
copilot > Relaxation spiked to 0.81 about 35 seconds ago, no quality issues since.
```

Press Ctrl+D to exit.

### 5. Web UI — Chat + Live Brain View

```bash
# Read-only mode (monitoring + analysis)
pieeg-agent web

# With device control (preview mode)
pieeg-agent web --allow-actions

# With device control (execution mode)
pieeg-agent web --allow-actions --execute
```

Open **http://localhost:8000** for a browser-based interface with:

#### Live Brain State Cards
- **Focus/Relax/Engagement** — 0-1 indices updated every second
- **Band Powers** — delta, theta, alpha, beta, gamma distribution
- **Signal Quality** — per-channel health (good/flat/rail/noisy/line)
- **Connectivity** — cross-channel coupling in selected band
- **Artifacts** — blink/jaw/movement detection feed

#### Chat Interface
- **Multi-turn conversation** — maintains context across questions
- **Streaming responses** — see the copilot think in real-time
- **Tool call visibility** — shows which neural/device tools are invoked
- **LLM settings** — switch providers/models on the fly

#### Pattern Training UI
- **Record segments** — "Record 'meditation' for 60 seconds"
- **Train classifiers** — L2+group-lasso with LORO-CV
- **Channel importance** — see which electrodes matter
- **Live inference** — test trained patterns in real-time

#### System Control Panel (with `--allow-actions`)
- **Filter adjustment** — change band-pass range
- **Recording control** — start/stop CSV capture
- **OSC streaming** — enable output to external apps
- **Device status** — sample rate, channels, register preset

**Architecture:**
- Backend: FastAPI + WebSocket (2 Hz snapshots, chat events)
- Frontend: Vite + React/TypeScript, dark theme
- Same copilot/tools as CLI — web is just another interface

### 6. Lab Notebook — Pattern Recognition & Connectivity

**Pattern Training** (record and compare mental states):
```bash
you > record a pattern called "eyes-closed-rest" for 20 seconds
  (calls: record_segment)
copilot > Got it. Hold still... [20s window captured]
          Saved with 0.94 signal quality, alpha dominant.

you > record "eyes-open-focus" for 20 seconds
  [repeat]

you > train a classifier from those two patterns
  (calls: train_pattern)
copilot > Trained "focus-vs-rest" with balanced accuracy 0.89 (LORO-CV).
          Top cue: beta increase in channels C3/C4.

you > explain the focus-vs-rest pattern
  (calls: explain_pattern)
copilot > [Shows channel importance, feature weights, CV score]
```

Pattern classifiers use:
- **L2 + group-lasso regularization** for spatial sparsity
- **Leave-One-Rep-Out CV** (no temporal leakage)
- **Balanced accuracy** (mean of sensitivity/specificity)
- **Channel importance** shows which electrodes matter

**Connectivity Analysis** (cross-channel coupling):
```bash
you > show me connectivity in the alpha band
  (calls: connectivity)
copilot > Mean alpha coupling: 0.34 across all channels.
          Strongest pair: C3–C4 (r=0.72)
          Most connected: C3, least: Fp1

you > record a session called "meditation" for 60 seconds
  (calls: record_session)
copilot > Captured 60s window: alpha dominant (0.81), high relaxation (0.76),
          strong C3–C4 coupling (0.68). Saved to session store.

you > compare my "meditation" session to "baseline-rest"
  (calls: compare_sessions)
copilot > [Contrasts with Cohen's d per feature]
          Biggest change: alpha power +0.92 SD
          Relaxation: +0.71 SD
          C3–C4 coupling: +0.45 SD
```

Sessions capture:
- Band power means/spreads per channel
- Focus/relax/engagement indices with variance
- Artifact counts + quality scores
- Connectivity (if n_frames ≥ 8)

Comparison uses **within-session Cohen's d** (descriptive effect size, NOT a clinical/generalisation claim).

### 7. Direct Device Control (CLI)

**PiEEG-server control plane** — direct commands without LLM:

```bash
# Check device status
pieeg-agent control status
# Output:
#   Sample rate: 250 Hz
#   Channels: 8 active
#   Filter: 0.5–45 Hz (enabled)
#   Recording: inactive
#   LSL: streaming as "PiEEG"

# Adjust filter
pieeg-agent control set-filter --lowcut 1 --highcut 40

# Recording control
pieeg-agent control record start
pieeg-agent control record stop

# OSC streaming (for Unity, VRChat, Max/MSP, etc.)
pieeg-agent control osc start --host 127.0.0.1 --port 9000
pieeg-agent control osc stop

# Apply ADS1299 register preset
pieeg-agent control register-preset normal

# View action audit log
pieeg-agent control audit
pieeg-agent control audit --limit 10
```

**Audit log** tracks all attempts:
```
Audit log: ~/.pieeg-agent/audit.jsonl  (15 entries, showing 5)

  2026-06-11 14:32:18  [     OK] set_filter     - 1-40 Hz applied
  2026-06-11 14:32:45  [     OK] start_recording - data_20260611_143245.csv
  2026-06-11 14:33:02  [COOLDOWN] start_recording - denied: 2s remaining
  2026-06-11 14:33:15  [DRY-RUN] start_osc      - preview mode, not executed
  2026-06-11 14:34:01  [     OK] stop_recording - 16s captured
```

**Use cases:**
- **Scripting**: Automate filter changes, recording sessions
- **Integration**: Call from Python scripts, shell pipelines
- **Debugging**: Direct control when copilot is overkill

---

## 🔬 Scientific Approach: Honest Metrics

This agent is designed for **neurofeedback research and UX prototyping**, not clinical use. Metrics are presented with scientific honesty:

### Within-Session Relative Indices

`focus`, `relax`, `engagement` are **normalised to the session's own range** (rolling 10-minute window). They say "high *for you, right now*" — not absolute or clinical values.

- **Warm-up required**: Indices read 0.50 until the signal shows variance. A `~` prefix marks this period, and the copilot says "still warming up" instead of inventing numbers.
- **Quality gates**: Poor electrode contact or artifact is flagged *before* reporting state. Bad signal → meaningless indices.

### Event Detection with Hysteresis

Transitions require:
- **Threshold crossing** (default: 0.70 for high, 0.30 for low)
- **Minimum dwell time** (default: 2 s — no flicker on brief excursions)
- **Quality floor** (default: 0.50 — don't emit events during artifact)

This prevents spurious events from noise or motion.

### Why This Matters for Developers

- **Reproducible**: Seed the mock server, get deterministic signal
- **Testable**: Event logs have timestamps, quality verdicts are per-channel
- **Debuggable**: Every LLM claim is traceable to a tool call with exact state
- **Honest**: Users learn when the system *doesn't know* (warm-up, poor signal) instead of hallucinating confidence

---

## ⚙️ Configuration

### Interactive Setup Wizard

First-time setup is guided by an interactive wizard:

```bash
pieeg-agent web  # or chat, ask
```

If no LLM provider is configured, you'll see:

1. **Provider selection** — 7 options (Anthropic, OpenAI, Groq, Together, Ollama, LM Studio, Echo)
2. **API key input** — Secure prompt (hidden) for cloud providers; local providers skip this
3. **Save option** — Persist to `~/.pieeg-agent/config.json` (chmod 600)

**Configuration persistence:**
- Saved config auto-loads on next run
- Override with environment variables (`PIEEG_LLM_PROVIDER`, `ANTHROPIC_API_KEY`)
- Reset with `pieeg-agent config reset`

**Non-interactive mode** (Docker, CI, systemd):
```bash
export PIEEG_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
pieeg-agent web  # skips wizard, uses env vars
```

### Environment Variables

All tunables have env-var overrides and CLI flags:

| Setting | Env Var | Default | Notes |
|---------|---------|---------|-------|
| LLM provider | `PIEEG_LLM_PROVIDER` | `anthropic` | `openai`, `groq`, `ollama`, etc. |
| Model name | `PIEEG_LLM_MODEL` | provider default | e.g. `claude-3-5-sonnet-20241022` |
| API key | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Not needed for local (Ollama) |
| LSL stream name | `PIEEG_LSL_NAME` | `PiEEG` | Auto-discover if not set |
| Ring buffer (s) | `PIEEG_RING_SECONDS` | `60.0` | How much history to keep |
| Feature rate (Hz) | — | `8.0` | FFT update frequency |
| State rate (Hz) | — | `1.0` | Index smoothing output |
| Mains freq (Hz) | — | `50.0` | Line-noise check (use `60` for US) |

Example:
```bash
export PIEEG_LLM_PROVIDER=ollama
export PIEEG_LLM_MODEL=llama3.2
pieeg-agent chat                # uses local Llama, no key needed
```

### Supported Providers

| Provider | Type | API Key Needed | Get Key From |
|----------|------|----------------|--------------|
| Anthropic | Cloud | ✅ `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| OpenAI | Cloud | ✅ `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Groq | Cloud | ✅ `GROQ_API_KEY` | https://console.groq.com/keys |
| Together AI | Cloud | ✅ `TOGETHER_API_KEY` | https://api.together.xyz/settings/api-keys |
| Ollama | Local | ❌ None | Install from https://ollama.ai |
| LM Studio | Local | ❌ None | Install from https://lmstudio.ai |
| Echo | Debug | ❌ None | Built-in keyword matcher |

---

## 🧪 Development & Testing

### Installation Modes

```bash
# Minimal (monitoring + analysis only)
pip install -e .

# With device control
pip install -e ".[server]"

# Full dev environment
pip install -e ".[dev,server]"
```

### Testing Without Hardware

```bash
# Mock EEG stream (deterministic signal)
pieeg-server --mock --lsl --seed 42

# Validate stream throughput
pieeg-agent ingest --seconds 10
# Exit code 0 = no samples lost, ring never overflowed

# Monitor live (no LLM needed)
pieeg-agent monitor
```

### Testing Without API Keys

```bash
# Echo provider (keyword-based simulator)
pieeg-agent web --provider echo                    # web UI, real tools
pieeg-agent chat --provider echo                   # CLI chat
pieeg-agent ask --provider echo "am I focused?"   # one-shot query

# Echo provider executes REAL tool calls (get_neural_state, etc.)
# Only the LLM reasoning is simulated via keyword matching
```

### Running Tests

```bash
pytest tests/                          # all tests
pytest tests/test_ring.py             # specific module
pytest -v --tb=short                   # verbose, short tracebacks
```

**Test coverage:**
- Ring buffer overwrite logic
- FFT feature extraction accuracy
- Event detector hysteresis
- Provider wire format (tool calling)
- Gate enforcement (allowlist, cooldown, dry-run)
- Perception cascade integration

### Module Overview

| Module | Purpose | Dependencies |
|--------|---------|-------------|
| `ingest/` | LSL intake, ring buffer | `pylsl` only |
| `perceive/` | Feature extraction, state estimation, events | NumPy, SciPy (pure DSP) |
| `decode/` | Pattern training, connectivity, sessions | scikit-learn |
| `llm/` | Provider abstraction (HTTP-only, no SDKs) | stdlib `urllib` |
| `agent/` | Copilot, tool dispatch, conversation loop | depends on above |
| `server/` | WebSocket client, action gate, audit log | `websockets` (optional) |
| `web/` | FastAPI backend + React/TS frontend | FastAPI, Vite |

**Decoupling principles:**
- Perception runs without LLM (works in `monitor` mode)
- LSL intake never blocks on downstream processing
- LLM provider is swappable (Anthropic → Ollama → LM Studio)
- Web UI is optional (CLI works standalone)

---

## 🎯 Roadmap

- [x] **Phase 0**: High-rate LSL intake (chunked pulls, ring buffer)
- [x] **Phase 1**: Perception cascade (features → state → events)
- [x] **Phase 2**: LLM copilot (read-only tools, provider-agnostic)
- [x] **Phase 3**: Gated device actions (allowlist, dry-run, audit)
- [x] **Phase A**: Artifact detection (blinks, jaw, movement) + quality tracking
- [x] **Phase B**: Pattern training (L2+group-lasso, LORO-CV, balanced accuracy)
- [x] **Phase C**: Web UI (FastAPI + React/TS, chat + live brain cards)
- [x] **Phase D**: Connectivity analysis + session comparison (lab notebook tools)
- [ ] **Phase E**: P300 event-related potential decoder (single-trial BCI control)
- [ ] **Phase F**: Multi-modal fusion (EEG + EOG + EMG streams, cross-modal tools)

---

## 📚 Learn More

- **[PiEEG Hardware](https://pieeg.com)** — Open-source EEG shield for Raspberry Pi
- **[Lab Streaming Layer](https://labstreaminglayer.org)** — Time-sync for multi-modal bio signals
- **[Anthropic Tools](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)** — How the copilot calls functions
- **[Honest Neurofeedback Metrics](https://www.frontiersin.org/articles/10.3389/fnhum.2016.00301/full)** — Why we surface warm-up and quality

---

## 📄 License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International) — see [LICENSE](LICENSE) for details.

You are free to share and adapt this work for non-commercial purposes with attribution.

---

## 🙏 Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation, here's how to help:

### Reporting Issues

Include the following for faster resolution:

**For bugs:**
```bash
pieeg-agent config          # current configuration
pieeg-agent streams         # LSL stream discovery
# Paste error traceback
```

**For feature requests:**
- **Use case**: What problem are you solving?
- **Proposed API**: CLI commands or copilot interactions
- **Why existing tools don't work**: What's missing?

**For metric/algorithm changes:**
- **Citations**: Papers or validated implementations
- **Test data**: Public datasets showing improvement
- **Scientific justification**: Why this threshold/method?

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/PiEEG-agent
cd PiEEG-agent

# Install with dev dependencies
pip install -e ".[dev,server]"

# Run tests before making changes
pytest tests/

# Make your changes, add tests, verify
pytest tests/
```

### Code Style

- **Honest metrics**: Clearly document within-session vs generalizable claims
- **No blocking**: LSL ingestion must never wait on LLM/network
- **Language-sized**: Tools return events/verdicts, not voltage arrays
- **Typed interfaces**: Use dataclasses/TypedDicts for tool contracts
- **Decoupled layers**: Perception works without LLM, LLM works without device control

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] New features have tests
- [ ] Documentation updated (README, docstrings)
- [ ] Metrics have citations (if scientific claims)
- [ ] No vendor SDK dependencies (use stdlib HTTP)

---

## 📋 Appendix: Environment Variables Reference

All configuration via environment variables (optional, defaults shown):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIEEG_LSL_NAME` | `PiEEG` | LSL stream name |
| `PIEEG_LSL_TYPE` | `EEG` | LSL stream type |
| `PIEEG_LSL_RESOLVE_BY` | `type` | Resolve by `name` or `type` |
| `PIEEG_RING_SECONDS` | `60` | Ring buffer depth (seconds) |
| `PIEEG_LLM_PROVIDER` | `anthropic` | LLM provider (see Supported Providers) |
| `PIEEG_LLM_MODEL` | provider default | Model override (e.g. `gpt-4`, `llama3.2`) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key |
| `TOGETHER_API_KEY` | — | Together AI API key |
| `PIEEG_WS_URL` | `ws://localhost:1616` | PiEEG-server WebSocket URL |
| `PIEEG_WS_TOKEN` | — | WebSocket auth token (if server requires) |
| `PIEEG_AUDIT_LOG` | `~/.pieeg-agent/audit.jsonl` | Action audit log path |

---

## 📂 Project Structure

```
pieeg_agent/
  config.py              # Configuration + provider registry
  __main__.py            # CLI entry point
  
  ingest/                # LSL data ingestion (no LLM dependency)
    lsl_inlet.py         #   Background LSL thread
    ring.py              #   Thread-safe ring buffer (60s default)
  
  perceive/              # Signal processing cascade (pure DSP)
    features.py          #   T1: FFT → band powers @ 8 Hz
    quality.py           #   Per-channel quality verdicts
    state.py             #   T2: Focus/relax/engagement @ 1 Hz
    events.py            #   T3: Sparse transitions (debounced)
    artifacts.py         #   Blink/jaw/movement detection
    cascade.py           #   Cascade orchestration thread
  
  decode/                # Lab notebook ML features
    patterns.py          #   Pattern classifier training
    classifier.py        #   L2 + group-lasso implementation
    connectivity.py      #   Cross-channel coupling
    session.py           #   Session capture + comparison
    store.py             #   Pattern/session persistence
  
  llm/                   # LLM provider abstraction (no SDKs)
    provider.py          #   Provider interface
    anthropic.py         #   Anthropic Messages API
    openai_compat.py     #   OpenAI-compatible providers
    factory.py           #   Provider selection
    echo.py              #   Debug/testing provider
  
  agent/                 # Copilot + tool dispatch
    tools.py             #   Read-only neural tools
    decode_tools.py      #   Pattern/connectivity/session tools
    actuator_tools.py    #   Gated device actions
    copilot.py           #   Conversation loop
  
  server/                # Device control (optional)
    client.py            #   WebSocket client (sync)
    gate.py              #   Safety gate + audit log
    actions.py           #   Typed action facade
  
  web/                   # Web UI (optional)
    app.py               #   FastAPI backend
    engine.py            #   WebSocket state streaming
    frontend/            #   Vite + React/TypeScript
```
