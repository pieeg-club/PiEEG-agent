# PiEEG-agent

**Natural language EEG lab notebook.** Train pattern classifiers, analyze connectivity, compare sessions — all by talking to an AI copilot that reads your live brain signals.

Reads from any [Lab Streaming Layer](https://labstreaminglayer.org) (LSL) EEG source. Optional control plane for [PiEEG](https://pieeg.com) hardware. Works with synthetic signal for development.

---

## ⚡ 60-Second Start

```bash
# Terminal 1: Start mock EEG stream
pieeg-server --mock --lsl

# Terminal 2: Launch web UI
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...  # or use --provider echo for testing
pieeg-agent web
```

Open http://localhost:8000 → Chat with your brain data in real-time.

**No hardware?** Mock server generates realistic multi-channel EEG over LSL.

**No API key?** If you run `pieeg-agent web` (or `chat`, `ask`) without configuring an LLM provider, you'll get an **interactive setup wizard**:

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
```

The wizard sets environment variables for your session — **no files are written**. To persist configuration, set the env vars in your shell profile or use `.env` files.

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

## 🧠 Architecture: Perceive → Reason → Act

```
┌──────────────────────────────────────────────────────────────┐
│  Any LSL EEG source (PiEEG, OpenBCI, Muse, mock, etc.)      │
│    └─> Lab Streaming Layer (LSL) outlet                      │
└────────────────────────┬─────────────────────────────────────┘
                         │ High-rate multi-channel EEG
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  PERCEIVE: Ring buffer + cascade                             │
│    • LSLInlet: chunked pulls, no backpressure                │
│    • Features: FFT → band powers (δθαβγ), quality scores     │
│    • State: EMA smoothing → focus/relax/engagement [0-1]     │
│    • Events: debounced transitions ("focus_high" @timestamp) │
│    • Artifacts: blink/jaw/movement detection                 │
└────────────────────────┬─────────────────────────────────────┘
                         │ 1 Hz state + sparse events
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  DECODE: ML + lab features                                   │
│    • Patterns: L2+group-lasso binary classifiers (LORO-CV)   │
│    • Connectivity: cross-channel amplitude coupling (r)      │
│    • Sessions: labelled windows + Cohen's d comparison       │
└────────────────────────┬─────────────────────────────────────┘
                         │ tool-callable summaries
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  REASON: LLM copilot (provider-agnostic)                     │
│    • Tools: neural state, patterns, connectivity, sessions   │
│    • Providers: Anthropic, OpenAI, Groq, Ollama, LM Studio   │
│    • No vendor SDK — plain HTTP via stdlib                   │
│    • Web UI: FastAPI + React/TS (same copilot backend)       │
└────────────────────────┬─────────────────────────────────────┘
                         │ tool calls
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  ACT: Gated control plane (opt-in, PiEEG-server only)        │
│    • WebSocket → PiEEG-server :1616                          │
│    • Allowlist + dry-run + cooldown + audit log              │
│    • Filter, recording, OSC, register presets                │
└──────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- **Scientifically honest**: Indices are within-session relative, not clinical claims. Warm-up and poor signal are surfaced before giving numbers.
- **Debuggable**: Channel-level quality verdicts, event logs with timestamps, tool call traces.
- **Decoupled**: Swap LLM providers without changing perception. Run perception without LLM. Test with mock signal.
- **Lab-grade features**: Pattern training (regularized classifiers + LORO-CV), connectivity analysis (amplitude coupling), session comparison (Cohen's d) — all in natural language.
- **Safe by default**: Device actions require opt-in flags and are audited.

---

## 📖 Usage Examples

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
pieeg-agent web
# → http://localhost:8765
```

Opens a browser-based interface with:
- **Live chat** — same multi-turn conversation as CLI, with streaming responses
- **Brain state cards** — real-time focus/relax/engagement, band powers, quality, connectivity
- **Pattern training UI** — record/compare mental states with visual progress
- **Artifact feed** — scrolling log of signal events

All powered by the SAME copilot + cascade + tools as CLI — web is just another front-end.

Backend: FastAPI + WebSocket streams (snapshot @ 2 Hz, chat events)  
Frontend: Vite + React/TypeScript, single-page app, dark theme

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

### 7. Device Control (PiEEG-server only)

**Optional control plane** for PiEEG hardware. Not needed for other LSL sources.

**Direct control** (you invoke explicitly):
```bash
pieeg-agent control status                          # read device state
pieeg-agent control set-filter --lowcut 1 --highcut 40
pieeg-agent control record start
pieeg-agent control osc start --host 127.0.0.1 --port 9000
pieeg-agent control audit                            # see action log
```

**Copilot with hands** (opt-in, dry-run by default):
```bash
pieeg-agent chat --allow-actions                    # preview only
pieeg-agent chat --allow-actions --execute          # actually act
```

Every action passes through a gate:
- **Allowlist** — only permitted actions can run
- **Dry-run** — previewed, not executed (default for copilot)
- **Cooldown** — minimum interval between executions
- **Audit** — logged with timestamp, user, outcome

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

### Interactive Setup

When you run a command that needs an LLM provider (`web`, `chat`, `ask`) without proper configuration, you'll get an **interactive setup wizard** that guides you through:

1. **Provider selection** — Choose from 7 providers (cloud or local)
2. **API key input** — Secure prompt (hidden input) for cloud providers
3. **Automatic env var setup** — Sets variables for your session

The wizard only runs in interactive terminals (when `stdin.isatty()` is true). For non-interactive environments (CI, Docker, systemd), set environment variables directly.

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

```bash
# Install with dev dependencies
pip install -e ".[dev,server]"

# Run tests
pytest tests/

# Test with mock signal (deterministic)
pieeg-server --mock --lsl
pieeg-agent monitor --seconds 10

# Test with echo provider (no API key needed)
pieeg-agent web --provider echo                    # web UI with debug mode
pieeg-agent chat --provider echo                   # CLI chat, tool calls are real
pieeg-agent ask --provider echo "what is my focus level?"  # triggers get_neural_state tool

# Test provider without EEG stream
python -m pieeg_agent.llm.factory --provider anthropic
```

**Key modules:**
- `ingest/` — LSL intake, ring buffer (no LLM dependency)
- `perceive/` — Feature extraction, state estimation, events (pure DSP)
- `llm/` — Provider abstraction, HTTP-only (no vendor SDKs)
- `agent/` — Copilot, tool dispatch, conversation loop
- `server/` — WebSocket client, action gate, audit log

**Tests validate:**
- Ring buffer handles overwrite correctly
- Feature extraction matches known FFT results
- Event detector hysteresis prevents flicker
- Provider wire format (request/response, tool calling)
- Gate enforcement (allowlist, cooldown, dry-run)

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

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Contributing

Issues and PRs welcome. Please include:
- **For bugs**: `pieeg-agent config` output, LSL stream info, error trace
- **For features**: Use case, proposed CLI/API, why current tools don't suffice
- **For metrics**: Citations for thresholds, validation on public datasets
|------|--------|--------|
| `server_status` | read sample rate, channels, filter, recording, LSL state | read-only |
| `list_webhooks` | list configured webhook rules | read-only |
| `set_filter` | enable/disable/retune the band-pass filter | yes |
| `start_recording` / `stop_recording` | server-side CSV recording | yes |
| `apply_register_preset` | ADS1299 preset (normal / short / test / temp) | yes |
| `start_osc` / `stop_osc` | OSC output stream (e.g. to VRChat) | yes |

In the default `--allow-actions` mode (no `--execute`) the copilot can *reach*
for an action but only ever previews it — you see exactly what it would do:

```
you > start recording
  (consulted: start_recording)
copilot > I held that back — this session is in dry-run mode, so I previewed
start_recording instead of sending it. Re-run me with --execute if you want
that to actually start the recording.

you > what's the filter set to right now?
  (consulted: server_status)
copilot > The band-pass filter is on. I can retune it, but only as a preview
unless you give me --execute.
```

With `--execute`, the same requests run for real and the copilot reports the
gate's actual outcome — and a repeat of the same action is refused while it's
on cooldown:

```
you > start recording, then drop the band-pass to 1–30 Hz
  (consulted: start_recording, set_filter)
copilot > Done. Recording is running (writing a CSV on the server) and the
band-pass is now 1–30 Hz.

you > actually, start a fresh recording
  (consulted: start_recording)
copilot > That was denied — start_recording is on cooldown for a couple more
seconds to stop repeat actions firing back-to-back. Ask me again shortly.
```

Every attempt — previewed, executed or denied — is written to an audit log, so
there's always a record of what the agent did or tried to do. By default it
persists to `~/.pieeg-agent/audit.jsonl` (override with `--audit-log PATH` or
`$PIEEG_AUDIT_LOG`, or turn it off with `--no-audit-log`). Read it back with:

```bash
pieeg-agent control audit            # last 20 attempts (newest last)
pieeg-agent control audit --limit 5  # just the last 5
```

```
Audit log: ~/.pieeg-agent/audit.jsonl  (2 entries, showing 2)

  2026-06-08 11:57:47  [     OK] set_filter
  2026-06-08 11:57:49  [DRY-RUN] reg_preset  -  dry-run: not sent to the device
```

`control audit` is a local file read — it never touches the server.

Install the control-plane client with `pip install -e ".[server]"` (it pulls
in `websockets`). The client is synchronous — a background reader thread that
drops EEG data frames and demuxes the server's broadcast replies by key — so it
matches the rest of the agent (no asyncio).

## Configuration

Environment variables (all optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIEEG_LSL_NAME` | `PiEEG` | LSL stream name |
| `PIEEG_LSL_TYPE` | `EEG` | LSL stream type |
| `PIEEG_LSL_RESOLVE_BY` | `type` | resolve by `name` or `type` |
| `PIEEG_RING_SECONDS` | `60` | short-term memory depth |
| `PIEEG_LLM_PROVIDER` | `anthropic` | `anthropic`/`openai`/`groq`/`together`/`ollama`/`lmstudio` |
| `PIEEG_LLM_MODEL` | per provider | model id override |
| `ANTHROPIC_API_KEY` | — | key for the default provider |
| `PIEEG_WS_URL` | `ws://localhost:1616` | PiEEG-server control plane |
| `PIEEG_WS_TOKEN` | — | control-plane auth token, if the server requires one |
| `PIEEG_AUDIT_LOG` | `~/.pieeg-agent/audit.jsonl` | where gated actions are recorded |

## Layout

```
pieeg_agent/
  config.py            # settings + provider registry
  ingest/
    ring.py            # thread-safe ring buffer (short-term memory)
    lsl_inlet.py       # background LSL inlet thread + multi-group discovery
  perceive/
    features.py        # T1 sliding-FFT band powers (per channel)
    quality.py         # per-channel signal-quality verdicts
    state.py           # T2 smoothed NeuralState (~1 Hz)
    events.py          # T3 debounced transitions (Schmitt + min-dwell)
    cascade.py         # the perception thread wiring it together
  llm/
    provider.py        # provider-agnostic interface (the contract)
    anthropic.py       # native Anthropic Messages API adapter
    openai_compat.py   # OpenAI-compatible chat-completions adapter
    factory.py         # get_provider(config) — selects an adapter by kind
  agent/
    tools.py           # read-only neural tools (+ Toolset / CombinedToolset)
    actuator_tools.py  # opt-in gated control tools (the agent's hands)
    copilot.py         # the tool-using conversational loop
  server/
    client.py          # synchronous WebSocket control client (reply demux)
    gate.py            # allowlist / dry-run / cooldown + audit log
    actions.py         # typed reads + gated actions facade
  __main__.py          # CLI: streams · ingest · monitor · ask · chat · control · config
```
