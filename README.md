# PiEEG-agent

A provider-agnostic LLM agent that **perceives live brain activity** from
[PiEEG-server](../PiEEG-server) over Lab Streaming Layer (LSL), reduces the
high-rate signal into language-sized neural state and events, **reasons** with
a pluggable LLM (Anthropic by default), and **acts** through PiEEG-server's
control plane.

> Status: **Phase 3 — gated device actions.** The agent can now *act* on the
> device through PiEEG-server's WebSocket control plane — set the band-pass
> filter, start/stop recording, drive OSC output, apply register presets — and
> every mutating call is **gated**: allowlisted, dry-run by default for the
> copilot, cooldown-limited and audited. Use it directly (`pieeg-agent control
> …`) or give the copilot opt-in hands (`chat --allow-actions`).

## Why a reduction cascade

An LLM reasons in seconds and costs tokens; EEG arrives at 250–500 Hz × 8–32
channels. Raw samples can never go in a prompt. The agent progressively trades
temporal resolution for semantic density:

| Tier | Rate | Representation | Reaches the LLM? |
|------|------|----------------|------------------|
| T0 raw | 250–500 Hz | float32 µV frames | never |
| T1 features | 4–10 Hz | band powers, ratios, quality | never (aggregated) |
| T2 state | ~1 Hz | focus / relaxation / artifacts (smoothed) | on demand |
| T3 events | sparse | debounced transitions & epochs | **yes — primary input** |
| T4 reasoning | on event/query | NL + tool calls | — |

Three rules: **ingestion never blocks on the LLM**, **the LLM pulls via tools**
(it is never pushed raw data), and **everything the model sees is already
language-sized**.

## Install (dev)

```bash
pip install -e .                  # core: perception + copilot (Anthropic/OpenAI via HTTP)
pip install -e ".[server]"        # adds the PiEEG-server control-plane client (Phase 3)
```

The LLM adapters talk to the providers over plain HTTP, so **no vendor SDK is
required** — the default Anthropic backend and every OpenAI-compatible backend
(OpenAI, Groq, Together, Ollama, LM Studio) work out of the box with just a key.

Requires `pylsl` (pulls in the native `liblsl`). EEG perception needs no
hardware — drive it with the mock server.

## Prove the intake (Phase 0)

```bash
# Terminal 1 — the producer (synthetic EEG, no hardware):
pieeg-server --mock --lsl

# Terminal 2 — the agent's perception intake:
pieeg-agent streams                 # discover LSL outlets
pieeg-agent ingest --seconds 10     # drain into the ring, print live stats
pieeg-agent config                  # show resolved settings + provider
```

`ingest` exits 0 only if the consumer keeps up with the stream (no losses, no
growing backlog) — a real end-to-end check of the high-rate consumer.

## Watch the brain (Phase 1)

```bash
# Terminal 1 — the producer:
pieeg-server --mock --lsl

# Terminal 2 — the perception cascade:
pieeg-agent monitor --seconds 20          # live state + events to the console
pieeg-agent monitor --mains 60 --quiet    # 60 Hz line-noise check, events only
```

A PiEEG profile can publish several outlets that all advertise type `EEG`
(`EEG_PiEEG`, `EOG_PiEEG`, `AUX_PiEEG`). `monitor` discovers them all and
auto-selects the brain-EEG group; pass `--by name --name EEG_PiEEG` to pin one.

Each second prints a language-sized snapshot:

```
10:04:36 | focus 0.62 relax 0.41 engage 0.55 | d0.03 t0.01 a0.94 b0.01 g0.01 | dom Alpha | Q 1.00 clean
   >> 10:04:48  relax_high  -  relax rose to 0.74
   !! 10:05:02  quality_drop  -  signal quality degraded to 0.42 (Ch2)
```

A leading `~` marks the warm-up window: the focus/relax/engagement indices are
**within-session relative** (0…1 against a rolling range), so until the signal
has shown some spread they honestly read 0.50. They say "high for you, right
now" — not an absolute or clinical measure.

## Talk to the brain (Phase 2)

The copilot reasons over the same cascade. It needs an LLM provider — the
default is Anthropic, but any OpenAI-compatible backend works (set
`--provider`/`PIEEG_LLM_PROVIDER`), including local ones (Ollama, LM Studio)
that need no key.

```bash
export ANTHROPIC_API_KEY=sk-…            # or use --provider ollama, etc.
pieeg-server --mock --lsl               # Terminal 1

pieeg-agent ask "am I relaxed right now?"   # one-shot question
pieeg-agent chat                            # interactive session
pieeg-agent ask --provider groq --model llama-3.3-70b-versatile "how's my signal?"
```

The model never sees raw EEG. It calls **read-only tools** that pull from the
live cascade:

| Tool | Returns |
|------|---------|
| `get_neural_state` | smoothed focus / relax / engagement, dominant band, quality |
| `get_band_powers` | relative band powers (optionally per channel) |
| `get_recent_events` | the debounced event log |
| `get_channel_quality` | per-channel verdicts (good/flat/rail/noisy/line) |
| `summarize_last` | a one-line status string |

A typical session — the copilot keeps context across turns, so follow-ups just
work:

```
you > how's my signal looking?
  (consulted: get_channel_quality)
copilot > Clean. All four channels read "good", line-noise is low and overall
quality is 1.00 — nothing flat or railing, so the indices below are
trustworthy.

you > ok, am I focused?
  (consulted: get_neural_state)
copilot > Focus is low for you right now (0.34) and alpha is the dominant band,
which usually means a relaxed, eyes-resting state rather than active
concentration.

you > did anything change in the last minute?
  (consulted: get_recent_events)
copilot > One shift: relaxation rose to 0.74 about 40 s ago. No quality drops
since then, so that looks like a real change rather than an artifact.
```

Right after start-up you get the honest-metrics guardrails instead of a number:

```
you > am I focused?
  (consulted: get_neural_state)
copilot > The stream is still warming up, so I can't give you a meaningful
reading yet — the indices calibrate against your own range and currently sit at
a neutral 0.50. Give it a few more seconds.
```

The system prompt holds the copilot to that honest-metrics line: indices are
within-session relative, warm-up and poor signal quality are surfaced before
any conclusion, and every claim about the brain must come from a tool call.
The provider layer is plain HTTP — **no vendor SDK is imported** — so swapping
backends is just config.

## Act on the device (Phase 3)

Reading the brain is free; *acting* on the device is not. The agent talks to
PiEEG-server's WebSocket control plane (`ws://localhost:1616`) through a gate
that enforces three independent checks and audits every attempt:

- **allowlist** — only explicitly enabled actions can run (default: none),
- **dry-run** — an allowed action is *previewed* (what would be sent), never
  executed — the safe default for the copilot,
- **cooldown** — a minimum interval between real executions of an action.

### Direct control (explicit, human-invoked)

```bash
pieeg-server --mock --lsl                       # Terminal 1

pieeg-agent control status                       # read the server snapshot
pieeg-agent control set-filter --lowcut 1 --highcut 40
pieeg-agent control set-filter --off --dry-run   # preview without sending
pieeg-agent control record start                 # … and: record stop
pieeg-agent control reg-preset test_signal       # ADS1299 square-wave self-test
pieeg-agent control osc start --host 127.0.0.1 --port 9000
pieeg-agent control webhooks                      # list configured rules
pieeg-agent control audit                         # read the recent audit log
```

`control` runs for real by default (you asked); add `--dry-run` to preview.
Each call prints the outcome — `[OK]`, `[DRY-RUN]`, `[DENIED]` or `[ERROR]`.

### Copilot with hands (opt-in, gated)

```bash
pieeg-agent chat --allow-actions                 # dry-run preview by default
pieeg-agent chat --allow-actions --execute       # actually perform actions
```

`--allow-actions` adds a curated, safe set of control tools to the copilot
(filter, recording, OSC, register *presets* — raw register writes are never
exposed). Without `--execute` the copilot can only *preview* actions, so you
can watch what it would do before letting it do anything. A session that is
given hands uses a stricter system prompt: act only when asked, read status
before changing it, never retry in a loop, and always report the gate's real
outcome (executed vs previewed vs denied).

| Tool | Effect | Gated? |
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
