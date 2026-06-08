# PiEEG-agent

A provider-agnostic LLM agent that **perceives live brain activity** from
[PiEEG-server](../PiEEG-server) over Lab Streaming Layer (LSL), reduces the
high-rate signal into language-sized neural state and events, **reasons** with
a pluggable LLM (Anthropic by default), and **acts** through PiEEG-server's
control plane.

> Status: **Phase 1 — perception cascade.** On top of the Phase 0 intake, the
> agent now reduces the raw stream into per-channel band powers, signal-quality
> verdicts, a smoothed ~1 Hz neural state, and debounced events — all printed
> live, no LLM yet. The reasoning loop and server actions land in later phases.

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
pip install -e ".[anthropic]"     # default provider
# or .[openai] for OpenAI-compatible backends (OpenAI, Groq, Together, Ollama, LM Studio)
```

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
    provider.py        # provider-agnostic interface (contract)
  __main__.py          # CLI: streams · ingest · monitor · config
```

## License

MIT — matches the sibling PiEEG-server.
