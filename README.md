# PiEEG-agent

A provider-agnostic LLM agent that **perceives live brain activity** from
[PiEEG-server](../PiEEG-server) over Lab Streaming Layer (LSL), reduces the
high-rate signal into language-sized neural state and events, **reasons** with
a pluggable LLM (Anthropic by default), and **acts** through PiEEG-server's
control plane.

> Status: **Phase 0 — ingestion spine.** The LSL intake, ring buffer,
> configuration, and the provider-agnostic LLM interface are in place. The
> reasoning loop and server actions land in later phases.

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

`ingest` exits 0 only if the recent sample rate keeps up with the stream's
nominal rate — a real end-to-end check of the high-rate consumer.

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
    lsl_inlet.py       # background LSL inlet thread
  llm/
    provider.py        # provider-agnostic interface (contract)
  __main__.py          # CLI: streams · ingest · config
```

## License

MIT — matches the sibling PiEEG-server.
