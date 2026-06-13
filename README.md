<div align="center">
  
# PiEEG-agent

[![Test](https://github.com/pieeg-club/PiEEG-agent/actions/workflows/test.yml/badge.svg)](https://github.com/pieeg-club/PiEEG-agent/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![LSL](https://img.shields.io/badge/LSL-compatible-green)](https://labstreaminglayer.org/)
[![Platform](https://img.shields.io/badge/platform-PiEEG%20|%20Any%20LSL%20Device-c51a4a)](https://pieeg.com)
[![Discord](https://img.shields.io/discord/1059637443548987462?color=5865F2&logo=discord&logoColor=white&label=Discord)](https://discord.gg/neJ45FR6Sv)

**Natural language EEG lab notebook.** Train pattern classifiers, analyze connectivity, compare sessions — all by talking to an AI copilot that reads your live brain signals.

<img width="1918" alt="PiEEG-agent web interface showing chat, brain state cards, and pattern training" src="https://github.com/user-attachments/assets/2b643271-4904-4d37-a149-7f8c91163528" />

</div>

---

## 📦 Quick Start

### One-Command Install

**Linux / macOS / WSL:**
```bash
# One-line remote install (recommended)
curl -sSL https://raw.githubusercontent.com/pieeg-club/PiEEG-agent/main/install.sh | bash

# Or clone and setup manually
git clone https://github.com/pieeg-club/PiEEG-agent.git
cd PiEEG-agent
chmod +x setup.sh && ./setup.sh
```

**Windows:**
```cmd
git clone https://github.com/pieeg-club/PiEEG-agent.git
cd PiEEG-agent
install.cmd
```

**What it does:**
- ✅ Creates Python virtual environment
- ✅ Installs all dependencies
- ✅ Sets up command-line launcher
- ✅ Runs verification tests

**No Node.js required** — the React frontend is prebuilt and included.

### Manual Install

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package
pip install -e ".[web,dev]"
```

---

## ⚡ Usage

### Web Interface (Recommended)

```bash
# Start with any LSL stream (PiEEG, OpenBCI, Muse, etc.)
pieeg-agent web

# Or test with synthetic data (no hardware)
pieeg-server --mock --lsl  # Terminal 1
pieeg-agent web            # Terminal 2
```

Open **http://localhost:8000** for:
- 💬 **Chat with brain copilot** — "am I focused?", "train a relaxation pattern"
- 🧠 **Live brain state cards** — focus/relax/engagement, band powers, quality
- 🎨 **Pattern training UI** — record mental states, train classifiers
- 📊 **Connectivity visualization** — cross-channel coupling analysis
- 🎛️ **Device control** (optional) — adjust filters, start recording, enable OSC

**First-time setup:**  
You'll be prompted to select an LLM provider and enter an API key (or use local Ollama/LM Studio). Configuration is saved to `~/.pieeg-agent/config.json`.

### Command-Line Interface

```bash
# Discover LSL streams
pieeg-agent streams

# Monitor live state (no LLM needed)
pieeg-agent monitor

# Ask one-shot questions
pieeg-agent ask "am I focused?"

# Interactive chat
pieeg-agent chat
```

### Device Control (PiEEG Server Integration)

```bash
# Terminal 1: Start PiEEG server with LSL
pieeg-server --lsl

# Terminal 2: Launch agent with device control
pieeg-agent web --allow-actions --execute
```

**Action modes:**
- `pieeg-agent web` — Read-only (monitoring + analysis)
- `pieeg-agent web --allow-actions` — Preview mode (shows what it would do)
- `pieeg-agent web --allow-actions --execute` — Execute mode (actually controls device)

All actions pass through a safety gate with cooldowns, allowlists, and audit logging.

---

## 🧠 What Can It Do?

### 1. Real-Time Brain State Monitoring

**Neural State Tools:**
- `get_neural_state` — Current focus, relaxation, engagement (0-1, session-relative)
- `get_band_powers` — Delta, theta, alpha, beta, gamma power distribution
- `get_channel_quality` — Per-channel signal quality (good/flat/rail/noisy/line)
- `get_recent_events` — Debounced state transitions and quality changes

**Artifact Detection:**
- `find_artifacts` — Eye blinks (single/double), jaw clenches, motion artifacts

**Example:**
```
you > how's my signal?
copilot > All channels read "good", quality 0.98 — trustworthy signal.

you > am I focused or relaxed?
copilot > Relaxed. Alpha is dominant (0.87) and focus is low (0.31).
```

### 2. Pattern Training & Recognition

**Train custom mental state classifiers:**

```bash
you > train a pattern called "meditation"
copilot > Starting pattern training for "meditation". I'll guide you through recording segments.
          
          First, I need a baseline. Sit normally with eyes open.
          Ready to record 'rest' for 4 seconds? (say yes when ready)

you > yes
copilot > [Recording rest...] ✓ Captured 32 frames, quality 0.94.
          
          Now, close your eyes and meditate for 4 seconds.
          Ready? (say yes)

you > yes
copilot > [Recording active...] ✓ Captured 32 frames, quality 0.96.
          
          That's rep 1 of 3. Let me record another rest segment...
          [repeats 2 more times]
          
          Training classifier... Done!
          
          Pattern "meditation" trained:
          - Balanced accuracy: 0.89 (leave-one-rep-out CV)
          - Top cue: Alpha increase in O1/O2 (occipital)
          - Channel importance: O1★★★ O2★★★ C3★ C4★
```

**Features:**
- **L2 + group-lasso regularization** for spatial sparsity
- **Leave-one-rep-out cross-validation** (no temporal leakage)
- **Balanced accuracy** (immune to class imbalance)
- **Channel importance** shows which electrodes matter
- **Live scoring** — patterns activate in real-time after training

**Pattern Tools:**
- `start_pattern_training` — Begin teaching a new pattern
- `record_segment` — Capture labeled rest/active segments
- `finish_pattern_training` — Fit detector and validate
- `list_patterns` — Show all trained patterns with CV scores
- `detect_patterns` — Live activation of all loaded patterns
- `explain_pattern` — Channel importance and feature weights
- `delete_pattern` — Remove a pattern

### 3. Spectral Analysis

**Deep frequency-domain analysis:**

```bash
you > analyze my alpha rhythm
copilot > Spectral analysis:
          - Individual alpha peak (IAF): 10.2 Hz
          - Alpha power: 0.73 (dominant)
          - 1/f slope: -1.1 (typical healthy EEG)
          - Spectral entropy: 0.68 (moderate complexity)
          - Frontal alpha asymmetry: -0.05 (balanced)
```

**analyze_spectrum** returns:
- **Individual alpha frequency (IAF)** — your personal alpha peak
- **Aperiodic 1/f slope** — reflects excitation/inhibition balance
- **Theta/beta ratio** — cognitive load / vigilance marker
- **Spectral entropy** — signal complexity measure
- **Frontal alpha asymmetry** — hemispheric balance (approach/withdrawal)

### 4. Connectivity Analysis

```bash
you > show connectivity in the alpha band
copilot > Alpha-band connectivity (amplitude coupling):
          Mean correlation: 0.34
          Strongest pair: C3↔C4 (r=0.72) — motor cortex synchrony
          Most connected: C3 (6 strong links)
          Least connected: Fp1 (2 weak links)
```

**connectivity** tool computes cross-channel amplitude coupling in any frequency band. Useful for:
- Network topology analysis
- Hemispheric synchrony
- Rest vs. task comparisons

### 5. Session Recording & Comparison

**Capture labeled windows for later comparison:**

```bash
you > record a session called "meditation" for 60 seconds
copilot > [Capturing...] ✓ Session saved.
          Alpha dominant (0.81), high relaxation (0.76)
          C3-C4 coupling: 0.68

you > record "eyes-open-focus" for 60 seconds
[later...]

you > compare meditation vs eyes-open-focus
copilot > Session comparison (within-session Cohen's d):
          
          Alpha power:        +0.92 SD (meditation higher)
          Beta power:         -0.54 SD (meditation lower)
          Relaxation index:   +0.71 SD
          C3-C4 coupling:     +0.45 SD
          
          Meditation shows classic eyes-closed signature:
          increased alpha, decreased beta, stronger bilateral sync.
```

**Session Tools:**
- `record_session` — Capture labeled window with full metrics
- `list_sessions` — Show all recorded sessions
- `session_summary` — Detailed view of one session
- `compare_sessions` — Statistical comparison with effect sizes

### 6. Jupyter Notebook Generation

**Create publication-ready analysis notebooks:**

```bash
you > create a notebook analyzing my meditation pattern
copilot > [Generating notebook...]
          ✓ Created: meditation_analysis.ipynb
          
          Includes:
          - Pattern ROC curve (LORO-CV)
          - Feature importance rankings
          - Per-channel band power evolution
          - Confusion matrix
          - Statistical summary table
```

**create_jupyter_notebook** auto-populates templates with:
- Session data from store
- Pattern classifiers with provenance
- Matplotlib/seaborn visualization code
- Markdown explanations

**Templates available:**
- Session summary (band powers, artifacts, quality)
- Pattern performance (ROC, confusion matrix, feature importance)
- Connectivity heatmaps
- Spectral evolution (time × frequency)

### 7. Web Search & Documentation

**Search neuroscience literature:**

```bash
you > search pubmed for P300 event-related potentials
copilot > Found 5 papers:
          1. "The P300 wave of the human event-related potential"
             [shows snippet + PubMed link]
          ...
```

**Web Tools:**
- `search_web` — Wikipedia or PubMed search
- `fetch_url` — Fetch from trusted domains (scientific publishers, Wikipedia)
- `list_trusted_sources` — Show allowed domains

**Documentation Tools:**
- `search_docs` — Search PiEEG documentation
- `fetch_doc` — Retrieve specific doc pages

---

## 🔬 Scientific Approach: Honest Metrics

This agent is designed for **neurofeedback research and UX prototyping**, not clinical use.

### Within-Session Relative Indices

`focus`, `relax`, `engagement` are **normalized to the session's own range** (rolling 10-minute window). They mean "high *for you, right now*" — not absolute or clinical values.

- **Warm-up period**: Indices show 0.50 until signal variance is established. A `~` prefix marks this, and the copilot says "still warming up" instead of inventing numbers.
- **Quality gates**: Poor electrode contact or artifacts are flagged *before* reporting state.

### Cross-Validated Pattern Classifiers

Pattern training uses:
- **L2 + group-lasso regularization** — drives whole channels to zero if uninformative
- **Leave-one-rep-out CV** — temporally-correlated frames in one rep can't leak into their own test
- **Balanced accuracy** — mean of sensitivity and specificity, immune to class imbalance

**No overfitting claims**: CV scores are reported honestly. If you have 3 reps, you get 3-fold LORO-CV.

### Effect Sizes for Session Comparisons

Session comparisons report **within-session Cohen's d** (standardized mean difference). This is a *descriptive* effect size for your data, **not a generalization or clinical claim**.

**Example caveat in output:**
> "Cohen's d = 0.8 within this session. This does NOT mean the effect generalizes across days, subjects, or conditions."

### Event Detection with Hysteresis

State transitions require:
- Threshold crossing (default: 0.70 for high, 0.30 for low)
- Minimum dwell time (2 seconds — no flicker)
- Quality floor (0.50 — no events during artifacts)

**No spurious events from noise.**

---

## 🏗️ Architecture

### Design Principles

| Principle | Implementation | Why It Matters |
|-----------|----------------|----------------|
| **Ingestion never blocks** | Dedicated LSL thread, ring buffer | No sample loss even if LLM is slow |
| **LLM pulls, never pushed** | Tools request state on demand | Token costs stay sane |
| **Language-sized representations** | Events, not voltage arrays | Models reason about "focus_high", not floats |
| **Scientifically honest metrics** | Within-session normalization, warm-up flags | Users know what numbers mean |
| **Decoupled layers** | Swap providers, run without LLM, test with mock | Debuggable, testable, maintainable |
| **Safe by default** | Dry-run, cooldown, audit log | AI can't spam device commands |

### Perception Cascade

**The challenge**: LLMs reason at ~1 Hz. EEG arrives at 250-500 Hz × 8 channels = 2,000-4,000 samples/second.

**The solution**: Progressive semantic compression

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
1. High-rate intake runs in dedicated thread — never waits on LLM
2. LLM pulls via tools — never spammed with raw data
3. Everything the model sees is language-sized — indices, events, verdicts

### Full System with PiEEG Hardware

```
┌────────────────────────────────────────────────────────────────┐
│  PiEEG Hardware + Server (Raspberry Pi or PC)                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ADS1299 → pieeg-server                                   │ │
│  │   • 8-ch EEG @ 250 Hz                                    │ │
│  │   • Hardware filtering                                   │ │
│  │   • CSV recording                                        │ │
│  │   • LSL broadcast (data)                                 │ │
│  │   • WebSocket :1616 (control)                            │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────┬──────────┘
         │ LSL (data)                                │ WS (control)
         ▼                                           ▼
┌────────────────────────────────────────────────────────────────┐
│  PiEEG-Agent                                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ INGEST: LSL intake → 60s ring buffer                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ PERCEIVE: Cascade (T0→T1→T2→T3)                         │ │
│  │   T1: FFT → band powers @ 8 Hz                           │ │
│  │   T2: State indices @ 1 Hz                               │ │
│  │   T3: Sparse events (debounced)                          │ │
│  │   Artifacts: blink/jaw/motion detection                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ DECODE: Lab notebook tools                               │ │
│  │   • Pattern training (L2+group-lasso, LORO-CV)           │ │
│  │   • Connectivity (cross-channel coupling)                │ │
│  │   • Sessions (capture + compare with Cohen's d)          │ │
│  │   • Spectral (IAF, 1/f slope, theta/beta, entropy)       │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ REASON: LLM copilot                                      │ │
│  │   Providers: Anthropic, OpenAI, Groq, Ollama, etc.       │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ACT: Safety gate (with --allow-actions)                  │ │
│  │   ✓ Allowlist → ✓ Cooldown → ✓ Dry-run → ✓ Audit        │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────────┘
         │ http://localhost:8000
         ▼
┌────────────────────────────────────────────────────────────────┐
│  Web Interface (browser)                                       │
│    • Chat with brain copilot                                   │
│    • Live brain state cards                                    │
│    • Pattern training UI                                       │
│    • System control panel                                      │
└────────────────────────────────────────────────────────────────┘
```

### Module Overview

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `ingest/` | LSL intake, ring buffer | `lsl_inlet.py`, `ring.py` |
| `perceive/` | Feature extraction, state estimation, events | `features.py`, `state.py`, `events.py`, `artifacts.py`, `quality.py`, `cascade.py` |
| `decode/` | Pattern training, connectivity, sessions, spectral | `patterns.py`, `classifier.py`, `connectivity.py`, `session.py`, `spectral.py`, `features.py`, `calibrate.py`, `train.py`, `store.py` |
| `llm/` | Provider abstraction (HTTP-only, no SDKs) | `provider.py`, `anthropic.py`, `openai_compat.py`, `factory.py`, `echo.py` |
| `agent/` | Copilot, tool dispatch | `copilot.py`, `tools.py`, `decode_tools.py`, `actuator_tools.py`, `utility_tools.py`, `web_tools.py`, `doc_tools.py`, `context.py` |
| `server/` | Device control (optional) | `client.py`, `gate.py`, `actions.py` |
| `web/` | FastAPI backend + React frontend | `app.py`, `engine.py`, `frontend/` |

---

## ⚙️ Configuration

### Interactive Setup Wizard

First-time usage launches an interactive setup:

1. **Provider selection** — Anthropic, OpenAI, Groq, Together, Ollama, LM Studio, Echo
2. **API key input** — Secure prompt (hidden) for cloud providers
3. **Save option** — Persist to `~/.pieeg-agent/config.json` (chmod 600)

**Manage configuration:**
```bash
pieeg-agent config          # View current settings
pieeg-agent config reset    # Delete saved config
```

**Non-interactive mode** (Docker, CI):
```bash
export PIEEG_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
pieeg-agent web  # Skips wizard
```

### Supported LLM Providers

| Provider | Type | API Key Needed | Get Key From |
|----------|------|----------------|--------------|
| Anthropic | Cloud | ✅ `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |
| OpenAI | Cloud | ✅ `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Groq | Cloud | ✅ `GROQ_API_KEY` | https://console.groq.com/keys |
| Together AI | Cloud | ✅ `TOGETHER_API_KEY` | https://api.together.xyz/settings/api-keys |
| Ollama | Local | ❌ None | Install from https://ollama.ai |
| LM Studio | Local | ❌ None | Install from https://lmstudio.ai |
| Echo | Debug | ❌ None | Built-in (keyword-based simulator) |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIEEG_LSL_NAME` | `PiEEG` | LSL stream name |
| `PIEEG_LSL_TYPE` | `EEG` | LSL stream type |
| `PIEEG_RING_SECONDS` | `60` | Ring buffer depth (seconds) |
| `PIEEG_LLM_PROVIDER` | `anthropic` | LLM provider |
| `PIEEG_LLM_MODEL` | provider default | Model override |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key |
| `TOGETHER_API_KEY` | — | Together AI API key |
| `PIEEG_WS_URL` | `ws://localhost:1616` | PiEEG-server WebSocket |
| `PIEEG_WS_TOKEN` | — | WebSocket auth token |
| `PIEEG_AUDIT_LOG` | `~/.pieeg-agent/audit.jsonl` | Action audit log path |

---

## 🧪 Development & Testing

### Installation Modes

```bash
# Minimal (monitoring + analysis)
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
# Exit code 0 = no samples lost

# Monitor live (no LLM needed)
pieeg-agent monitor
```

### Testing Without API Keys

```bash
# Echo provider (keyword-based simulator with real tools)
pieeg-agent web --provider echo
pieeg-agent chat --provider echo
pieeg-agent ask --provider echo "am I focused?"
```

### Running Tests

```bash
pytest tests/                    # All tests
pytest tests/test_ring.py        # Specific module
pytest -v --tb=short             # Verbose with short tracebacks
```

**Test coverage includes:**
- Ring buffer overwrite logic
- FFT feature extraction
- Event detector hysteresis
- Pattern classifier LORO-CV
- Connectivity computation
- Session comparison
- Provider wire format
- Gate enforcement (allowlist, cooldown)

---

## 📚 Learn More

- **[PiEEG Hardware](https://pieeg.com)** — Open-source EEG shield for Raspberry Pi
- **[Lab Streaming Layer](https://labstreaminglayer.org)** — Time-sync for multi-modal biosignals
- **[Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)** — How the copilot calls functions
- **[Discord Community](https://discord.gg/neJ45FR6Sv)** — Get help, share projects

---

## 🤝 Contributing

Contributions welcome! Whether fixing bugs, adding features, or improving documentation:

### Reporting Issues

**For bugs**, include:
```bash
pieeg-agent config          # Current configuration
pieeg-agent streams         # LSL stream discovery
# Paste error traceback
```

**For feature requests**, include:
- **Use case**: What problem are you solving?
- **Proposed API**: CLI commands or copilot interactions
- **Why existing tools don't work**: What's missing?

**For scientific/algorithmic changes**, include:
- **Citations**: Papers or validated implementations
- **Test data**: Public datasets showing improvement
- **Justification**: Why this threshold/method?

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/PiEEG-agent
cd PiEEG-agent

# Install with dev dependencies
pip install -e ".[dev,server]"

# Run tests before changes
pytest tests/

# Make changes, add tests, verify
pytest tests/
```

### Code Style

- **Honest metrics**: Clearly document within-session vs generalizable claims
- **No blocking**: LSL ingestion must never wait on LLM/network
- **Language-sized**: Tools return events/verdicts, not voltage arrays
- **Typed interfaces**: Use dataclasses/TypedDicts for tool contracts
- **Decoupled layers**: Perception works without LLM, LLM works without device

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] New features have tests
- [ ] Documentation updated (README, docstrings)
- [ ] Metrics have citations (if scientific claims)
- [ ] No vendor SDK dependencies (use stdlib HTTP)

---

## 🎯 Roadmap

- [x] **High-rate LSL intake** — Chunked pulls, ring buffer
- [x] **Perception cascade** — Features → state → events
- [x] **LLM copilot** — Read-only tools, provider-agnostic
- [x] **Gated device actions** — Allowlist, dry-run, audit
- [x] **Artifact detection** — Blinks, jaw, movement + quality tracking
- [x] **Pattern training** — L2+group-lasso, LORO-CV, balanced accuracy
- [x] **Web UI** — FastAPI + React, chat + live brain cards
- [x] **Connectivity analysis** — Cross-channel coupling
- [x] **Session comparison** — Lab notebook tools
- [x] **Pattern health monitoring** — Confidence scoring, degradation alerts
- [ ] **Jupyter notebook templates** — Auto-generated analysis notebooks
- [ ] **P300 decoder** — Event-related potential for single-trial BCI control
- [ ] **Multi-modal fusion** — EEG + EOG + EMG streams

---

## 📄 License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International) — see [LICENSE](LICENSE)

You are free to share and adapt this work for non-commercial purposes with attribution.

