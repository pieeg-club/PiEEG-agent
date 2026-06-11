# LLM Provider Guide — Avoiding Rate Limits

This guide explains how to configure PiEEG Agent to avoid rate limit errors and use local/offline LLM providers.

## The Problem

When using cloud LLM providers (Anthropic, OpenAI, etc.), you may encounter rate limit errors like:

> ⚠️ Rate limit exceeded: The API is receiving too many requests. Please wait a moment and try again.

## Solutions (in order of recommendation)

### 1. Use Local LLM Providers (Best for Privacy & Reliability)

PiEEG Agent supports fully local LLM inference — no internet required, no rate limits, completely private.

#### Option A: Ollama (Recommended)

**Install Ollama:**
1. Download from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3.1`
3. Configure PiEEG Agent:
   ```bash
   export PIEEG_LLM_PROVIDER=ollama
   python -m pieeg_agent web
   ```

**Why Ollama:**
- Simple installation (one command)
- Fast inference on consumer GPUs
- Wide model selection (Llama 3, Mistral, CodeLlama, etc.)
- Active community

#### Option B: LM Studio

**Install LM Studio:**
1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load any GGUF model from their library
3. Start the local server (default: `http://localhost:1234`)
4. Configure PiEEG Agent:
   ```bash
   export PIEEG_LLM_PROVIDER=lmstudio
   python -m pieeg_agent web
   ```

**Why LM Studio:**
- User-friendly GUI
- No command-line needed
- Supports Mac/Windows/Linux
- Built-in model downloader

### 2. Switch to Higher-Limit Cloud Providers

If you prefer cloud providers but need higher limits:

#### Groq (Fastest, Generous Free Tier)

```bash
export PIEEG_LLM_PROVIDER=groq
export GROQ_API_KEY=your_key_here
python -m pieeg_agent web
```

**Features:**
- Free tier: 14,400 requests/day (vs Anthropic's ~50/day on free tier)
- Ultra-fast inference (LPU architecture)
- Llama 3.3 70B model

#### Together AI (Good Balance)

```bash
export PIEEG_LLM_PROVIDER=together
export TOGETHER_API_KEY=your_key_here
python -m pieeg_agent web
```

**Features:**
- Pay-as-you-go with reasonable rates
- Wide model selection
- Good uptime

### 3. Chrome AI Fallback (Automatic Client-Side Rescue)

**NEW**: PiEEG Agent now includes automatic Chrome AI fallback when the backend is rate-limited.

#### How It Works

1. You send a chat message
2. Backend LLM hits rate limit
3. Frontend detects the error
4. Falls back to Chrome's built-in AI (Gemini Nano)
5. Response generated entirely client-side (no internet needed)
6. User sees: *"Backend rate-limited — using Chrome AI fallback (Gemini Nano)…"*

#### Setup

1. **Browser Requirements:**
   - Chrome 127+ (Canary/Dev/Beta channels)
   - Supported device (x86_64 with 4GB+ RAM)

2. **Enable the Feature:**
   - Open `chrome://flags/#prompt-api-for-gemini-nano`
   - Set to **"Enabled"**
   - Restart Chrome
   - The API uses the global `ai` object (modern API)

3. **First Use:**
   - Model downloads automatically (~1.5 GB)
   - Happens in background on first fallback
   - Cached locally afterward

4. **Check Status:**
   - Click LLM settings icon in PiEEG Agent UI
   - See "Chrome AI Fallback" section
   - Green dot = ready
   - Yellow dot = needs download
   - Gray dot = unavailable

#### Limitations

- **Context window**: Smaller than cloud models (~1024 tokens response)
- **No tool calling**: Can answer questions but can't execute EEG analysis tools
- **Device-dependent**: Not all devices support on-device inference

#### When to Use

- **Primary use**: Automatic fallback when cloud APIs fail
- **Good for**: Simple Q&A, signal quality questions, troubleshooting advice
- **Not for**: Complex multi-turn conversations requiring tool use (pattern training, connectivity analysis)

## Configuration Reference

### Environment Variables

```bash
# Provider selection
PIEEG_LLM_PROVIDER=anthropic|openai|groq|together|ollama|lmstudio|echo

# Model selection (optional, uses provider default if omitted)
PIEEG_LLM_MODEL=model_name_here

# API keys (only needed for cloud providers)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
```

### Runtime Configuration

You can also change providers without restarting:

1. Click the **LLM settings** icon in the UI header
2. Select provider
3. Choose model
4. Enter API key (if needed)
5. Save (restart required)

## Comparison Matrix

| Provider | Cost | Rate Limits | Latency | Privacy | Tool Calling | Setup |
|----------|------|-------------|---------|---------|--------------|-------|
| **Ollama** | Free | None | Fast* | 100% local | Yes | Easy |
| **LM Studio** | Free | None | Fast* | 100% local | Yes | Easy |
| **Chrome AI** | Free | None | Ultra-fast | 100% local | **No** | Medium |
| **Groq** | Free tier | High | Ultra-fast | Cloud | Yes | Easy |
| **Together** | Paid | Medium | Fast | Cloud | Yes | Easy |
| **Anthropic** | Paid | Low (free tier) | Medium | Cloud | Yes | Easy |
| **OpenAI** | Paid | Medium | Medium | Cloud | Yes | Easy |

*Latency depends on your hardware (GPU/CPU)

## Troubleshooting

### "Rate limit exceeded" still appearing

**If using cloud provider:**
- Switch to Ollama/LM Studio (no limits)
- Or wait 1-5 minutes between requests
- Or upgrade to paid tier with higher limits

**If Chrome AI fallback isn't working:**
- Check `chrome://flags/#prompt-api-for-gemini-nano` is enabled
- Restart Chrome
- Check LLM settings → Chrome AI status
- Try a different Chrome channel (Canary/Dev)

### Ollama "connection refused"

```bash
# Check Ollama is running
ollama list

# Start Ollama service (if not running)
ollama serve

# Test connection
curl http://localhost:11434/v1/models
```

### LM Studio not connecting

1. Open LM Studio
2. Click "Local Server" tab
3. Click "Start Server"
4. Verify URL is `http://localhost:1234`
5. Check firewall isn't blocking port 1234

### Chrome AI shows "unavailable"

**Possible causes:**
- Chrome version too old (need 127+)
- Device not supported (check Chrome release notes)
- Flag not enabled (see Setup section above)
- Not using Chrome (Chromium-based browsers may not work)

## Recommendations

**For most users:**
- Start with **Ollama** (best balance of performance, privacy, and reliability)
- Enable **Chrome AI** as emergency fallback

**For experimentation:**
- Use **Groq** free tier (fastest cloud option)

**For production:**
- Use **Ollama** with a quantized Llama 3.1 model
- Falls back to Chrome AI if Ollama crashes
- No rate limits, no costs, full privacy

**For cloud preference:**
- **Together AI** (good model selection, reasonable rates)
- **Groq** (fastest, but limited model options)

## Next Steps

1. Choose a provider from the options above
2. Set environment variables or use UI configuration
3. Restart PiEEG Agent
4. Test with "What is my brain doing right now?"
5. Monitor LLM settings → Chrome AI status
6. Enjoy rate-limit-free conversations!
