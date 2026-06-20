# OpenRouter Model Catalog

## Single Source of Truth

**This catalog is the ONLY place model IDs should be defined.** It auto-updates via the pre-commit hook.

### How It Works

1. **Pre-commit hook** runs `scripts/update_models.py`
2. Script fetches https://openrouter.ai/api/v1/models
3. Trims to essential fields (id, name, price, capabilities)
4. Writes to `openrouter_models.json`
5. Commit only proceeds if catalog is fresh

### Why OpenRouter?

- **Auto-tracking aliases**: `~anthropic/claude-sonnet-latest` always points to newest Sonnet
- **All providers aggregated**: Anthropic, OpenAI, Google, Meta, etc. in one API
- **Never deprecates**: Model IDs stay valid, -latest aliases auto-update
- **Price transparency**: Every model has input/output pricing

### Model Deprecation Strategy

**PREFERRED: Use OpenRouter**
- Default provider is `openrouter`
- Uses `-latest` aliases that auto-track
- Zero maintenance needed

**Direct APIs (anthropic, openai, groq, together)**
- Need periodic manual updates to `AUTO_FALLBACK_MODELS`
- Check vendor deprecation docs every 6 months
- Anthropic: https://docs.anthropic.com/en/docs/resources-and-support/deprecations
- OpenAI: https://platform.openai.com/docs/deprecations

### Maintenance

**Never hardcode model IDs in code.** Use:

```python
from pieeg_agent.llm.catalog import get_recommended_model, find_fallback

# Get current recommended model
model = get_recommended_model("anthropic", tier="balanced")

# Find fallback automatically
fallback = find_fallback(model)
```

**Update catalog manually (if needed):**
```bash
python scripts/update_models.py
```

**Validate catalog freshness:**
```bash
git diff pieeg_agent/data/openrouter_models.json
```

If models are >1 month old, re-run the update script.
