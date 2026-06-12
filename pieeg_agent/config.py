"""Configuration and LLM provider registry for PiEEG-agent.

Two ideas live here:

1. ``PROVIDERS`` — a *data-only* registry describing how to reach each LLM
   backend. The agent is provider-agnostic: every entry declares a ``kind``
   (``"anthropic"`` or ``"openai"``) that selects the wire adapter
   implemented in :mod:`pieeg_agent.llm` (Phase 2+). Adding a new
   OpenAI-compatible backend is just a new dict entry.

2. ``AgentConfig`` — the resolved runtime settings, assembled from defaults,
   environment variables and explicit overrides. It never *holds* a secret
   longer than needed and keeps the API key out of ``repr``.

No network or LLM imports happen here; this module is safe to import in any
context (CLI, tests, ingestion-only runs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ── Provider registry ──────────────────────────────────────────────────────
#
# ``kind`` maps to a concrete adapter in pieeg_agent.llm:
#   "anthropic" → native Anthropic Messages API
#   "openai"    → OpenAI-compatible /chat/completions (OpenAI, Groq, Together,
#                 Ollama, LM Studio, …) — same wire format, different base_url
#
# Local backends (Ollama, LM Studio) have ``env_key=""`` because they need no
# API key. Models are sensible defaults and can be overridden per run.

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "label": "Anthropic",
        "kind": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-20250514",
    },
    "openai": {
        "label": "OpenAI",
        "kind": "openai",
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "label": "Groq",
        "kind": "openai",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "together": {
        "label": "Together AI",
        "kind": "openai",
        "env_key": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "ollama": {
        "label": "Ollama (local)",
        "kind": "openai",
        "env_key": "",  # no key needed
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
    },
    "lmstudio": {
        "label": "LM Studio (local)",
        "kind": "openai",
        "env_key": "",  # no key needed
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
    },
    "echo": {
        "label": "Echo (debug)",
        "kind": "echo",
        "env_key": "",  # no key needed
        "base_url": "",
        "default_model": "echo-debug-v1",
    },
}

DEFAULT_PROVIDER = "anthropic"

# ── Automatic fallback model mapping ───────────────────────────────────────
#
# When rate limits or errors occur, the system can automatically fall back to
# a smaller/cheaper model from the same provider. This mapping defines the
# automatic fallback for common models. Users can override with env vars.

AUTO_FALLBACK_MODELS: dict[str, str] = {
    # Anthropic: Sonnet/Opus → Haiku
    "claude-sonnet-4-20250514": "claude-3-5-haiku-20241022",
    "claude-opus-4-20250514": "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022": "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229": "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229": "claude-3-haiku-20240307",
    
    # OpenAI: GPT-4 → GPT-4o-mini
    "gpt-4o": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4o-mini",
    "gpt-4": "gpt-4o-mini",
    "o1": "gpt-4o-mini",
    "o1-mini": "gpt-4o-mini",
    
    # Groq: 70B → 8B
    "llama-3.3-70b-versatile": "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile": "llama-3.1-8b-instant",
    "mixtral-8x7b-32768": "llama-3.1-8b-instant",
    
    # Together AI: 70B → 8B
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
}


# ── Runtime configuration ──────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Resolved runtime configuration for one agent process."""

    # ── Perception (LSL inlet) ──────────────────────────────────────────
    lsl_name: str = "PiEEG"
    lsl_type: str = "EEG"
    lsl_resolve_by: str = "type"          # "name" | "type"
    lsl_resolve_timeout: float = 5.0      # seconds
    ring_seconds: float = 60.0            # short-term memory depth

    # ── Reasoning (LLM) ─────────────────────────────────────────────────
    provider: str = DEFAULT_PROVIDER
    model: str = ""                       # resolved from registry if blank
    base_url: str = ""                    # resolved from registry if blank
    api_key: str = field(default="", repr=False)  # never printed
    
    # ── Fallback LLM (optional resilience) ──────────────────────────────
    fallback_provider: str = ""           # e.g., "groq" or "ollama"
    fallback_model: str = ""              # resolved from registry if blank
    fallback_base_url: str = ""           # resolved from registry if blank
    fallback_api_key: str = field(default="", repr=False)

    # ── Action (PiEEG-server control plane; used from Phase 3) ──────────
    ws_url: str = "ws://localhost:1616"
    dashboard_url: str = "http://localhost:1617"

    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, **overrides) -> "AgentConfig":
        """Build a config from environment variables, then apply overrides.

        Recognised environment variables:
          PIEEG_LSL_NAME, PIEEG_LSL_TYPE, PIEEG_LSL_RESOLVE_BY,
          PIEEG_LSL_RESOLVE_TIMEOUT, PIEEG_RING_SECONDS,
          PIEEG_LLM_PROVIDER, PIEEG_LLM_MODEL,
          PIEEG_LLM_FALLBACK_PROVIDER, PIEEG_LLM_FALLBACK_MODEL,
          PIEEG_WS_URL, PIEEG_DASHBOARD_URL,
          plus the selected provider's API-key variable (e.g. ANTHROPIC_API_KEY).

        ``overrides`` (keyword args) win over the environment and are meant
        for CLI flags. ``None`` overrides are ignored so callers can pass
        optional flags unconditionally.
        """
        g = os.environ.get
        data: dict = dict(
            lsl_name=g("PIEEG_LSL_NAME", "PiEEG"),
            lsl_type=g("PIEEG_LSL_TYPE", "EEG"),
            lsl_resolve_by=g("PIEEG_LSL_RESOLVE_BY", "type"),
            lsl_resolve_timeout=_as_float(g("PIEEG_LSL_RESOLVE_TIMEOUT"), 5.0),
            ring_seconds=_as_float(g("PIEEG_RING_SECONDS"), 60.0),
            provider=g("PIEEG_LLM_PROVIDER", DEFAULT_PROVIDER),
            model=g("PIEEG_LLM_MODEL", ""),
            fallback_provider=g("PIEEG_LLM_FALLBACK_PROVIDER", ""),
            fallback_model=g("PIEEG_LLM_FALLBACK_MODEL", ""),
            ws_url=g("PIEEG_WS_URL", "ws://localhost:1616"),
            dashboard_url=g("PIEEG_DASHBOARD_URL", "http://localhost:1617"),
        )
        data.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**data)
        cfg._resolve_provider_defaults()
        return cfg

    def _resolve_provider_defaults(self) -> None:
        """Fill blank model / base_url / api_key from the registry + env.
        
        Also automatically configures fallback to a smaller model from the same
        provider if no explicit fallback is configured.
        """
        # Primary provider
        spec = PROVIDERS.get(self.provider)
        if not spec:
            return
        if not self.model:
            self.model = spec["default_model"]
        if not self.base_url:
            self.base_url = spec["base_url"]
        if not self.api_key and spec.get("env_key"):
            self.api_key = os.environ.get(spec["env_key"], "")
        
        # Fallback provider (if explicitly configured)
        if self.fallback_provider:
            fallback_spec = PROVIDERS.get(self.fallback_provider)
            if fallback_spec:
                if not self.fallback_model:
                    self.fallback_model = fallback_spec["default_model"]
                if not self.fallback_base_url:
                    self.fallback_base_url = fallback_spec["base_url"]
                if not self.fallback_api_key and fallback_spec.get("env_key"):
                    self.fallback_api_key = os.environ.get(fallback_spec["env_key"], "")
        
        # Automatic fallback: use same provider with smaller model if available
        elif self.model in AUTO_FALLBACK_MODELS:
            self.fallback_provider = self.provider
            self.fallback_model = AUTO_FALLBACK_MODELS[self.model]
            self.fallback_base_url = self.base_url
            self.fallback_api_key = self.api_key

    # ── Introspection helpers ──────────────────────────────────────────
    @property
    def provider_spec(self) -> dict:
        return PROVIDERS.get(self.provider, {})

    @property
    def provider_kind(self) -> str:
        return self.provider_spec.get("kind", "")

    @property
    def needs_api_key(self) -> bool:
        return bool(self.provider_spec.get("env_key"))

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty == OK).

        Ingestion-only runs don't need an LLM, so the CLI decides whether
        these warnings are fatal for the chosen command.
        """
        problems: list[str] = []
        if self.provider not in PROVIDERS:
            known = ", ".join(sorted(PROVIDERS))
            problems.append(
                f"Unknown LLM provider {self.provider!r}. Known: {known}."
            )
            return problems
        if self.lsl_resolve_by not in ("name", "type"):
            problems.append(
                f"lsl_resolve_by must be 'name' or 'type', got {self.lsl_resolve_by!r}."
            )
        if self.needs_api_key and not self.has_api_key:
            env_key = self.provider_spec.get("env_key")
            problems.append(
                f"Provider {self.provider!r} needs an API key — set ${env_key}."
            )
        return problems


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
