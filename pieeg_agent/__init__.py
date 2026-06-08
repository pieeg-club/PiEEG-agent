"""PiEEG-agent — a provider-agnostic LLM agent for live brain activity.

The agent perceives a high-rate EEG stream from PiEEG-server over Lab
Streaming Layer (LSL), reduces it into language-sized neural state and
events, reasons with a pluggable LLM provider (Anthropic by default), and
acts through PiEEG-server's control plane.

This package is built in phases; Phase 0 ships the ingestion spine:
  * ``pieeg_agent.ingest`` — LSL inlet thread + thread-safe ring buffer
  * ``pieeg_agent.llm``    — provider-agnostic LLM interface (contract only)
  * ``pieeg_agent.config`` — configuration + provider registry
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
