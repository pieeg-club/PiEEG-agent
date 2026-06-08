"""The agent core — tools and the conversational copilot.

This package sits above perception and the LLM layer:

* :mod:`pieeg_agent.agent.tools` — read-only :class:`NeuralTools` that pull
  language-sized facts from the perception cascade.
* :mod:`pieeg_agent.agent.copilot` — :class:`Copilot`, the tool-using Q&A loop
  pairing a provider with those tools.

Gated server *actions* (the actuator side) arrive in a later phase as their
own module, keeping the read-only boundary explicit.
"""

from __future__ import annotations

from .copilot import SYSTEM_PROMPT, Copilot, CopilotResult
from .tools import NeuralTools, Tool

__all__ = [
    "NeuralTools",
    "Tool",
    "Copilot",
    "CopilotResult",
    "SYSTEM_PROMPT",
]
