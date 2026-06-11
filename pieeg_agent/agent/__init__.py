"""The agent core — tools and the conversational copilot.

This package sits above perception and the LLM layer:

* :mod:`pieeg_agent.agent.tools` — read-only :class:`NeuralTools` that pull
  language-sized facts from the perception cascade.
* :mod:`pieeg_agent.agent.copilot` — :class:`Copilot`, the tool-using Q&A loop
  pairing a provider with those tools.
* :mod:`pieeg_agent.agent.actuator_tools` — opt-in :class:`ActuatorTools` that
  *act* on the device through the gated server-actions facade.

The read-only senses and the gated actuators live in separate modules so the
boundary between observing the brain and controlling the device stays obvious;
:class:`CombinedToolset` merges them when a session is given hands.
"""

from __future__ import annotations

from .actuator_tools import SAFE_ACTIONS, ActuatorTools
from .copilot import ACTUATOR_SYSTEM_PROMPT, SYSTEM_PROMPT, Copilot, CopilotResult
from .decode_tools import DecodeTools
from .doc_tools import DocumentationTools
from .tools import CombinedToolset, NeuralTools, Tool, Toolset
from .utility_tools import UtilityTools

__all__ = [
    "NeuralTools",
    "DecodeTools",
    "DocumentationTools",
    "UtilityTools",
    "Tool",
    "Toolset",
    "CombinedToolset",
    "ActuatorTools",
    "SAFE_ACTIONS",
    "Copilot",
    "CopilotResult",
    "SYSTEM_PROMPT",
    "ACTUATOR_SYSTEM_PROMPT",
]

