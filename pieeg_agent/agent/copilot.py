"""The copilot — a conversational brain companion over the live cascade.

This is the reasoning loop: it pairs an :class:`~pieeg_agent.llm.provider.
LLMProvider` with the read-only :class:`~pieeg_agent.agent.tools.NeuralTools`
and runs the normalized tool-use cycle so the model can *ask* about the brain
state in natural language and answer the user.

The cycle (see :mod:`pieeg_agent.llm.provider`):

  1. append the user's turn,
  2. call ``provider.complete`` with the conversation + tool specs,
  3. if the model requested tools, run each read-only tool and append a
     ``tool`` result turn, then loop,
  4. stop when the model returns text and no tool calls.

The copilot is intentionally *read-only* in this phase: the only tools wired
in are senses, never actuators. It keeps conversation history so a ``chat``
session is multi-turn, and caps tool iterations so a misbehaving model can't
spin forever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..llm.provider import LLMProvider, Message, Usage
from .tools import Toolset

logger = logging.getLogger("pieeg.agent.copilot")

SYSTEM_PROMPT = """\
You are PiEEG Copilot, a concise assistant embedded in a live brain-computer \
interface. A person is wearing an EEG headset connected to PiEEG-server, and \
its signal is reduced for you into language-sized facts you read through tools.

You can call read-only tools to inspect the current neural state, band powers, \
recent events and per-channel signal quality. Always ground claims about the \
brain in a tool call from this session — never invent numbers.

Be honest about the metrics:
- focus / relax / engagement are 0..1 values **relative to this session's own \
range**, not absolute or clinical measures. Describe them as "high/low for you \
right now".
- If a state reports warming_up=true, say the readings are still settling.
- If signal quality is poor or channels are flagged, say so before drawing \
conclusions; bad electrodes make the indices meaningless.

You observe and explain; you do not control the device in this mode. Keep \
answers short and plain. If no data is available yet, say the stream is still \
warming up rather than guessing."""


ACTUATOR_SYSTEM_PROMPT = SYSTEM_PROMPT + """

In THIS session you also have a small set of control tools that can change the \
device (filter, recording, OSC output, register presets). Treat them with \
care:
- Only act when the user clearly asks you to, or when it is plainly needed to \
answer them. Prefer the least-invasive action, and read the current status \
before changing it.
- Every control tool is gated. A call may come back as "dry_run" (previewed, \
not sent), "denied" (not permitted or on cooldown) or "executed". Always read \
that outcome and tell the user plainly what actually happened — never claim an \
action took effect if it was only previewed or denied.
- Do not repeat or retry an action in a loop. One attempt, then report back."""


@dataclass
class CopilotResult:
    """The outcome of one :meth:`Copilot.ask` call."""

    text: str
    tool_calls: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0


class Copilot:
    """A multi-turn, tool-using conversational layer over the cascade."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: Toolset,
        *,
        system: str = SYSTEM_PROMPT,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_tool_iters: int = 6,
    ):
        self._provider = provider
        self._tools = tools
        self._system = system
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_tool_iters = max_tool_iters
        self._history: list[Message] = []

    # ── conversation surface ─────────────────────────────────────────────
    def reset(self) -> None:
        """Forget the conversation so far (tools/provider are kept)."""
        self._history.clear()

    @property
    def history(self) -> list[Message]:
        return self._history

    def ask(self, question: str) -> CopilotResult:
        """Answer ``question``, running tool calls as the model requests them.

        Conversation history is preserved across calls so follow-ups have
        context. The tool-use loop is bounded by ``max_tool_iters``.
        """
        self._history.append(Message(role="user", content=question))

        total = Usage()
        used_tools: list[str] = []
        for iteration in range(1, self._max_tool_iters + 1):
            resp = self._provider.complete(
                system=self._system,
                messages=self._history,
                tools=self._tools.specs(),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            total = total + resp.usage

            # Record the assistant turn (text and/or tool requests).
            self._history.append(
                Message(
                    role="assistant",
                    content=resp.text,
                    tool_calls=list(resp.tool_calls),
                )
            )

            if not resp.wants_tools:
                return CopilotResult(
                    text=resp.text,
                    tool_calls=used_tools,
                    usage=total,
                    iterations=iteration,
                )

            # Execute each requested tool and feed results back.
            for call in resp.tool_calls:
                used_tools.append(call.name)
                result = self._tools.call(call.name, call.arguments)
                logger.debug("tool %s(%s) -> %s", call.name, call.arguments, result)
                self._history.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(result),
                    )
                )

        # Tool budget exhausted — make one final answer attempt without tools
        # so the user still gets a reply instead of silence.
        final = self._provider.complete(
            system=self._system,
            messages=self._history,
            tools=None,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        total = total + final.usage
        self._history.append(Message(role="assistant", content=final.text))
        return CopilotResult(
            text=final.text or "(stopped after the tool-call limit)",
            tool_calls=used_tools,
            usage=total,
            iterations=self._max_tool_iters,
        )
