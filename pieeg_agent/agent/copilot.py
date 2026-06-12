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
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator, Literal

from ..llm.provider import LLMProvider, LLMResponse, Message, Usage
from .context import ContextManager, should_compress_tool_result, compress_session_payload
from .tools import Toolset

logger = logging.getLogger("pieeg.agent.copilot")

SYSTEM_PROMPT = """\
You are PiEEG Copilot, a concise assistant embedded in a live brain-computer \
interface. A person is wearing an EEG headset connected to PiEEG-server, and \
its signal is reduced for you into language-sized facts you read through tools.

Your senses (all read-only tools, always ground claims in a fresh call):
- get_neural_state / get_band_powers — the smoothed ~1 Hz state and spectral \
shape. focus / relax / engagement are convenience indices, not the whole story.
- analyze_spectrum — deeper spectral detail: individual alpha-peak frequency, \
aperiodic 1/f slope, theta/beta ratio, entropy, frontal alpha asymmetry.
- find_artifacts — discrete events: blinks (and double-blinks), jaw clenches, \
motion. Use these for "did I just blink/clench" and to explain quality drops.
- get_recent_events / get_channel_quality — transitions and per-channel signal \
quality.
- connectivity — cross-channel amplitude coupling right now (how the electrodes' \
band-power envelopes move together). Use this for "which channels are coupled" \
or "how connected is the alpha band".

Trainable patterns are your most powerful sense — prefer them when the user \
asks about a specific learned state:
- list_patterns / detect_patterns — what you can recognise, and which patterns \
are firing right now. explain_pattern justifies one (its cross-validated score, \
which channels it reads). A trained detector is far more meaningful than the \
generic focus/relax indices.
- To TEACH a new pattern, run it as a TURN-BASED conversation — NEVER record \
several takes in one message. Recording samples the LIVE signal the instant you \
call record_segment, so the user must already be settled into the right state, \
and you must hand control back between every take:
  1. start_pattern_training(name). Then tell the user to relax for the baseline \
and ask them to say when they're ready. END YOUR TURN — do NOT record yet.
  2. When the user confirms, call record_segment(label="rest") ONCE. Then tell \
them what to do for the active state and ask them to say when ready. END YOUR \
TURN — only one recording per message.
  3. When they confirm, call record_segment(label="active") ONCE — that is one \
rep. Then ask them to relax again and wait for their go-ahead.
  4. Alternate rest/active like this for 3-5 reps total, ALWAYS one recording \
per turn and ALWAYS waiting for the user to say they're ready first.
  5. After each record_segment, read the "action" field in the result. When it \
says you have enough, call finish_pattern_training. The system caps training at \
8 reps and allows only one recording per turn, so you can never get stuck in a \
loop — if you try to record twice in one turn the second is refused.
  6. finish_pattern_training fits, cross-validates and saves the detector. \
Report the balanced accuracy honestly: under ~0.7 is weak, 0.7-0.85 decent, \
over 0.85 strong.

The lab notebook (sessions) is for labelled windows you record and compare:
- record_session(label, seconds) captures a window (e.g. "eyes-closed-rest" for \
20s) and saves a summary: band means/spread, focus/relax/engagement, signal \
quality, artifacts, connectivity. Tell the user what to do *before* calling.
- list_sessions / analyze_session — what windows you've recorded and re-open \
any by label.
- compare_sessions(a, b) contrasts two session summaries with Cohen's d per \
feature (band powers, focus/relax/engagement, quality), ranked by effect size. \
Use this for "what changed between my rest and my focus block".
- forget_session(label) deletes a saved recording.

When the user asks about PiEEG hardware, server setup, electrode placement, or \
troubleshooting (not about your own capabilities):
- search_docs — search PiEEG ecosystem documentation for hardware specs, server \
configuration, LSL streaming setup, signal quality issues, etc. Use this for \
questions like "how do I place electrodes?", "what's the sampling rate?", "LSL \
not working", "signal is noisy". Ground setup/troubleshooting answers in these \
docs rather than guessing.

When the user asks about BCI concepts, neuroscience, or research (general \
knowledge beyond PiEEG-specific setup):
- search_web — search Wikipedia or PubMed for BCI/neuroscience topics. Use this \
for questions like "what is the P300?", "alpha waves and meditation", "motor \
imagery BCI", or "EEG research papers on focus". Returns titles, snippets and \
URLs. Choose source="wikipedia" for general concepts or source="pubmed" for \
research papers.
- fetch_url — fetch content from trusted scientific/BCI sources (Wikipedia, \
PubMed, arxiv, Nature, PLoS, Frontiers, etc.) or PiEEG websites. Domain must be \
on the allowlist — use list_trusted_sources to see what's available.
- list_trusted_sources — show which domains are allowed for fetch_url (Wikipedia, \
scientific publishers, PiEEG sites). Filter by category: "wikipedia", "scientific", \
"pieeg", or "all".

General utility tools for workflow automation and introspection:
- list_tools — discover all currently available tools across all toolsets. \
Use this to answer "what can you do?" or when you need to check what \
capabilities are available. By default returns concise output; set verbose=true \
for full parameter schemas. Filter by toolset (neural, decode, documentation, \
actuator, utility) to focus on specific capabilities.
- read_file / write_file — read or write text files (code, logs, configs, CSVs). \
Use for loading data, saving analysis results, or reading user-provided files.
- read_image — read image files and view them as base64 data URIs (PNG, JPG, \
GIF, BMP, WebP). Useful for inspecting plots, spectrograms, or visualizations.
- list_directory — list contents of a directory with file sizes and modification \
times. Supports recursive listing.
- create_notebook / run_notebook / read_notebook — create and execute Jupyter \
notebooks for data analysis. create_notebook makes a new .ipynb with specified \
cells (code/markdown) and automatically adds a header with EEG session metadata \
(stream name, channels, sampling rate, date). run_notebook executes it and \
returns outputs, read_notebook reads structure without executing. Use these for \
reproducible analysis workflows.

Be honest about the metrics:
- focus / relax / engagement are 0..1 values **relative to this session's own \
range**, not absolute or clinical measures. Describe them as "high/low for you \
right now".
- If a state reports warming_up=true, say the readings are still settling.
- If signal quality is poor or channels are flagged, say so before drawing \
conclusions; bad electrodes make the indices meaningless.
- Cohen's d from compare_sessions is within-session descriptive only, not a \
generalisation or clinical claim.

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


@dataclass
class CopilotEvent:
    """One incremental event from :meth:`Copilot.ask_stream`.

    A single ``ask_stream`` call emits a flat sequence:

    * ``token`` — ``text`` is the next chunk of the assistant's prose.
    * ``tool_start`` — the model asked to run ``name``; ``arguments`` is the
      parsed input. Emitted just before the tool executes so a UI can show it.
    * ``tool_result`` — that tool finished; ``result`` is its return payload.
    * ``done`` — terminal event carrying the full ``text``, the ordered
      ``tool_calls`` that ran, accumulated ``usage`` and ``iterations``.
    """

    type: Literal["token", "tool_start", "tool_result", "done"]
    text: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    result: Any = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[str] = field(default_factory=list)
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
        max_tool_iters: int = 10,
        context_manager: ContextManager | None = None,
        min_request_interval: float = 0.5,
    ):
        self._provider = provider
        self._tools = tools
        self._system = system
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_tool_iters = max_tool_iters
        self._history: list[Message] = []
        self._context_manager = context_manager or ContextManager()
        self._min_request_interval = min_request_interval
        self._last_request_time: float = 0.0

    # ── conversation surface ─────────────────────────────────────────────
    def reset(self) -> None:
        """Forget the conversation so far (tools/provider are kept)."""
        self._history.clear()
        self._context_manager.reset()

    @property
    def history(self) -> list[Message]:
        return self._history

    def ask(self, question: str) -> CopilotResult:
        """Answer ``question``, running tool calls as the model requests them.

        Conversation history is preserved across calls so follow-ups have
        context. The tool-use loop is bounded by ``max_tool_iters``. This is a
        thin blocking wrapper over :meth:`ask_stream`, so both the CLI and the
        streaming web surface drive the exact same loop.
        """
        done = CopilotEvent(type="done")
        for event in self.ask_stream(question):
            if event.type == "done":
                done = event
        return CopilotResult(
            text=done.text,
            tool_calls=done.tool_calls,
            usage=done.usage,
            iterations=done.iterations,
        )

    def ask_stream(self, question: str) -> Iterator[CopilotEvent]:
        """Answer ``question`` incrementally, yielding :class:`CopilotEvent`.

        Same bounded tool-use loop as :meth:`ask`, but assistant prose streams
        out as ``token`` events and each tool call surfaces as a
        ``tool_start`` / ``tool_result`` pair. The final ``done`` event mirrors
        what :meth:`ask` would have returned.
        """
        self._history.append(Message(role="user", content=question))
        
        # Check if compression is needed before starting the tool loop
        if self._context_manager.should_compress(self._history):
            compressed, stats = self._context_manager.compress(self._history)
            self._history = compressed
            logger.info(
                "Compressed conversation history: %d → %d tokens (%.1f%% reduction)",
                stats.original_tokens,
                stats.compressed_tokens,
                (1 - stats.compression_ratio) * 100,
            )

        total = Usage()
        used_tools: list[str] = []
        # Pattern training is interactive: recording samples the *live* signal
        # the instant it is called, so the user must physically settle into each
        # state between takes. Cap recordings at one per user turn so the model
        # cannot chain takes (which freezes the UI for seconds and captures the
        # wrong mental state) — a second take in the same turn is short-circuited
        # with guidance to hand control back to the user.
        records_this_turn = 0

        for iteration in range(1, self._max_tool_iters + 1):
            resp = yield from self._stream_turn(self._tools.specs())
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
                yield CopilotEvent(
                    type="done",
                    text=resp.text,
                    tool_calls=used_tools,
                    usage=total,
                    iterations=iteration,
                )
                return

            # Execute each requested tool and feed results back.
            for call in resp.tool_calls:
                used_tools.append(call.name)

                # Enforce the one-recording-per-turn rule for guided training.
                if call.name == "record_segment" and records_this_turn >= 1:
                    guidance = {
                        "status": "wait_for_user",
                        "action": (
                            "You already recorded one segment this turn. STOP — "
                            "do NOT record again now. Tell the user exactly what "
                            "to do for the next segment (relax for 'rest', or "
                            "perform the pattern for 'active'), ask them to say "
                            "when they are ready, then END YOUR TURN and wait for "
                            "their reply before the next record_segment."
                        ),
                    }
                    yield CopilotEvent(
                        type="tool_start", name=call.name, arguments=call.arguments
                    )
                    yield CopilotEvent(
                        type="tool_result", name=call.name, result=guidance
                    )
                    self._history.append(
                        Message(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(guidance),
                        )
                    )
                    continue
                if call.name == "record_segment":
                    records_this_turn += 1

                yield CopilotEvent(
                    type="tool_start", name=call.name, arguments=call.arguments
                )
                result = self._tools.call(call.name, call.arguments)
                logger.debug("tool %s(%s) -> %s", call.name, call.arguments, result)

                # Surface guided-training guidance to the logs for debugging.
                if call.name == "record_segment" and isinstance(result, dict) \
                        and "action" in result:
                    logger.info("Training guidance: %s", result["action"])
                
                # Compress large payloads before storing in history
                result_for_history = result
                if should_compress_tool_result(call.name, result):
                    result_for_history = compress_session_payload(result)
                    logger.debug(
                        "Compressed %s result: %d → %d chars",
                        call.name,
                        len(json.dumps(result)),
                        len(json.dumps(result_for_history)),
                    )

                yield CopilotEvent(type="tool_result", name=call.name, result=result)
                self._history.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(result_for_history),
                    )
                )

        # Tool budget exhausted — make one final answer attempt without tools
        # so the user still gets a reply instead of silence.
        final = yield from self._stream_turn(None)
        total = total + final.usage
        text = final.text or "(stopped after the tool-call limit)"
        self._history.append(Message(role="assistant", content=final.text))
        yield CopilotEvent(
            type="done",
            text=text,
            tool_calls=used_tools,
            usage=total,
            iterations=self._max_tool_iters,
        )

    def _stream_turn(
        self, tools
    ) -> "Generator[CopilotEvent, None, LLMResponse]":
        """Stream one provider turn, yielding ``token`` events.

        Returns (via ``StopIteration.value``, i.e. ``yield from``) the final
        assembled :class:`LLMResponse` for the caller to act on.
        """
        # Rate limiting: ensure minimum interval between API requests
        if self._min_request_interval > 0:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                sleep_time = self._min_request_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self._last_request_time = time.time()
        
        response = LLMResponse()
        for event in self._provider.stream_complete(
            system=self._system,
            messages=self._history,
            tools=tools,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        ):
            if event.type == "text" and event.text:
                yield CopilotEvent(type="token", text=event.text)
            elif event.type == "final" and event.response is not None:
                response = event.response
        return response
