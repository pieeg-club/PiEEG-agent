"""The copilot tool-use loop, driven by a scripted fake provider.

These pin the normalized cycle without any network: a fake provider returns a
queued sequence of :class:`LLMResponse`s (first a tool request, then a final
answer), and we assert the copilot executes the tool, feeds the result back,
preserves history, and bounds runaway tool loops.
"""

from pieeg_agent.agent.copilot import Copilot
from pieeg_agent.agent.tools import NeuralTools
from pieeg_agent.llm.provider import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)
from tests.test_tools import FakeCascade, mk_state


class ScriptedProvider(LLMProvider):
    """Returns queued responses; records each ``complete`` call it sees."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # (messages, tools) per invocation

    def complete(self, *, system, messages, tools=None, max_tokens=1024,
                 temperature=0.0):
        self.calls.append((list(messages), tools))
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(text="(out of script)")


def test_single_turn_text_answer():
    provider = ScriptedProvider([LLMResponse(text="You look calm.")])
    copilot = Copilot(provider, NeuralTools(FakeCascade(state=mk_state())))
    result = copilot.ask("how am I?")
    assert result.text == "You look calm."
    assert result.tool_calls == []
    assert result.iterations == 1
    # History holds the user turn and the assistant answer.
    assert copilot.history[0].role == "user"
    assert copilot.history[-1].role == "assistant"


def test_tool_call_then_answer():
    provider = ScriptedProvider([
        LLMResponse(
            tool_calls=[ToolCall(id="t1", name="get_neural_state", arguments={})],
            usage=Usage(10, 2),
        ),
        LLMResponse(text="Your focus is 0.42.", usage=Usage(8, 4)),
    ])
    copilot = Copilot(provider, NeuralTools(FakeCascade(state=mk_state())))
    result = copilot.ask("what is my focus?")

    assert result.text == "Your focus is 0.42."
    assert result.tool_calls == ["get_neural_state"]
    assert result.iterations == 2
    # Usage accumulates across both turns.
    assert result.usage.input_tokens == 18
    assert result.usage.output_tokens == 6

    # A tool-result message was inserted between the two assistant turns.
    roles = [m.role for m in copilot.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = copilot.history[2]
    assert tool_msg.tool_call_id == "t1"
    assert "focus" in tool_msg.content

    # The second provider call saw the tool result in the conversation.
    second_messages = provider.calls[1][0]
    assert any(m.role == "tool" for m in second_messages)


def test_unknown_tool_is_fed_back_as_error():
    provider = ScriptedProvider([
        LLMResponse(tool_calls=[ToolCall(id="t1", name="bogus", arguments={})]),
        LLMResponse(text="Sorry, I could not check that."),
    ])
    copilot = Copilot(provider, NeuralTools(FakeCascade(state=mk_state())))
    result = copilot.ask("do the bogus thing")
    assert result.text == "Sorry, I could not check that."
    # The error is delivered to the model as a tool result, not raised.
    tool_msg = copilot.history[2]
    assert "error" in tool_msg.content


def test_tool_iteration_budget_is_bounded():
    # A provider that always asks for a tool must not loop forever.
    always_tool = LLMResponse(
        tool_calls=[ToolCall(id="t", name="get_neural_state", arguments={})]
    )
    provider = ScriptedProvider([always_tool] * 50)
    copilot = Copilot(
        provider,
        NeuralTools(FakeCascade(state=mk_state())),
        max_tool_iters=3,
    )
    result = copilot.ask("loop please")
    # 3 looped completes + 1 final no-tools attempt = 4 calls.
    assert len(provider.calls) == 4
    assert result.iterations == 3
    # Final attempt is made without tools.
    assert provider.calls[-1][1] is None


def test_reset_clears_history():
    provider = ScriptedProvider([
        LLMResponse(text="one"),
        LLMResponse(text="two"),
    ])
    copilot = Copilot(provider, NeuralTools(FakeCascade(state=mk_state())))
    copilot.ask("first")
    copilot.reset()
    assert copilot.history == []
    copilot.ask("second")
    # Only the second exchange remains.
    assert [m.content for m in copilot.history if m.role == "user"] == ["second"]
