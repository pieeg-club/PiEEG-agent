"""Test for rate limiting during rapid API calls."""

import time
from unittest.mock import Mock, patch

import pytest

from pieeg_agent.agent.copilot import Copilot
from pieeg_agent.agent.tools import Toolset
from pieeg_agent.llm.provider import LLMProvider, LLMResponse, StreamEvent


def test_rate_limiting_enforced():
    """Verify that rate limiting delays rapid API calls."""
    
    # Create a mock provider that tracks call times
    call_times = []
    
    class MockProvider(LLMProvider):
        name = "mock"
        
        def complete(self, **kwargs):
            call_times.append(time.time())
            return LLMResponse(text="ok")
        
        def stream_complete(self, **kwargs):
            call_times.append(time.time())
            yield StreamEvent(type="text", text="ok")
            yield StreamEvent(type="final", response=LLMResponse(text="ok"))
    
    # Create copilot with 0.5s minimum interval
    provider = MockProvider()
    tools = Mock(spec=Toolset)
    tools.specs.return_value = []
    
    copilot = Copilot(
        provider=provider,
        tools=tools,
        min_request_interval=0.5,
    )
    
    # Make 3 rapid calls
    for i in range(3):
        copilot.ask(f"Question {i}")
    
    # Verify calls were rate-limited
    assert len(call_times) == 3
    
    # Check intervals between calls
    interval_1 = call_times[1] - call_times[0]
    interval_2 = call_times[2] - call_times[1]
    
    # Each interval should be at least 0.5s (allowing small timing variance)
    assert interval_1 >= 0.45, f"First interval too short: {interval_1:.3f}s"
    assert interval_2 >= 0.45, f"Second interval too short: {interval_2:.3f}s"


def test_rate_limiting_disabled():
    """Verify that rate limiting can be disabled with min_request_interval=0."""
    
    call_times = []
    
    class MockProvider(LLMProvider):
        name = "mock"
        
        def complete(self, **kwargs):
            call_times.append(time.time())
            return LLMResponse(text="ok")
        
        def stream_complete(self, **kwargs):
            call_times.append(time.time())
            yield StreamEvent(type="text", text="ok")
            yield StreamEvent(type="final", response=LLMResponse(text="ok"))
    
    provider = MockProvider()
    tools = Mock(spec=Toolset)
    tools.specs.return_value = []
    
    # Disable rate limiting
    copilot = Copilot(
        provider=provider,
        tools=tools,
        min_request_interval=0.0,
    )
    
    # Make 3 rapid calls
    start = time.time()
    for i in range(3):
        copilot.ask(f"Question {i}")
    elapsed = time.time() - start
    
    # Should complete quickly without delays
    assert elapsed < 0.3, f"Calls were delayed when rate limiting was disabled: {elapsed:.3f}s"
