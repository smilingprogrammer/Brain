import pytest
import asyncio
from brain_regions.reasoning.logical_reasoning import LogicalReasoning
from brain_regions.reasoning.analogical_reasoning import AnalogicalReasoning
from brain_regions.reasoning.causal_reasoning import CausalReasoning
from brain_regions.reasoning.creative_reasoning import CreativeReasoning
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


@pytest.fixture
async def setup_reasoning():
    """Setup reasoning modules for testing"""
    event_bus = EventBus()
    gemini = GeminiService()

    modules = {
        "logical": LogicalReasoning(event_bus, gemini),
        "analogical": AnalogicalReasoning(event_bus, gemini),
        "causal": CausalReasoning(event_bus, gemini),
        "creative": CreativeReasoning(event_bus, gemini)
    }

    for module in modules.values():
        await module.initialize()

    return modules


@pytest.mark.asyncio
async def test_logical_reasoning(setup_reasoning):
    """Test logical reasoning module"""
    modules = await setup_reasoning
    logical = modules["logical"]

    # Test simple syllogism
    problem = "All humans are mortal. Socrates is human. Is Socrates mortal?"
    result = await logical.reason(problem, {})

    assert result["success"] is True
    assert result["confidence"] > 0.7
    assert "yes" in result["conclusion"].lower() or "mortal" in result["conclusion"].lower()


@pytest.mark.asyncio
async def test_analogical_reasoning(setup_reasoning):
    """Test analogical reasoning module"""
    modules = await setup_reasoning
    analogical = modules["analogical"]

    # Test analogy
    problem = "Find analogy between computer brain and human brain"
    result = await analogical.reason(problem, {})

    assert result["success"] is True
    assert len(result["mappings"]) > 0
    assert result["confidence"] > 0.5


@pytest.mark.asyncio
async def test_causal_reasoning(setup_reasoning):
    """Test causal reasoning module"""
    modules = await setup_reasoning
    causal = modules["causal"]

    # Test causal chain
    problem = "If temperature rises, ice melts. What happens to sea level?"
    result = await causal.reason(problem, {})

    assert result["success"] is True
    assert len(result["causal_chains"]) > 0
    assert result["confidence"] > 0.6


@pytest.mark.asyncio
async def test_creative_reasoning(setup_reasoning):
    """Test creative reasoning module"""
    modules = await setup_reasoning
    creative = modules["creative"]

    # Test creative problem
    problem = "Design a new type of umbrella"
    result = await creative.reason(problem, {})

    assert result["success"] is True
    assert len(result["solutions"]) > 0
    assert result["ideas_generated"] > 3


@pytest.mark.asyncio
async def test_reasoning_integration():
    """Test integration between reasoning modules"""
    event_bus = EventBus()
    gemini = GeminiService()

    # Create modules
    logical = LogicalReasoning(event_bus, gemini)
    causal = CausalReasoning(event_bus, gemini)

    await logical.initialize()
    await causal.initialize()

    # Test complex problem requiring both
    problem = "If all birds can fly and penguins are birds, what happens when a penguin jumps off a cliff?"

    # First logical reasoning
    logical_result = await logical.reason(problem, {})

    # Then causal reasoning with logical context
    causal_result = await causal.reason(problem, {"logical_analysis": logical_result})

    assert logical_result["success"] is True
    assert causal_result["success"] is True

    # Should identify the logical contradiction
    assert logical_result["confidence"] < 0.9  # Lower confidence due to contradiction