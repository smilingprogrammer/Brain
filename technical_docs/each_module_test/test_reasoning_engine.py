# test_reasoning_engine.py
import asyncio
from brain_regions.gemini.reasoning_engine import ReasoningEngine
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_reasoning_engine():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    engine = ReasoningEngine(gemini, bus)
    await engine.initialize()

    print("=== Reasoning Engine Test ===\n")

    # Test 1: Single strategy reasoning
    print("1. Testing individual reasoning strategies:")

    problem = "A farmer needs to cross a river with a fox, a chicken, and grain. The boat can only carry the farmer and one item. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How can the farmer get everything across safely?"

    strategies = ["deductive", "causal", "counterfactual"]

    for strategy in strategies:
        print(f"\n{strategy.upper()} reasoning:")
        result = await engine.reason(problem, {}, strategies=[strategy])

        if result['success']:
            path = result['reasoning_paths'][0]
            print(f"Approach: {path['result']['approach']}")
            print(f"Confidence: {path['result']['confidence']:.2f}")
            print(f"Reasoning preview: {path['result']['reasoning'][:150]}...")

    # Test 2: Multi-path reasoning
    print("\n2. Testing multi-path reasoning:")

    complex_problem = "If artificial general intelligence is achieved, what would be the most important considerations for humanity?"

    result = await engine.reason(complex_problem, {})

    if result['success']:
        print(f"\nPaths explored: {len(result['reasoning_paths'])}")

        for path in result['reasoning_paths']:
            print(f"\n- {path['strategy']}: confidence {path['result']['confidence']:.2f}")

        synthesis = result['synthesis']
        print(f"\nSynthesis:")
        print(f"  Conclusion: {synthesis['conclusion'][:200]}...")
        print(f"  Overall confidence: {synthesis['confidence']:.2f}")

        if synthesis['agreements']:
            print(f"  Points of agreement: {synthesis['agreements'][:2]}")

        if synthesis['insights']:
            print(f"  Key insights: {synthesis['insights'][:2]}")


asyncio.run(test_reasoning_engine())