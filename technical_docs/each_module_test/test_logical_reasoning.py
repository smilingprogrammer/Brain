# test_logical_reasoning.py
import asyncio
from brain_regions.reasoning.logical_reasoning import LogicalReasoning
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_logical_reasoning():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    logic = LogicalReasoning(bus, gemini)
    await logic.initialize()

    print("=== Logical Reasoning Test ===\n")

    # Test cases
    test_problems = [
        "If all cats are animals and all animals need food, do cats need food?",
        "John is older than Mary. Mary is older than Tom. Is John older than Tom?",
        "If it rains, the ground gets wet. The ground is not wet. Did it rain?",
        "All birds have wings. Penguins are birds. Do penguins have wings?"
    ]

    for problem in test_problems:
        print(f"\nProblem: {problem}")
        print("-" * 50)

        result = await logic.reason(problem, {})

        if result['success']:
            print(f"Conclusion: {result.get('conclusion', 'N/A')}")
            print(f"Proof type: {result.get('proof_type', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")

            # Show first few proof steps
            if result.get('proof_steps'):
                print("Proof steps:")
                for i, step in enumerate(result['proof_steps'][:3]):
                    print(f"  {i + 1}. {step.get('statement', '')}")

            # Show validation
            if result.get('validation'):
                print(f"Valid: {result['validation'].get('valid', False)}")
                if result['validation'].get('issues'):
                    print(f"Issues: {result['validation']['issues'][:2]}")
        else:
            print("Reasoning failed!")


asyncio.run(test_logical_reasoning())