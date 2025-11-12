# test_analogical_reasoning.py
import asyncio
from brain_regions.reasoning.analogical_reasoning import AnalogicalReasoning
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_analogical_reasoning():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    analogical = AnalogicalReasoning(bus, gemini)
    await analogical.initialize()

    print("=== Analogical Reasoning Test ===\n")

    # Test cases
    test_analogies = [
        {
            "source": "computer brain",
            "target": "human brain",
            "problem": "How is a computer brain similar to a human brain?"
        },
        {
            "source": "atom",
            "target": "solar system",
            "problem": "What can we learn about atoms from the solar system model?"
        },
        {
            "source": "river network",
            "target": "blood vessels",
            "problem": "How do river networks relate to blood vessel systems?"
        }
    ]

    for test in test_analogies:
        print(f"\nAnalogy: {test['source']} → {test['target']}")
        print(f"Question: {test['problem']}")
        print("-" * 50)

        result = await analogical.process({
            "source": test['source'],
            "target": test['target']
        })

        if result['success']:
            print(f"\nMappings found: {len(result.get('mappings', []))}")

            # Show mappings
            for i, mapping in enumerate(result.get('mappings', [])[:3]):
                print(f"\n{i + 1}. {mapping.get('source_element', '')} → {mapping.get('target_element', '')}")
                print(f"   Relationship: {mapping.get('relationship', '')}")
                print(f"   Type: {mapping.get('mapping_type', '')}")
                print(f"   Strength: {mapping.get('strength', 0):.2f}")

            # Show insights
            print("\nInsights:")
            for i, insight in enumerate(result.get('insights', [])[:3]):
                print(f"{i + 1}. {insight}")

            # Show quality assessment
            quality = result.get('quality', {})
            print(f"\nAnalogy Quality:")
            print(f"  Overall score: {quality.get('score', 0):.2f}")
            print(f"  Strength: {quality.get('strength', 'N/A')}")
            print(f"  Usefulness: {quality.get('usefulness', 'N/A')}")

        asyncio.run(test_analogical_reasoning())