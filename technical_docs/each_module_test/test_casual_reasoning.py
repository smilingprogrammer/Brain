# test_causal_reasoning.py
import asyncio
from brain_regions.reasoning.causal_reasoning import CausalReasoning
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_causal_reasoning():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    causal = CausalReasoning(bus, gemini)
    await causal.initialize()

    print("=== Causal Reasoning Test ===\n")

    # Test scenarios
    test_scenarios = [
        {
            "scenario": "If ocean temperature rises by 2 degrees",
            "query": "What happens to marine ecosystems?"
        },
        {
            "scenario": "A new technology makes batteries 10x cheaper",
            "query": "How does this affect electric vehicle adoption?"
        },
        {
            "scenario": "A major volcano erupts",
            "query": "What are the climate effects?"
        }
    ]

    for test in test_scenarios:
        print(f"\nScenario: {test['scenario']}")
        print(f"Query: {test['query']}")
        print("-" * 50)

        result = await causal.process({
            "scenario": test['scenario'],
            "query": test['query']
        })

        if result['success']:
            # Show causal graph
            graph = result.get('causal_graph', {})
            print(f"\nCausal Graph:")
            print(f"  Nodes: {len(graph.get('nodes', []))}")
            print(f"  Edges: {len(graph.get('edges', []))}")

            # Show causal chains
            chains = result.get('causal_chains', [])
            print(f"\nCausal Chains: {len(chains)} found")
            for i, chain in enumerate(chains[:3]):
                print(f"\n{i + 1}. Chain (strength: {chain.get('strength', 0):.2f}):")
                print(f"   Path: {' → '.join(chain.get('path', []))}")
                print(f"   Type: {chain.get('type', 'N/A')}")

            # Show interventions
            interventions = result.get('interventions', [])
            if interventions:
                print(f"\nPossible Interventions:")
                for i, intervention in enumerate(interventions[:3]):
                    print(f"{i + 1}. Target: {intervention.get('target', '')}")
                    print(f"   Type: {intervention.get('type', '')}")
                    print(f"   Impact: {intervention.get('impact', {})}")

            # Show predictions
            predictions = result.get('predictions', [])
            if predictions:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:3]):
                    print(f"{i + 1}. {pred.get('outcome', '')}")
                    print(f"   Timeframe: {pred.get('timeframe', '')}")
                    print(f"   Probability: {pred.get('probability', 0):.2f}")


asyncio.run(test_causal_reasoning())