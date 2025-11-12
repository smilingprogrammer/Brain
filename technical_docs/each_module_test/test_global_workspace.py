# test_global_workspace.py
import asyncio
from brain_regions.integration.global_workspace import GlobalWorkspace
from core.event_bus import EventBus


async def test_global_workspace():
    # Setup
    bus = EventBus()
    gw = GlobalWorkspace(bus)
    await gw.initialize()

    print("=== Global Workspace Test ===\n")

    # Simulate brain region outputs competing for attention
    print("1. Simulating attention competition:")

    # Add competing information
    await gw._add_to_competition("language", {
        "text": "Important linguistic information",
        "confidence": 0.8,
        "urgency": 0.6
    })
    print("Added language output (confidence: 0.8)")

    await gw._add_to_competition("working_memory", {
        "content": "Critical memory update",
        "confidence": 0.9,
        "salience": 0.85
    })
    print("Added working memory (confidence: 0.9)")

    await gw._add_to_competition("reasoning", {
        "conclusion": "Logical deduction complete",
        "confidence": 0.95,
        "importance": "high"
    })
    print("Added reasoning output (confidence: 0.95)")

    await gw._add_to_competition("emotional", {
        "state": "mild concern",
        "confidence": 0.6,
        "emotional_valence": -0.3
    })
    print("Added emotional state (confidence: 0.6)")

    # Run competition
    print("\n2. Running attention competition:")
    winners = await gw._run_attention_competition()

    print(f"\nWinners ({len(winners)} regions):")
    for region, info in winners.items():
        print(f"  - {region}: salience={info['salience']:.2f}")

    # Test integration
    print("\n3. Testing integration:")
    integrated = await gw._integrate_representations(winners)

    print(f"\nIntegrated State:")
    print(f"  Primary focus: {integrated['primary_focus']['region'] if integrated['primary_focus'] else 'None'}")
    print(f"  Secondary elements: {len(integrated['secondary_elements'])}")
    print(f"  Integrated meaning: {integrated['integrated_meaning']}")
    print(f"  Overall confidence: {integrated['confidence']:.2f}")

    # Check state
    state = gw.get_state()
    print(f"\nGlobal Workspace State:")
    print(f"  Competing regions: {state['competing_regions']}")
    print(f"  Competition size: {state['competition_size']}")
    print(f"  History length: {state['history_length']}")


asyncio.run(test_global_workspace())