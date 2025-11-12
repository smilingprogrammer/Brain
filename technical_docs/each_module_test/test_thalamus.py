# test_thalamus.py
import asyncio
from brain_regions.integration.thalamus import Thalamus
from core.event_bus import EventBus


async def test_thalamus():
    # Setup
    bus = EventBus()
    thalamus = Thalamus(bus)
    await thalamus.initialize()

    print("=== Thalamus Test ===\n")

    # Test 1: Basic routing
    print("1. Testing information routing:")

    test_data = [
        {
            "source": "language_comprehension",
            "data": {"text": "Question detected", "complexity_score": 0.8},
            "priority": 0.7
        },
        {
            "source": "working_memory",
            "data": {"capacity_used": 0.9, "alert": "near_full"},
            "priority": 0.9
        },
        {
            "source": "reasoning",
            "data": {"conclusion": "Task complete", "confidence": 0.85},
            "priority": 0.6
        }
    ]

    for item in test_data:
        print(f"\nRouting from: {item['source']}")
        print(f"  Priority: {item['priority']}")

        result = await thalamus.process(item)

        if result['success']:
            print(f"  Routed to: {result['routed_to']}")
            print(f"  Gate value: {result['gate_value']:.2f}")
        else:
            print(f"  Blocked: {result['reason']}")

    # Test 2: Gate control
    print("\n2. Testing gate control:")

    # Close a gate
    await bus.emit("update_gate", {"region": "motor", "value": 0.0})
    print("Closed motor gate")

    # Try routing to closed gate
    result = await thalamus.process({
        "source": "executive",
        "data": {"action_required": True},
        "priority": 0.8
    })
    print(f"Routing result: {result}")

    # Reopen gate
    await bus.emit("update_gate", {"region": "motor", "value": 1.0})
    print("\nReopened motor gate")

    # Test 3: Check state
    state = thalamus.get_state()
    print(f"\n3. Thalamus State:")
    print(f"  Active routes: {state['active_routes']}")
    print(f"  Gate states: {dict(list(state['gate_states'].items())[:3])}")
    print(f"  Total routed: {state['total_routed']}")


asyncio.run(test_thalamus())