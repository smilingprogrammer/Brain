# test_attention_controller.py
import asyncio
from brain_regions.executive.attention import AttentionController
from core.event_bus import EventBus


async def test_attention_controller():
    # Setup
    bus = EventBus()
    attention = AttentionController(bus)
    await attention.initialize()

    print("=== Attention Controller Test ===\n")

    # Test 1: Distributed attention
    print("1. Testing distributed attention:")

    result = await attention.process({
        "command": "distribute",
        "targets": ["language", "memory", "reasoning"],
        "weights": [0.4, 0.3, 0.3]
    })

    print(f"Distribution result: {result['distribution']}")

    # Test 2: Focused attention
    print("\n2. Testing focused attention:")

    result = await attention.process({
        "command": "focus",
        "target": "reasoning",
        "intensity": 0.8
    })

    print(f"Focus result:")
    print(f"  Target: {result['focus']}")
    print(f"  Intensity: {result['intensity']}")

    # Test 3: Attention requests
    print("\n3. Testing attention requests:")

    # Simulate high-priority request
    await bus.emit("request_attention", {
        "requester": "working_memory",
        "priority": 0.9,
        "duration": 2.0
    })

    await asyncio.sleep(0.5)  # Let it process

    # Check allocation
    allocation = await attention.process({"command": "get_allocation"})
    print(f"\nCurrent allocation:")
    for region, weight in allocation['allocation'].items():
        print(f"  {region}: {weight:.2f}")

    print(f"\nAttention mode: {allocation['mode']}")
    print(f"Total allocated: {allocation['total_allocated']:.2f}")

    # Test 4: Wait for duration to expire
    print("\n4. Waiting for attention release...")
    await asyncio.sleep(2.5)

    allocation = await attention.process({"command": "get_allocation"})
    print(f"After release: {allocation['allocation']}")


asyncio.run(test_attention_controller())