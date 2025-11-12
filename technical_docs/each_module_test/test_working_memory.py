# test_working_memory.py
import asyncio
from brain_regions.memory.working_memory import WorkingMemory
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_working_memory():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    wm = WorkingMemory(bus, gemini, capacity=5)
    await wm.initialize()

    print("=== Working Memory Test ===\n")

    # Test 1: Basic storage
    print("1. Testing basic storage:")
    items = [
        {"content": {"text": "The sky is blue"}, "type": "fact"},
        {"content": {"text": "Water boils at 100°C"}, "type": "fact"},
        {"content": {"text": "Gravity pulls objects down"}, "type": "fact"}
    ]

    for item in items:
        await wm.store(item)
        print(f"Stored: {item['content']['text']}")

    print(f"Buffer size: {len(wm.buffer)}/{wm.capacity}")

    # Test 2: Retrieval
    print("\n2. Testing retrieval:")
    query = "What color is the sky?"
    relevant = await wm.retrieve(query, k=2)
    print(f"Query: {query}")
    print(f"Retrieved {len(relevant)} items")
    for item in relevant:
        print(f"  - {item.get('content', {}).get('text', 'N/A')}")

    # Test 3: Overflow and compression
    print("\n3. Testing compression (adding more items):")
    more_items = [
        {"content": {"text": "Plants need sunlight"}, "type": "fact"},
        {"content": {"text": "Fish live in water"}, "type": "fact"},
        {"content": {"text": "Birds can fly"}, "type": "fact"}
    ]

    for item in more_items:
        await wm.store(item)
        print(f"Added: {item['content']['text']}")
        print(f"Buffer size: {len(wm.buffer)}/{wm.capacity}")

    # Check compression
    if wm.compressed_history:
        print(f"\nCompression occurred!")
        print(f"Compressed items: {wm.compressed_history[0]['item_count']}")
        print(f"Summary preview: {wm.compressed_history[0]['summary'][:100]}...")

    # Test 4: Get current state
    print("\n4. Current state:")
    state = wm.get_state()
    print(f"Buffer size: {state['buffer_size']}")
    print(f"Compressed count: {state['compressed_count']}")
    print(f"Capacity usage: {state['capacity_usage']:.1%}")


asyncio.run(test_working_memory())

# Expected Output
# === Working Memory Test ===
#
# 1. Testing basic storage:
# Stored: The sky is blue
# Stored: Water boils at 100°C
# Stored: Gravity pulls objects down
# Buffer size: 3/5
#
# 2. Testing retrieval:
# Query: What color is the sky?
# Retrieved 2 items
#   - The sky is blue
#   - Water boils at 100°C
#
# 3. Testing compression (adding more items):
# Added: Plants need sunlight
# Buffer size: 4/5
# Added: Fish live in water
# Buffer size: 5/5
# Added: Birds can fly
# Buffer size: 3/5
#
# Compression occurred!
# Compressed items: 2
# Summary preview: The key points are: the sky is blue, water boils at 100°C...
#
# 4. Current state:
# Buffer size: 3
# Compressed count: 1
# Capacity usage: 60.0%