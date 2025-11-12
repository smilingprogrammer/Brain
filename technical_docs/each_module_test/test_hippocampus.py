# test_hippocampus.py
import asyncio
from brain_regions.memory.hippocampus import Hippocampus
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_hippocampus():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    hippo = Hippocampus(bus, gemini, capacity=100)
    await hippo.initialize()

    print("=== Hippocampus Test ===\n")

    # Test 1: Episode encoding
    print("1. Encoding episodes:")
    episodes = [
        {"event": "Learned about gravity", "context": "physics class", "emotion": "curious"},
        {"event": "Solved math problem", "context": "homework", "emotion": "satisfied"},
        {"event": "Read about neurons", "context": "neuroscience book", "emotion": "fascinated"}
    ]

    for ep in episodes:
        await hippo.store(ep, metadata={"source": "test"})
        print(f"Encoded: {ep['event']}")

    print(f"Total episodes: {hippo.state['episode_count']}")

    # Test 2: Pattern completion
    print("\n2. Testing retrieval with pattern completion:")
    queries = [
        "physics and gravity",
        "solving problems",
        "learning about brain"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        retrieved = await hippo.retrieve(query, k=2)

        for i, episode in enumerate(retrieved):
            content = episode.get('content', {})
            similarity = episode.get('retrieval_similarity', 0)
            print(f"  {i + 1}. {content.get('event', 'N/A')} (similarity: {similarity:.2f})")

    # Test 3: Consolidation
    print("\n3. Testing consolidation:")
    await hippo.consolidate()
    print(f"Replay count: {hippo.state['replay_count']}")

    # Test 4: Check state
    state = hippo.get_state()
    print(f"\nFinal state:")
    print(f"  Episodes: {state['episode_count']}")
    print(f"  Last retrieval: {state['last_retrieval']}")
    print(f"  Replay count: {state['replay_count']}")


asyncio.run(test_hippocampus())