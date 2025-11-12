# test_language_comprehension.py
import asyncio
from brain_regions.language.comprehension import LanguageComprehension
from core.event_bus import EventBus


async def test_language_comprehension():
    # Setup
    bus = EventBus()
    language = LanguageComprehension(bus)
    await language.initialize()

    # Test cases
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the meaning of life?",
        "If water becomes denser, marine life will be affected due to pressure changes."
    ]

    for text in test_texts:
        print(f"\n{'=' * 50}")
        print(f"Input: {text}")

        result = await language.process({"text": text})

        print(f"Tokens ({len(result['tokens'])}): {result['tokens'][:5]}...")
        print(f"Entities: {result['entities']}")
        print(f"Complexity: {result['complexity_score']:.2f}")
        print(f"Embedding shape: {len(result['embedding'])}")

        # Check dependencies
        if result['dependencies']:
            print(f"Dependencies: {result['dependencies'][:3]}")


asyncio.run(test_language_comprehension())

# Expected Output
# ==================================================
# Input: The quick brown fox jumps over the lazy dog.
# Tokens (10): ['The', 'quick', 'brown', 'fox', 'jumps']...
# Entities: []
# Complexity: 0.30
# Embedding shape: 768
# Dependencies: [('The', 'det', 'fox'), ('quick', 'amod', 'fox'), ('brown', 'amod', 'fox')]
#
# ==================================================
# Input: What is the meaning of life?
# Tokens (7): ['What', 'is', 'the', 'meaning', 'of']...
# Entities: []
# Complexity: 0.20
# Embedding shape: 768
# Dependencies: [('What', 'nsubj', 'is'), ('is', 'ROOT', 'is'), ('the', 'det', 'meaning')]