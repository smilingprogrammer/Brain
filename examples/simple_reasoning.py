import asyncio
from main import CognitiveTextBrain


async def run_examples():
    """Run simple reasoning examples"""

    brain = CognitiveTextBrain()
    await brain.initialize()

    examples = [
        "If it's raining, the ground gets wet. It's raining. What happens to the ground?",
        "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        "John is taller than Mary. Mary is taller than Sue. Who is the tallest?",
    ]

    for example in examples:
        print(f"\nQuery: {example}")
        response = await brain.process_text(example)
        print(f"Response: {response}")

    await brain.shutdown()


if __name__ == "__main__":
    asyncio.run(run_examples())