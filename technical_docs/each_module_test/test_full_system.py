# test_full_system.py
import asyncio
import time
from main import CognitiveTextBrain


async def test_full_system():
    print("=== Full Cognitive Brain System Test ===\n")

    # Initialize brain
    print("Initializing cognitive brain...")
    brain = CognitiveTextBrain()
    await brain.initialize()
    print("Brain initialized successfully!\n")

    # Test cases covering different cognitive abilities
    test_cases = [
        {
            "category": "Simple Factual",
            "query": "What is the capital of France?",
            "expected_elements": ["Paris"]
        },
        {
            "category": "Logical Reasoning",
            "query": "If all birds have wings and penguins are birds, do penguins have wings?",
            "expected_elements": ["yes", "penguins have wings"]
        },
        {
            "category": "Causal Analysis",
            "query": "What would happen if the moon suddenly disappeared?",
            "expected_elements": ["tides", "gravity", "orbit"]
        },
        {
            "category": "Creative Problem",
            "query": "Design a new sport that can be played in zero gravity",
            "expected_elements": ["float", "space", "rules"]
        },
        {
            "category": "Complex Reasoning",
            "query": "How would society change if humans could photosynthesize like plants?",
            "expected_elements": ["food", "energy", "social"]
        },
        {
            "category": "Analogical Thinking",
            "query": "How is the human brain similar to a computer network?",
            "expected_elements": ["processing", "memory", "connections"]
        }
    ]

    # Run tests
    for i, test in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"Test {i + 1}: {test['category']}")
        print(f"{'=' * 60}")
        print(f"Query: {test['query']}")

        # Measure time
        start_time = time.time()

        # Process query
        print("\nThinking...")
        response = await brain.process_text(test['query'])

        # Calculate latency
        latency = (time.time() - start_time) * 1000

        print(f"\nResponse: {response}")
        print(f"\nLatency: {latency:.1f}ms")

        # Check for expected elements
        response_lower = response.lower()
        found_elements = [elem for elem in test['expected_elements']
                          if elem.lower() in response_lower]

        print(f"Expected elements found: {len(found_elements)}/{len(test['expected_elements'])}")
        if len(found_elements) < len(test['expected_elements']):
            missing = [elem for elem in test['expected_elements']
                       if elem.lower() not in response_lower]
            print(f"Missing: {missing}")

    # Test conversation context
    print(f"\n{'=' * 60}")
    print("Testing Context Retention")
    print(f"{'=' * 60}")

    # First statement
    print("\nStatement 1: My favorite color is blue.")
    await brain.process_text("My favorite color is blue.")

    # Follow-up question
    print("\nQuestion: What is my favorite color?")
    response = await brain.process_text("What is my favorite color?")
    print(f"Response: {response}")

    if "blue" in response.lower():
        print("✓ Context retained successfully!")
    else:
        print("✗ Context retention failed")

    # Cleanup
    print("\n\nShutting down brain...")
    await brain.shutdown()
    print("Test complete!")


# Run the test
if __name__ == "__main__":
    asyncio.run(test_full_system())