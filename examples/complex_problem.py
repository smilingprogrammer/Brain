import asyncio
from main import CognitiveTextBrain


async def solve_complex_problems():
    """Demonstrate complex problem solving"""

    brain = CognitiveTextBrain()
    await brain.initialize()

    problems = [
        {
            "title": "Environmental Impact Analysis",
            "question": "If all cars suddenly became electric overnight, what would be the environmental and economic impacts?",
            "follow_up": "What infrastructure changes would be needed?"
        },
        {
            "title": "Future Society Scenario",
            "question": "How would human society change if we discovered a way to transfer consciousness between bodies?",
            "follow_up": "What ethical frameworks would we need to develop?"
        },
        {
            "title": "Scientific Hypothesis",
            "question": "If gravity worked in reverse for 1% of all matter, how would the universe be different?",
            "follow_up": "Could life still evolve under these conditions?"
        }
    ]

    for problem in problems:
        print(f"\n{'=' * 60}")
        print(f"PROBLEM: {problem['title']}")
        print(f"{'=' * 60}")

        # Initial question
        print(f"\nQuestion: {problem['question']}")
        print("\nThinking...")

        response = await brain.process_text(problem['question'])
        print(f"\nResponse: {response}")

        # Follow-up question
        print(f"\nFollow-up: {problem['follow_up']}")
        print("\nThinking...")

        follow_up_response = await brain.process_text(problem['follow_up'])
        print(f"\nResponse: {follow_up_response}")

    await brain.shutdown()


if __name__ == "__main__":
    asyncio.run(solve_complex_problems())