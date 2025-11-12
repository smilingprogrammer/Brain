# test_creative_reasoning.py
import asyncio
from brain_regions.reasoning.creative_reasoning import CreativeReasoning
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_creative_reasoning():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    creative = CreativeReasoning(bus, gemini)
    await creative.initialize()

    print("=== Creative Reasoning Test ===\n")

    # Test problems
    test_problems = [
        {
            "problem": "Design a new type of transportation for a city with no roads",
            "constraints": ["Must be eco-friendly", "Accessible to all ages", "Cost-effective"]
        },
        {
            "problem": "Create a game that teaches empathy",
            "constraints": ["Playable by 2-6 people", "No technology required", "30 minutes or less"]
        },
        {
            "problem": "Invent a solution for loneliness in elderly people",
            "constraints": ["Respects privacy", "Easy to use", "Promotes genuine connection"]
        }
    ]

    for test in test_problems:
        print(f"\nProblem: {test['problem']}")
        print(f"Constraints: {', '.join(test['constraints'])}")
        print("-" * 50)

        result = await creative.process({
            "problem": test['problem'],
            "constraints": test['constraints']
        })

        if result['success']:
            # Show problem analysis
            space = result.get('problem_space', {})
            print(f"\nProblem Analysis:")
            print(f"  Core challenge: {space.get('core_challenge', 'N/A')}")
            print(f"  Dimensions explored: {len(space.get('dimensions', []))}")

            # Show creative process
            process = result.get('creative_process', {})
            print(f"\nCreative Process:")
            print(f"  Ideas generated: {process.get('divergent_ideas', 0)}")
            print(f"  Combinations: {process.get('combinations', 0)}")
            print(f"  Transformations: {process.get('transformations', 0)}")

            # Show solutions
            solutions = result.get('solutions', [])
            print(f"\nTop Solutions:")
            for sol in solutions:
                print(f"\n{sol.get('rank', 0)}. {sol.get('idea', '')}")
                print(f"   Strategy: {sol.get('strategy', '')}")
                print(f"   Creativity score: {sol.get('creativity_score', 0):.2f}")
                print(f"   Novelty: {sol.get('novelty', 0):.2f}")
                print(f"   Usefulness: {sol.get('usefulness', 0):.2f}")


asyncio.run(test_creative_reasoning())