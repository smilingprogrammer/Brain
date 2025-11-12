import asyncio
from main import CognitiveTextBrain


async def run_creative_tasks():
    """Demonstrate creative problem solving"""

    brain = CognitiveTextBrain()
    await brain.initialize()

    creative_prompts = [
        {
            "task": "Invention",
            "prompt": "Invent a device that helps people understand their pets' emotions better. Describe how it works and its key features.",
            "constraints": ["Must be non-invasive", "Affordable for average pet owner", "Works with multiple pet types"]
        },
        {
            "task": "Story Concept",
            "prompt": "Create a story concept where the main character can only move backwards through time, one day at a time.",
            "constraints": ["Must have a compelling conflict", "Explain how they communicate with others",
                            "Include a satisfying resolution"]
        },
        {
            "task": "Game Design",
            "prompt": "Design a board game that teaches quantum physics concepts to children.",
            "constraints": ["Age appropriate for 10-14 years", "Playable in 30-45 minutes", "Genuinely educational"]
        },
        {
            "task": "Social Innovation",
            "prompt": "Design a new social media platform that promotes deep thinking and meaningful connections instead of quick reactions.",
            "constraints": ["Must be engaging", "Financially sustainable", "Protects user privacy"]
        }
    ]

    for task_info in creative_prompts:
        print(f"\n{'=' * 60}")
        print(f"CREATIVE TASK: {task_info['task']}")
        print(f"{'=' * 60}")

        # Build prompt with constraints
        full_prompt = task_info['prompt']
        if task_info['constraints']:
            full_prompt += "\n\nConstraints:\n"
            for constraint in task_info['constraints']:
                full_prompt += f"- {constraint}\n"

        print(f"\nPrompt: {task_info['prompt']}")
        print(f"Constraints: {', '.join(task_info['constraints'])}")
        print("\nGenerating creative solution...")

        response = await brain.process_text(full_prompt)
        print(f"\nCreative Solution:\n{response}")

        # Ask for alternative approach
        alt_prompt = f"Provide a completely different approach to: {task_info['prompt']}"
        print("\nGenerating alternative approach...")

        alt_response = await brain.process_text(alt_prompt)
        print(f"\nAlternative Approach:\n{alt_response}")

    await brain.shutdown()


if __name__ == "__main__":
    asyncio.run(run_creative_tasks())