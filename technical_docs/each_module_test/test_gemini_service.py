# test_gemini_service.py
import asyncio
from brain_regions.gemini.gemini_service import GeminiService
from config.settings import settings


async def test_gemini_service():
    # Setup
    gemini = GeminiService()

    print("=== Gemini Service Test ===\n")

    # Test 1: Basic generation
    print("1. Testing basic text generation:")

    prompts = [
        ("What is 2+2?", "fast"),
        ("Explain quantum computing in simple terms", "balanced"),
        ("Write a haiku about artificial intelligence", "creative")
    ]

    for prompt, config in prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Config: {config}")

        result = await gemini.generate(prompt, config_name=config)

        if result['success']:
            print(f"Response: {result['text'][:200]}...")
            print(f"Tokens used: {result['usage']}")
        else:
            print(f"Error: {result['error']}")

    # Test 2: Structured generation
    print("\n2. Testing structured generation:")

    prompt = "List 3 benefits of exercise"
    schema = {
        "benefits": [
            {
                "benefit": "string",
                "explanation": "string",
                "importance": "float"
            }
        ]
    }

    result = await gemini.generate_structured(prompt, schema)

    if result['success'] and result['parsed']:
        print(f"Structured response:")
        for i, benefit in enumerate(result['parsed']['benefits']):
            print(f"\n{i + 1}. {benefit['benefit']}")
            print(f"   {benefit['explanation']}")
            print(f"   Importance: {benefit['importance']}")

    # Test 3: Different generation configs
    print("\n3. Testing generation configs:")

    test_prompt = "Solve this problem: How to make cities more sustainable?"

    for config_name in ["fast", "balanced", "creative"]:
        print(f"\n{config_name.upper()} config:")
        result = await gemini.generate(test_prompt, config_name=config_name)

        if result['success']:
            print(f"Response length: {len(result['text'])} chars")
            print(f"Preview: {result['text'][:100]}...")


asyncio.run(test_gemini_service())