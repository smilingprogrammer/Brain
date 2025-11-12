# test_knowledge_base.py
import asyncio
from brain_regions.gemini.knowledge_base import KnowledgeBase
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_knowledge_base():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    kb = KnowledgeBase(gemini, bus)
    await kb.initialize()

    print("=== Knowledge Base Test ===\n")

    # Test 1: Knowledge queries
    print("1. Testing knowledge queries:")

    queries = [
        ("What is photosynthesis?", "biology"),
        ("Explain the theory of relativity", "physics"),
        ("What are the principles of democracy?", None)
    ]

    for query, domain in queries:
        print(f"\nQuery: {query}")
        if domain:
            print(f"Domain: {domain}")

        result = await kb.query(query, domain)

        if result['success']:
            print(f"Knowledge retrieved: {result['knowledge'][:200]}...")

            if result['structured']:
                print(f"\nStructured data:")
                print(f"  Main concept: {result['structured'].get('main_concept')}")
                print(f"  Properties: {result['structured'].get('properties', [])[:3]}")

    # Test 2: Inference
    print("\n2. Testing inference:")

    premises = [
        "All mammals are warm-blooded",
        "Whales are mammals",
        "Warm-blooded animals regulate their body temperature"
    ]
    query = "Do whales regulate their body temperature?"

    print(f"Premises: {premises}")
    print(f"Query: {query}")

    result = await kb.infer(premises, query)

    if result['success']:
        inference = result['inference']
        print(f"\nInference:")
        print(f"  Conclusion: {inference['conclusion']}")
        print(f"  Confidence: {inference['confidence']}")
        print(f"  Steps: {inference['steps'][:2]}")

    # Test 3: Fact checking
    print("\n3. Testing fact checking:")

    statements = [
        "The Earth is flat",
        "Water boils at 100°C at sea level",
        "Humans use only 10% of their brain"
    ]

    for statement in statements:
        print(f"\nStatement: {statement}")

        result = await kb.check_fact(statement)

        if result['success']:
            fact_check = result['fact_check']
            print(f"  Verdict: {fact_check['verdict']}")
            print(f"  Confidence: {fact_check['confidence']}")
            print(f"  Explanation: {fact_check['explanation'][:100]}...")

    # Test 4: Related concepts
    print("\n4. Testing related concepts:")

    concepts = ["neuron", "democracy", "evolution"]

    for concept in concepts:
        print(f"\nConcept: {concept}")
        related = await kb.get_related_concepts(concept, n=5)
        print(f"Related concepts: {related}")


asyncio.run(test_knowledge_base())