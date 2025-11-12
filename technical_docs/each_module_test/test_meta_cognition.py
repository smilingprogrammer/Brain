# test_meta_cognition.py
import asyncio
from brain_regions.executive.meta_cognition import MetaCognition
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_meta_cognition():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    meta = MetaCognition(bus, gemini)
    await meta.initialize()

    print("=== Meta-Cognition Test ===\n")

    # Test 1: Cognitive state assessment
    print("1. Testing cognitive state assessment:")

    # Simulate some performance history
    test_performance = [
        {"success": True, "duration": 2.5, "confidence": 0.8, "task_type": "logical"},
        {"success": True, "duration": 3.0, "confidence": 0.9, "task_type": "logical"},
        {"success": False, "duration": 5.0, "confidence": 0.4, "task_type": "creative"},
        {"success": True, "duration": 2.0, "confidence": 0.85, "task_type": "factual"},
        {"success": False, "duration": 6.0, "confidence": 0.3, "task_type": "creative"}
    ]

    # Add to history
    for perf in test_performance:
        await bus.emit("task_complete", perf)

    await asyncio.sleep(0.5)  # Let events process

    # Assess state
    assessment = await meta.process({"operation": "assess"})

    print(f"\nCognitive State:")
    for key, value in assessment['cognitive_state'].items():
        print(f"  {key}: {value:.2f}")

    print(f"\nActive strategies: {assessment['active_strategies']}")
    print(f"\nQualitative assessment: {assessment['qualitative_assessment'][:200]}...")

    # Test 2: Performance reflection
    print("\n2. Testing performance reflection:")

    reflection = await meta.process({"operation": "reflect"})

    print(f"\nPerformance Patterns:")
    patterns = reflection['patterns']
    print(f"  Success by type: {dict(list(patterns['success_by_type'].items())[:3])}")
    print(f"  Error sequences: {patterns['error_sequences']}")
    print(f"  Areas for improvement: {patterns['improvement_areas']}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(reflection['recommendations'][:3]):
        print(f"  {i + 1}. {rec}")

    # Test 3: Strategy adaptation
    print("\n3. Testing strategy adaptation:")

    # Force low confidence
    meta.cognitive_state['confidence'] = 0.3
    meta.cognitive_state['cognitive_load'] = 0.9

    adaptation = await meta.process({"operation": "adapt"})

    print(f"\nAdaptations made:")
    for adapt in adaptation['adaptations']:
        print(f"  - {adapt}")

    print(f"\nActive strategies after adaptation: {adaptation['active_strategies']}")

    # Test 4: Error tracking
    print("\n4. Testing error pattern tracking:")

    # Simulate errors
    error_types = ["timeout", "logic_error", "timeout", "memory_overflow", "timeout"]
    for error in error_types:
        await bus.emit("error_occurred", {"error_type": error})

    await asyncio.sleep(0.5)

    state = meta.get_state()
    print(f"\nError patterns detected:")
    for error_type, count in state['error_patterns'].items():
        print(f"  {error_type}: {count} occurrences")


asyncio.run(test_meta_cognition())