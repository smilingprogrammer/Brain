# test_prefrontal_cortex.py
import asyncio
from brain_regions.executive.prefrontal_cortex import PrefrontalCortex
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus


async def test_prefrontal_cortex():
    # Setup
    bus = EventBus()
    gemini = GeminiService()
    pfc = PrefrontalCortex(bus, gemini)
    await pfc.initialize()

    print("=== Prefrontal Cortex Test ===\n")

    # Test 1: Planning
    print("1. Testing executive planning:")

    test_task = "Solve the problem: How to reduce plastic waste in oceans"
    context = {
        "working_memory": "Plastic pollution is a major environmental issue",
        "knowledge": "Recycling, alternatives, cleanup technologies exist"
    }

    plan = await pfc._generate_plan(test_task, context)

    print(f"\nTask: {test_task}")
    print(f"\nGenerated Plan:")
    print(f"  Main goal: {plan['main_goal']}")
    print(f"  Sub-goals ({len(plan['sub_goals'])}):")

    for i, goal in enumerate(plan['sub_goals'][:3]):
        print(f"\n  {i + 1}. {goal['goal']}")
        print(f"     Strategy: {goal['strategy']}")
        print(f"     Priority: {goal['priority']}")
        print(f"     Required regions: {goal['required_regions']}")

    print(f"\n  Success criteria: {plan['success_criteria'][:2]}")
    print(f"  Contingency plans: {plan['contingency_plans'][:1]}")

    # Test 2: Progress monitoring
    print("\n2. Testing progress monitoring:")

    # Simulate some execution history
    pfc.strategy_history = [
        {"success": True, "strategies_used": ["logical"]},
        {"success": True, "strategies_used": ["creative"]},
        {"success": False, "strategies_used": ["logical"]},
        {"success": True, "strategies_used": ["causal"]}
    ]

    progress = await pfc._assess_progress()
    print(f"\nProgress Assessment:")
    print(f"  Success rate: {progress['success_rate']:.1%}")
    print(f"  Needs adaptation: {progress['needs_adaptation']}")
    print(f"  Active goals: {progress['active_goals']}")

    # Test 3: Strategy adaptation
    if progress['needs_adaptation']:
        print("\n3. Testing strategy adaptation:")
        await pfc._adapt_strategy(progress)

    # Check state
    state = pfc.get_state()
    print(f"\nPrefrontal Cortex State:")
    print(f"  Active goals: {state['active_goals']}")
    print(f"  Current plan: {state['current_plan']}")
    print(f"  Recent success rate: {state['recent_success_rate']:.1%}")
    print(f"  Monitoring active: {state['monitoring_active']}")

asyncio.run(test_prefrontal_cortex())