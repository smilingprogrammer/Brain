# run_all_tests.py
import asyncio
import sys
from typing import List, Tuple


async def run_test(test_name: str, test_func):
    """Run a single test and report results"""
    print(f"\n{'=' * 70}")
    print(f"Running: {test_name}")
    print(f"{'=' * 70}")

    try:
        await test_func()
        return (test_name, "PASSED", None)
    except Exception as e:
        return (test_name, "FAILED", str(e))


async def run_all_tests():
    """Run all tests in sequence"""

    print("🧠 COGNITIVE BRAIN COMPREHENSIVE TEST SUITE 🧠")
    print("=" * 70)

    # Import all test functions
    from test_event_bus import test_event_bus
    from test_language_comprehension import test_language_comprehension
    from test_working_memory import test_working_memory
    from test_logical_reasoning import test_logical_reasoning
    from test_gemini_service import test_gemini_service
    from test_full_system import test_full_system

    # Define test suite
    tests = [
        ("Event Bus", test_event_bus),
        ("Language Comprehension", test_language_comprehension),
        ("Working Memory", test_working_memory),
        ("Logical Reasoning", test_logical_reasoning),
        ("Gemini Service", test_gemini_service),
        ("Full System Integration", test_full_system)
    ]

    # Run tests
    results = []
    for test_name, test_func in tests:
        result = await run_test(test_name, test_func)
        results.append(result)
        await asyncio.sleep(1)  # Brief pause between tests

    # Report summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, status, error in results:
        if status == "PASSED":
            print(f"✅ {test_name}: {status}")
            passed += 1
        else:
            print(f"❌ {test_name}: {status}")
            if error:
                print(f"   Error: {error}")
            failed += 1

    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / len(results) * 100):.1f}%")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)