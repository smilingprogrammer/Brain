import asyncio
from typing import Dict, List, Tuple
from main import CognitiveTextBrain


class AccuracyBenchmark:
    """Benchmark accuracy of reasoning tasks"""

    def __init__(self):
        self.results = {}
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load test cases with expected answers"""

        return {
            "logical": [
                ("All mammals are animals. All dogs are mammals. Are all dogs animals?", "yes"),
                ("If A implies B and B implies C, does A imply C?", "yes"),
                ("Some birds can fly. Penguins are birds. Can all penguins fly?", "no")
            ],
            "mathematical": [
                ("What is 15% of 200?", "30"),
                ("If x + 5 = 12, what is x?", "7"),
                ("What is the next number in the sequence: 2, 4, 8, 16?", "32")
            ],
            "causal": [
                ("If you heat water to 100°C at sea level, what happens?", "boils"),
                ("What happens to plants without sunlight?", "die"),
                ("If demand increases and supply stays constant, what happens to price?", "increases")
            ],
            "factual": [
                ("What is the largest planet in our solar system?", "jupiter"),
                ("Who wrote Romeo and Juliet?", "shakespeare"),
                ("What is the chemical symbol for gold?", "au")
            ]
        }

    async def run_all_benchmarks(self):
        """Run all accuracy benchmarks"""

        print("Starting accuracy benchmarks...")

        # Initialize brain
        brain = CognitiveTextBrain()
        await brain.initialize()

        # Test each category
        for category, test_cases in self.test_cases.items():
            await self.benchmark_category(brain, category, test_cases)

        # Cleanup
        await brain.shutdown()

        # Report results
        self.report_results()

    async def benchmark_category(self,
                                 brain: CognitiveTextBrain,
                                 category: str,
                                 test_cases: List[Tuple[str, str]]):
        """Benchmark a category of questions"""

        correct = 0
        total = len(test_cases)
        details = []

        print(f"\nTesting {category} reasoning...")

        for question, expected in test_cases:
            response = await brain.process_text(question)

            # Check if response contains expected answer
            response_lower = response.lower()
            expected_lower = expected.lower()

            is_correct = expected_lower in response_lower

            if is_correct:
                correct += 1
                print(f"✓ {question[:50]}...")
            else:
                print(f"✗ {question[:50]}...")
                print(f"  Expected: {expected}")
                print(f"  Got: {response[:100]}...")

            details.append({
                "question": question,
                "expected": expected,
                "response": response,
                "correct": is_correct
            })

        accuracy = correct / total if total > 0 else 0

        self.results[category] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": details
        }

    def report_results(self):
        """Generate accuracy report"""

        print("\n" + "=" * 50)
        print("ACCURACY BENCHMARK RESULTS")
        print("=" * 50)

        total_correct = 0
        total_questions = 0

        for category, stats in self.results.items():
            print(f"\n{category.upper()}:")
            print(f"  Accuracy: {stats['accuracy'] * 100:.1f}%")
            print(f"  Correct:  {stats['correct']}/{stats['total']}")

            total_correct += stats['correct']
            total_questions += stats['total']

            # Show failures
            failures = [d for d in stats['details'] if not d['correct']]
            if failures:
                print(f"  Failed questions:")
                for f in failures[:3]:  # Show first 3 failures
                    print(f"    Q: {f['question'][:60]}...")
                    print(f"    Expected: {f['expected']}")

        # Overall accuracy
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        print(f"\nOVERALL ACCURACY: {overall_accuracy * 100:.1f}%")
        print(f"Total: {total_correct}/{total_questions} correct")


if __name__ == "__main__":
    benchmark = AccuracyBenchmark()
    asyncio.run(benchmark.run_all_benchmarks())