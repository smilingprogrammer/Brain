import asyncio
import time
from typing import Dict, List
import numpy as np
from main import CognitiveTextBrain


class LatencyBenchmark:
    """Benchmark latency of different operations"""

    def __init__(self):
        self.results = {}

    async def run_all_benchmarks(self):
        """Run all latency benchmarks"""

        print("Starting latency benchmarks...")

        # Initialize brain
        brain = CognitiveTextBrain()
        await brain.initialize()

        # Test different query types
        await self.benchmark_simple_query(brain)
        await self.benchmark_logical_reasoning(brain)
        await self.benchmark_complex_reasoning(brain)
        await self.benchmark_creative_task(brain)

        # Cleanup
        await brain.shutdown()

        # Report results
        self.report_results()

    async def benchmark_simple_query(self, brain: CognitiveTextBrain):
        """Benchmark simple factual queries"""

        queries = [
            "What is the capital of France?",
            "How many days are in a week?",
            "What color is the sky?"
        ]

        latencies = []

        for query in queries:
            start = time.time()
            response = await brain.process_text(query)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

            print(f"Simple query latency: {latency:.1f}ms")

        self.results["simple_query"] = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }

    async def benchmark_logical_reasoning(self, brain: CognitiveTextBrain):
        """Benchmark logical reasoning tasks"""

        queries = [
            "If all A are B and all B are C, are all A also C?",
            "John is taller than Mary. Mary is taller than Sue. Who is shortest?",
            "If it rains, the ground gets wet. The ground is wet. Did it rain?"
        ]

        latencies = []

        for query in queries:
            start = time.time()
            response = await brain.process_text(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            print(f"Logical reasoning latency: {latency:.1f}ms")

        self.results["logical_reasoning"] = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }

    async def benchmark_complex_reasoning(self, brain: CognitiveTextBrain):
        """Benchmark complex multi-step reasoning"""

        queries = [
            "What would happen to the economy if all jobs were automated?",
            "How would society change if humans could live for 500 years?",
            "What are the implications of discovering faster-than-light travel?"
        ]

        latencies = []

        for query in queries:
            start = time.time()
            response = await brain.process_text(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            print(f"Complex reasoning latency: {latency:.1f}ms")

        self.results["complex_reasoning"] = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }

    async def benchmark_creative_task(self, brain: CognitiveTextBrain):
        """Benchmark creative generation tasks"""

        queries = [
            "Create a new sport that combines chess and swimming",
            "Design a city for people who can fly",
            "Invent a new emotion and describe it"
        ]

        latencies = []

        for query in queries:
            start = time.time()
            response = await brain.process_text(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            print(f"Creative task latency: {latency:.1f}ms")

        self.results["creative_task"] = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }

    def report_results(self):
        """Generate benchmark report"""

        print("\n" + "=" * 50)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 50)

        for task, stats in self.results.items():
            print(f"\n{task.upper()}:")
            print(f"  Mean: {stats['mean']:.1f}ms")
            print(f"  Std:  {stats['std']:.1f}ms")
            print(f"  Min:  {stats['min']:.1f}ms")
            print(f"  Max:  {stats['max']:.1f}ms")

        # Overall statistics
        all_means = [stats['mean'] for stats in self.results.values()]
        print(f"\nOVERALL AVERAGE: {np.mean(all_means):.1f}ms")


if __name__ == "__main__":
    benchmark = LatencyBenchmark()
    asyncio.run(benchmark.run_all_benchmarks())