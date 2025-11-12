# test_performance_monitoring.py
import asyncio
from core.metrics import metrics_collector, event_counter, reasoning_confidence
from main import CognitiveTextBrain


async def test_performance_monitoring():
    print("=== Performance Monitoring Test ===\n")

    # Initialize brain
    brain = CognitiveTextBrain()
    await brain.initialize()

    # Process several queries to generate metrics
    queries = [
        "What is 2+2?",
        "Explain the theory of evolution",
        "If A implies B and B implies C, does A imply C?",
        "Design a sustainable city",
        "What causes rain?"
    ]

    print("Processing queries to generate metrics...\n")

    for query in queries:
        print(f"Processing: {query}")
        await brain.process_text(query)
        await asyncio.sleep(0.5)  # Small delay between queries

    # Check metrics
    print("\n=== Metrics Report ===")

    # Get event counts
    print("\nEvent Counts:")
    # In a real implementation, you would access the metrics registry
    # This is a simplified example

    # Update some metrics manually for demonstration
    metrics_collector.update_working_memory_usage(5, 7)
    metrics_collector.update_reasoning_confidence("logical", 0.85)
    metrics_collector.update_active_goals(3)
    metrics_collector.update_attention_competition(4)

    print(f"System uptime: {metrics_collector.get_uptime():.1f} seconds")

    # Simulate getting metrics from Prometheus registry
    print("\nCognitive Metrics:")
    print(f"  Working Memory Usage: 71.4% (5/7 slots)")
    print(f"  Logical Reasoning Confidence: 0.85")
    print(f"  Active Goals: 3")
    print(f"  Attention Competition Size: 4 regions")

    # Test brain region states
    print("\n=== Brain Region States ===")

    for region_name, region in brain.regions.items():
        state = region.get_state()
        print(f"\n{region_name.upper()}:")

        # Display key metrics for each region
        if region_name == "working_memory":
            print(f"  Buffer size: {state.get('buffer_size', 0)}")
            print(f"  Capacity usage: {state.get('capacity_usage', 0):.1%}")
        elif region_name == "global_workspace":
            print(f"  Competing regions: {state.get('competing_regions', [])}")
            print(f"  Competition size: {state.get('competition_size', 0)}")
        elif region_name == "prefrontal_cortex":
            print(f"  Active goals: {state.get('active_goals', 0)}")
            print(f"  Success rate: {state.get('recent_success_rate', 0):.1%}")

    await brain.shutdown()
    print("\nMonitoring test complete!")


asyncio.run(test_performance_monitoring())