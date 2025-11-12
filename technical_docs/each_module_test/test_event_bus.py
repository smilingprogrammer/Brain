# test_event_bus.py
import asyncio
from core.event_bus import EventBus


async def test_event_bus():
    # Create event bus
    bus = EventBus()

    # Create a test handler
    received_data = []

    async def test_handler(data):
        print(f"Handler received: {data}")
        received_data.append(data)

    # Subscribe to an event
    bus.subscribe("test_event", test_handler)

    # Start event processing
    asyncio.create_task(bus.process_events())

    # Emit some events
    await bus.emit("test_event", {"message": "Hello"})
    await bus.emit("test_event", {"message": "World"})

    # Wait for processing
    await asyncio.sleep(0.5)

    # Check results
    print(f"Received {len(received_data)} events")
    print(f"Data: {received_data}")

    # Test unsubscribe
    bus.unsubscribe("test_event", test_handler)
    await bus.emit("test_event", {"message": "Should not receive"})

    await asyncio.sleep(0.5)
    print(f"Final count: {len(received_data)} (should be 2)")


# Run the test
asyncio.run(test_event_bus())