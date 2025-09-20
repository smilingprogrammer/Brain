import asyncio
from typing import Dict, Callable, Any, Set
from collections import defaultdict
import structlog
from .metrics import event_counter, event_latency

logger = structlog.get_logger()


class EventBus:
    """Async event bus for brain region communication"""

    def __init__(self):
        self.listeners: Dict[str, Set[Callable]] = defaultdict(set)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        self.listeners[event_type].add(handler)
        logger.info("subscription_added", event=event_type, handler=handler.__name__)

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        self.listeners[event_type].discard(handler)

    async def emit(self, event_type: str, data: Any):
        """Emit an event to all listeners"""
        event_counter.labels(event_type=event_type).inc()

        event = {
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }

        await self.event_queue.put(event)

    async def process_events(self):
        """Main event processing loop"""
        self.running = True

        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=0.1
                )

                event_type = event["type"]
                handlers = self.listeners.get(event_type, set())

                # Process all handlers concurrently
                if handlers:
                    with event_latency.labels(event_type=event_type).time():
                        await asyncio.gather(
                            *[handler(event["data"]) for handler in handlers],
                            return_exceptions=True
                        )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("event_processing_error", error=str(e))

    def stop(self):
        """Stop the event bus"""
        self.running = False