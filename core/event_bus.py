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
        logger.info("subscription_added", event_type=event_type, handler=handler.__name__)

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

        logger.info("event_emitted", event_type=event_type, queue_size=self.event_queue.qsize())
        await self.event_queue.put(event)

    async def process_events(self):
        """Main event processing loop"""
        self.running = True
        logger.info("event_bus_started", running=self.running)

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
                    logger.info("processing_event", event_type=event_type, handler_count=len(handlers))
                    with event_latency.labels(event_type=event_type).time():
                        results = await asyncio.gather(
                            *[handler(event["data"]) for handler in handlers],
                            return_exceptions=True
                        )
                        # Log any exceptions
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                logger.error("handler_exception", event_type=event_type, error=str(result))
                    logger.info("event_processed", event_type=event_type, queue_remaining=self.event_queue.qsize())

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("event_processing_error", error=str(e))

    def stop(self):
        """Stop the event bus"""
        self.running = False