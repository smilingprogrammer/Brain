from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps
from typing import Callable

# Create metrics registry
metrics_registry = CollectorRegistry()

# Define metrics
event_counter = Counter(
    'brain_events_total',
    'Total number of brain events',
    ['event_type'],
    registry=metrics_registry
)

event_latency = Histogram(
    'brain_event_latency_seconds',
    'Latency of brain event processing',
    ['event_type'],
    registry=metrics_registry
)

working_memory_usage = Gauge(
    'working_memory_usage_ratio',
    'Working memory capacity usage',
    registry=metrics_registry
)

reasoning_confidence = Gauge(
    'reasoning_confidence',
    'Current reasoning confidence level',
    ['reasoning_type'],
    registry=metrics_registry
)

active_goals = Gauge(
    'executive_active_goals',
    'Number of active goals in executive',
    registry=metrics_registry
)

attention_competition_size = Gauge(
    'attention_competition_size',
    'Number of regions competing for attention',
    registry=metrics_registry
)


# Decorator for timing functions
def measure_latency(event_type: str):
    """Decorator to measure function execution time"""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                event_latency.labels(event_type=event_type).observe(latency)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                event_latency.labels(event_type=event_type).observe(latency)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class MetricsCollector:
    """Collects and exposes metrics for monitoring"""

    def __init__(self):
        self.start_time = time.time()

    def update_working_memory_usage(self, current: int, capacity: int):
        """Update working memory usage metric"""
        usage_ratio = current / capacity if capacity > 0 else 0
        working_memory_usage.set(usage_ratio)

    def update_reasoning_confidence(self, reasoning_type: str, confidence: float):
        """Update reasoning confidence metric"""
        reasoning_confidence.labels(reasoning_type=reasoning_type).set(confidence)

    def update_active_goals(self, count: int):
        """Update active goals count"""
        active_goals.set(count)

    def update_attention_competition(self, size: int):
        """Update attention competition size"""
        attention_competition_size.set(size)

    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time


# Global metrics collector
metrics_collector = MetricsCollector()