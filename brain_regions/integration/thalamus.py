from typing import Dict, Any, Set
from collections import defaultdict
import asyncio
import time
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class Thalamus(BrainRegion):
    """Information routing and filtering hub"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # Routing tables
        self.routing_rules = defaultdict(list)  # source -> [(condition, target)]
        self.active_routes = defaultdict(set)  # source -> {targets}

        # Filtering and gating
        self.gate_states = defaultdict(lambda: 1.0)  # region -> openness (0-1)
        self.filter_thresholds = defaultdict(lambda: 0.3)

        # Information flow tracking
        self.flow_statistics = defaultdict(lambda: {"count": 0, "last_time": 0})
        self.bottlenecks = []

        self.state = {}

    async def initialize(self):
        """Initialize thalamus"""
        logger.info("initializing_thalamus")

        # Set up default routing rules
        self._setup_default_routes()

        # Subscribe to all events for routing
        self.event_bus.subscribe("route_information", self._on_route_request)
        self.event_bus.subscribe("update_gate", self._on_gate_update)

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process routing request"""

        source = input_data.get("source")
        data = input_data.get("data")
        priority = input_data.get("priority", 0.5)

        # Apply gating
        gate_value = self.gate_states[source]
        if gate_value < 0.1:
            return {
                "success": False,
                "reason": "Gate closed",
                "gate_value": gate_value
            }

        # Apply filtering
        if not self._passes_filter(source, data, priority):
            return {
                "success": False,
                "reason": "Filtered out",
                "threshold": self.filter_thresholds[source]
            }

        # Route information
        targets = await self._route(source, data)

        # Update statistics
        self._update_flow_stats(source, len(targets))

        return {
            "success": True,
            "routed_to": list(targets),
            "gate_value": gate_value
        }

    def _setup_default_routes(self):
        """Set up default routing rules"""

        # Sensory to processing
        self.routing_rules["language_comprehension"] = [
            (lambda d: True, "working_memory"),
            (lambda d: d.get("complexity_score", 0) > 0.7, "executive"),
            (lambda d: "question" in str(d).lower(), "reasoning")
        ]

        # Memory routes
        self.routing_rules["working_memory"] = [
            (lambda d: d.get("capacity_used", 0) > 0.8, "memory_consolidation"),
            (lambda d: True, "global_workspace")
        ]

        # Executive routes
        self.routing_rules["executive"] = [
            (lambda d: d.get("action_required", False), "motor"),
            (lambda d: d.get("reasoning_needed", False), "reasoning"),
            (lambda d: True, "global_workspace")
        ]

        # Reasoning routes
        self.routing_rules["reasoning"] = [
            (lambda d: True, "working_memory"),
            (lambda d: d.get("confidence", 0) > 0.8, "executive"),
            (lambda d: True, "global_workspace")
        ]

    async def _route(self, source: str, data: Dict) -> Set[str]:
        """Route information based on rules"""

        targets = set()

        # Check routing rules
        rules = self.routing_rules.get(source, [])

        for condition, target in rules:
            try:
                if condition(data):
                    # Check if target gate is open
                    if self.gate_states[target] > 0.1:
                        targets.add(target)

                        # Emit routed event
                        await self.event_bus.emit(f"{target}_input", {
                            "source": source,
                            "data": data,
                            "routed_by": "thalamus"
                        })
            except Exception as e:
                logger.error("routing_error",
                             source=source,
                             target=target,
                             error=str(e))

        # Update active routes
        self.active_routes[source] = targets

        return targets

    def _passes_filter(self, source: str, data: Dict, priority: float) -> bool:
        """Check if information passes filtering threshold"""

        threshold = self.filter_thresholds[source]

        # Priority can override threshold
        if priority > 0.9:
            return True

        # Check salience
        salience = data.get("salience", priority)

        # Apply gate modulation
        effective_threshold = threshold * (2 - self.gate_states[source])

        return salience >= effective_threshold

    def _update_flow_stats(self, source: str, target_count: int):
        """Update information flow statistics"""

        stats = self.flow_statistics[source]
        stats["count"] += 1
        stats["last_time"] = time.time()
        stats["targets"] = target_count

        # Check for bottlenecks
        if target_count == 0:
            self.bottlenecks.append({
                "source": source,
                "time": time.time(),
                "reason": "No valid targets"
            })

    async def _monitoring_loop(self):
        """Monitor information flow and adjust routing"""

        while True:
            try:
                # Check for overload
                await self._check_overload()

                # Adjust gates based on flow
                await self._adjust_gates()

                # Clean old bottlenecks
                self._clean_bottlenecks()

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error("thalamus_monitoring_error", error=str(e))

    async def _check_overload(self):
        """Check for information overload"""

        current_time = time.time()

        for source, stats in self.flow_statistics.items():
            # Check rate (messages per second)
            if stats["count"] > 0:
                time_diff = current_time - stats.get("start_time", current_time)
                if time_diff > 0:
                    rate = stats["count"] / time_diff

                    if rate > 10:  # More than 10 messages per second
                        # Reduce gate opening
                        self.gate_states[source] *= 0.9
                        logger.warning("thalamic_overload",
                                       source=source,
                                       rate=rate)

    async def _adjust_gates(self):
        """Dynamically adjust gate states"""

        # Gradually open closed gates
        for region in self.gate_states:
            if self.gate_states[region] < 1.0:
                self.gate_states[region] = min(
                    self.gate_states[region] * 1.05,
                    1.0
                )

        # Check for regions that need more flow
        for source, routes in self.active_routes.items():
            if not routes and self.flow_statistics[source]["count"] > 0:
                # No active routes but trying to send - open gates
                for target in self.routing_rules.get(source, []):
                    if isinstance(target, tuple):
                        target = target[1]
                    self.gate_states[target] = min(
                        self.gate_states[target] * 1.1,
                        1.0
                    )

    def _clean_bottlenecks(self):
        """Remove old bottleneck records"""

        current_time = time.time()
        self.bottlenecks = [
            b for b in self.bottlenecks
            if current_time - b["time"] < 60  # Keep last minute
        ]

    async def _on_route_request(self, data: Dict):
        """Handle routing requests"""
        result = await self.process(data)

        # Emit routing result
        await self.event_bus.emit("routing_complete", result)

    async def _on_gate_update(self, data: Dict):
        """Handle gate update requests"""

        region = data.get("region")
        value = data.get("value", 1.0)

        if region:
            old_value = self.gate_states[region]
            self.gate_states[region] = max(0.0, min(1.0, value))

            logger.info("thalamic_gate_updated",
                        region=region,
                        old_value=old_value,
                        new_value=self.gate_states[region])

    def get_state(self) -> Dict[str, Any]:
        return {
            "active_routes": {k: list(v) for k, v in self.active_routes.items()},
            "gate_states": dict(self.gate_states),
            "bottleneck_count": len(self.bottlenecks),
            "total_routed": sum(s["count"] for s in self.flow_statistics.values())
        }