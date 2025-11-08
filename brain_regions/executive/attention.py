from typing import Dict, Any, List, Optional
from collections import defaultdict
import asyncio
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class AttentionController(BrainRegion):
    """Controls attention allocation across brain regions"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # Attention state
        self.attention_weights = defaultdict(float)
        self.focus_stack = []  # Stack of attention focuses
        self.attention_capacity = 1.0  # Total attention to distribute

        # Attention modes
        self.attention_mode = "distributed"  # distributed, focused, vigilant
        self.vigilance_level = 0.5

        # Saliency factors
        self.saliency_weights = {
            "novelty": 0.3,
            "relevance": 0.4,
            "urgency": 0.2,
            "emotional": 0.1
        }

        self.state = {}

    async def initialize(self):
        """Initialize attention controller"""
        logger.info("initializing_attention_controller")

        # Subscribe to attention requests
        self.event_bus.subscribe("request_attention", self._on_attention_request)
        self.event_bus.subscribe("release_attention", self._on_attention_release)
        self.event_bus.subscribe("global_workspace_broadcast", self._on_global_broadcast)

        # Start attention management loop
        asyncio.create_task(self._attention_loop())

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process attention control commands"""

        command = input_data.get("command", "status")

        if command == "focus":
            return await self._set_focus(input_data)
        elif command == "distribute":
            return await self._distribute_attention(input_data)
        elif command == "get_allocation":
            return self._get_attention_allocation()
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

    async def _attention_loop(self):
        """Main attention management loop"""

        while True:
            try:
                # Update attention based on current state
                await self._update_attention()

                # Decay unused attention weights
                await self._decay_attention()

                # Check for attention switches
                await self._check_attention_switch()

                await asyncio.sleep(0.1)  # 10Hz update

            except Exception as e:
                logger.error("attention_loop_error", error=str(e))

    async def _set_focus(self, data: Dict) -> Dict:
        """Set focused attention on specific target"""

        target = data.get("target")
        intensity = data.get("intensity", 0.8)

        # Push current focus to stack
        if self.focus_stack:
            self.focus_stack.append(self.focus_stack[-1])

        # Set new focus
        self.attention_mode = "focused"
        self.attention_weights.clear()
        self.attention_weights[target] = intensity

        # Distribute remaining attention
        remaining = self.attention_capacity - intensity
        self._distribute_remaining(remaining, exclude=[target])

        logger.info("attention_focused", target=target, intensity=intensity)

        # Emit focus event
        await self.event_bus.emit("attention_focused", {
            "target": target,
            "intensity": intensity
        })

        return {
            "success": True,
            "focus": target,
            "intensity": intensity
        }

    async def _distribute_attention(self, data: Dict) -> Dict:
        """Distribute attention across multiple targets"""

        targets = data.get("targets", [])
        weights = data.get("weights", None)

        self.attention_mode = "distributed"

        if weights:
            # Use provided weights
            total = sum(weights)
            for target, weight in zip(targets, weights):
                self.attention_weights[target] = weight / total
        else:
            # Equal distribution
            weight = self.attention_capacity / len(targets) if targets else 0
            for target in targets:
                self.attention_weights[target] = weight

        return {
            "success": True,
            "distribution": dict(self.attention_weights)
        }

    async def _update_attention(self):
        """Update attention based on current mode and inputs"""

        if self.attention_mode == "vigilant":
            # High alertness, ready to switch
            await self._vigilant_scan()
        elif self.attention_mode == "focused":
            # Maintain focus but monitor for interrupts
            await self._monitor_interrupts()
        else:
            # Distributed mode - balance based on saliency
            await self._balance_attention()

    async def _vigilant_scan(self):
        """Scan for high-priority attention targets"""

        # Request saliency from all regions
        await self.event_bus.emit("report_saliency", {})

        # Would process responses and update attention

    async def _monitor_interrupts(self):
        """Monitor for interrupt-worthy events while focused"""

        # Check if any region has urgency > threshold
        interrupt_threshold = 0.9

        # In real implementation, would check saliency reports
        # and potentially break focus if urgent

    async def _balance_attention(self):
        """Balance attention based on saliency"""

        # Compute saliency scores for each region
        saliency_scores = {}

        for region in self.attention_weights:
            score = await self._compute_saliency(region)
            saliency_scores[region] = score

        # Normalize to attention weights
        total_saliency = sum(saliency_scores.values())
        if total_saliency > 0:
            for region, score in saliency_scores.items():
                self.attention_weights[region] = score / total_saliency

    async def _compute_saliency(self, region: str) -> float:
        """Compute saliency score for a region"""

        # Base saliency
        base = 0.5

        # Would compute based on:
        # - Novelty of information
        # - Relevance to current goals
        # - Urgency/time pressure
        # - Emotional significance

        return base

    async def _decay_attention(self):
        """Decay attention weights over time"""

        decay_rate = 0.95  # 5% decay per cycle

        for region in list(self.attention_weights.keys()):
            self.attention_weights[region] *= decay_rate

            # Remove if below threshold
            if self.attention_weights[region] < 0.01:
                del self.attention_weights[region]

    async def _check_attention_switch(self):
        """Check if attention should switch to new target"""

        # In focused mode, only switch for high urgency
        if self.attention_mode == "focused":
            return

        # Find highest saliency target not currently attended
        # Would implement switching logic here

    def _distribute_remaining(self, amount: float, exclude: List[str]):
        """Distribute remaining attention capacity"""

        # Simple equal distribution to non-excluded regions
        other_regions = ["working_memory", "reasoning", "language"]
        available = [r for r in other_regions if r not in exclude]

        if available:
            per_region = amount / len(available)
            for region in available:
                self.attention_weights[region] = per_region

    def _get_attention_allocation(self) -> Dict:
        """Get current attention allocation"""

        return {
            "success": True,
            "mode": self.attention_mode,
            "allocation": dict(self.attention_weights),
            "total_allocated": sum(self.attention_weights.values()),
            "vigilance_level": self.vigilance_level
        }

    async def _on_attention_request(self, data: Dict):
        """Handle attention requests from regions"""

        requester = data.get("requester")
        priority = data.get("priority", 0.5)
        duration = data.get("duration", None)

        # Evaluate request
        current_attention = self.attention_weights.get(requester, 0)

        if priority > 0.8 and current_attention < 0.5:
            # High priority, grant attention
            await self._set_focus({
                "target": requester,
                "intensity": min(priority, 0.9)
            })

            if duration:
                # Schedule release
                asyncio.create_task(self._scheduled_release(requester, duration))

    async def _on_attention_release(self, data: Dict):
        """Handle attention release requests"""

        target = data.get("target")

        if target in self.attention_weights:
            released = self.attention_weights[target]
            del self.attention_weights[target]

            # Redistribute released attention
            self._distribute_remaining(released, exclude=[])

    async def _on_global_broadcast(self, data: Dict):
        """React to global workspace broadcasts"""

        # Adjust attention based on global state
        integrated_state = data.get("integrated_state", {})

        if integrated_state.get("primary_focus"):
            focus_region = integrated_state["primary_focus"]["region"]

            # Increase attention to primary focus
            current = self.attention_weights.get(focus_region, 0)
            self.attention_weights[focus_region] = min(current + 0.2, 0.9)

    async def _scheduled_release(self, target: str, duration: float):
        """Release attention after duration"""

        await asyncio.sleep(duration)
        await self._on_attention_release({"target": target})

    def get_state(self) -> Dict[str, Any]:
        return {
            "attention_mode": self.attention_mode,
            "attention_allocation": dict(self.attention_weights),
            "focus_stack_depth": len(self.focus_stack),
            "vigilance_level": self.vigilance_level,
            "total_allocated": sum(self.attention_weights.values())
        }