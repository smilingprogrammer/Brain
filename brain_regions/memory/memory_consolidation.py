import asyncio
from typing import Dict, Any, List
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog
import time

logger = structlog.get_logger()


class MemoryConsolidation(BrainRegion):
    """Orchestrates memory consolidation between hippocampus and cortex"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.consolidation_active = False
        self.consolidation_cycles = 0
        self.last_consolidation = None

        # Parameters
        self.consolidation_interval = 300  # 5 minutes
        self.replay_speed = 20  # 20x speed
        self.theta_rhythm = 8  # Hz

    async def initialize(self):
        """Initialize memory consolidation system"""
        logger.info("initializing_memory_consolidation")

        # Start consolidation loop
        asyncio.create_task(self._consolidation_loop())

        # Subscribe to sleep/rest signals
        self.event_bus.subscribe("enter_rest_state", self._on_rest_state)
        self.event_bus.subscribe("high_memory_pressure", self._on_memory_pressure)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process consolidation requests"""

        command = input_data.get("command", "status")

        if command == "consolidate_now":
            await self._run_consolidation_cycle()
            return {"success": True, "cycles": self.consolidation_cycles}
        elif command == "status":
            return self.get_state()
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

    async def _consolidation_loop(self):
        """Main consolidation loop"""

        self.consolidation_active = True

        while self.consolidation_active:
            try:
                # Wait for interval
                await asyncio.sleep(self.consolidation_interval)

                # Run consolidation
                await self._run_consolidation_cycle()

            except Exception as e:
                logger.error("consolidation_error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry

    async def _run_consolidation_cycle(self):
        """Run a single consolidation cycle"""

        logger.info("starting_consolidation_cycle", cycle=self.consolidation_cycles)

        start_time = time.time()

        # Phase 1: Hippocampal replay
        await self._hippocampal_replay()

        # Phase 2: Cortical integration
        await self._cortical_integration()

        # Phase 3: Synaptic homeostasis
        await self._synaptic_homeostasis()

        duration = time.time() - start_time

        self.consolidation_cycles += 1
        self.last_consolidation = time.time()

        logger.info("consolidation_complete",
                    cycle=self.consolidation_cycles,
                    duration=duration)

        # Emit completion event
        await self.event_bus.emit("consolidation_cycle_complete", {
            "cycle": self.consolidation_cycles,
            "duration": duration
        })

    async def _hippocampal_replay(self):
        """Replay hippocampal memories at high speed"""

        # Request recent episodes from hippocampus
        await self.event_bus.emit("consolidate_memories", {
            "phase": "replay",
            "speed": self.replay_speed
        })

        # Simulate theta rhythm modulation
        for _ in range(int(self.theta_rhythm * 2)):  # 2 seconds of theta
            await asyncio.sleep(1 / self.theta_rhythm)

            # Trigger replay burst
            await self.event_bus.emit("replay_burst", {
                "rhythm": "theta",
                "frequency": self.theta_rhythm
            })

    async def _cortical_integration(self):
        """Integrate replayed memories into cortical networks"""

        # Request semantic integration
        await self.event_bus.emit("integrate_episodic_to_semantic", {
            "source": "hippocampal_replay"
        })

        # Allow time for integration
        await asyncio.sleep(2.0)

    async def _synaptic_homeostasis(self):
        """Normalize synaptic weights (sleep homeostasis)"""

        # Request memory systems to prune weak connections
        await self.event_bus.emit("prune_weak_memories", {
            "threshold": 0.3
        })

        # Request working memory compression
        await self.event_bus.emit("compress_working_memory", {
            "aggressive": True
        })

    async def _on_rest_state(self, data: Dict):
        """Handle rest state entry"""

        logger.info("entering_rest_state_consolidation")

        # Accelerate consolidation during rest
        old_interval = self.consolidation_interval
        self.consolidation_interval = 60  # 1 minute during rest

        # Run immediate consolidation
        await self._run_consolidation_cycle()

        # Restore normal interval when rest ends
        # (Would need exit_rest_state event)

    async def _on_memory_pressure(self, data: Dict):
        """Handle high memory pressure"""

        pressure_level = data.get("level", "medium")

        if pressure_level == "high":
            logger.warning("high_memory_pressure_consolidation")

            # Emergency consolidation
            await self._run_consolidation_cycle()

    def get_state(self) -> Dict[str, Any]:
        return {
            "consolidation_active": self.consolidation_active,
            "consolidation_cycles": self.consolidation_cycles,
            "last_consolidation": self.last_consolidation,
            "consolidation_interval": self.consolidation_interval,
            "time_until_next": (
                self.consolidation_interval - (time.time() - self.last_consolidation)
                if self.last_consolidation else self.consolidation_interval
            )
        }