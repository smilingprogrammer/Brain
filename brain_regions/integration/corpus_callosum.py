import asyncio
from typing import Dict, Any
from collections import deque
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class CorpusCallosum(BrainRegion):
    """Inter-hemispheric integration and coordination"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # Hemisphere states
        self.left_hemisphere = {
            "dominant_process": None,
            "activity_level": 0.5,
            "specialization": ["language", "logic", "sequential"]
        }

        self.right_hemisphere = {
            "dominant_process": None,
            "activity_level": 0.5,
            "specialization": ["spatial", "creative", "holistic"]
        }

        # Integration state
        self.integration_strength = 0.7
        self.synchrony = 0.5
        self.transfer_queue = deque(maxlen=100)

        # Coordination patterns
        self.coordination_mode = "balanced"  # balanced, left-dominant, right-dominant

        self.state = {}

    async def initialize(self):
        """Initialize corpus callosum"""
        logger.info("initializing_corpus_callosum")

        # Subscribe to hemisphere-specific events
        self.event_bus.subscribe("left_hemisphere_activity", self._on_left_activity)
        self.event_bus.subscribe("right_hemisphere_activity", self._on_right_activity)
        self.event_bus.subscribe("integrate_hemispheres", self._on_integration_request)

        # Start coordination loop
        asyncio.create_task(self._coordination_loop())

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process inter-hemispheric communication"""

        operation = input_data.get("operation", "transfer")

        if operation == "transfer":
            return await self._transfer_information(input_data)
        elif operation == "synchronize":
            return await self._synchronize_hemispheres()
        elif operation == "coordinate":
            return await self._coordinate_processing(input_data)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _transfer_information(self, data: Dict) -> Dict:
        """Transfer information between hemispheres"""

        source = data.get("source_hemisphere")
        content = data.get("content")
        transfer_type = data.get("type", "general")

        # Check integration strength
        if self.integration_strength < 0.2:
            return {
                "success": False,
                "reason": "Weak inter-hemispheric connection"
            }

        # Determine if transfer is appropriate
        if self._should_transfer(source, content, transfer_type):
            # Add to transfer queue
            transfer_record = {
                "source": source,
                "content": content,
                "type": transfer_type,
                "timestamp": asyncio.get_event_loop().time()
            }

            self.transfer_queue.append(transfer_record)

            # Route to opposite hemisphere
            target = "right" if source == "left" else "left"

            await self.event_bus.emit(f"{target}_hemisphere_input", {
                "content": content,
                "from_hemisphere": source,
                "transfer_type": transfer_type,
                "integration_strength": self.integration_strength
            })

            return {
                "success": True,
                "transferred_to": target,
                "integration_strength": self.integration_strength
            }

        return {
            "success": False,
            "reason": "Transfer not appropriate for content type"
        }

    async def _synchronize_hemispheres(self) -> Dict:
        """Synchronize activity between hemispheres"""

        # Calculate synchrony based on activity levels
        left_activity = self.left_hemisphere["activity_level"]
        right_activity = self.right_hemisphere["activity_level"]

        # Phase difference
        phase_diff = abs(left_activity - right_activity)

        # Update synchrony
        self.synchrony = 1.0 - phase_diff

        # Adjust activities toward synchronization
        if self.coordination_mode == "balanced":
            target_activity = (left_activity + right_activity) / 2

            self.left_hemisphere["activity_level"] = (
                    0.8 * left_activity + 0.2 * target_activity
            )
            self.right_hemisphere["activity_level"] = (
                    0.8 * right_activity + 0.2 * target_activity
            )

        return {
            "success": True,
            "synchrony": self.synchrony,
            "left_activity": self.left_hemisphere["activity_level"],
            "right_activity": self.right_hemisphere["activity_level"]
        }

    async def _coordinate_processing(self, data: Dict) -> Dict:
        """Coordinate processing between hemispheres"""

        task_type = data.get("task_type")
        complexity = data.get("complexity", 0.5)

        # Determine optimal hemisphere assignment
        assignment = self._determine_hemisphere_assignment(task_type, complexity)

        # Update coordination mode
        if assignment["primary"] == "left":
            self.coordination_mode = "left-dominant"
        elif assignment["primary"] == "right":
            self.coordination_mode = "right-dominant"
        else:
            self.coordination_mode = "balanced"

        # Emit coordination instructions
        await self.event_bus.emit("hemisphere_coordination", {
            "mode": self.coordination_mode,
            "primary": assignment["primary"],
            "support": assignment["support"],
            "integration_level": assignment["integration_level"]
        })

        return {
            "success": True,
            "coordination_mode": self.coordination_mode,
            "assignment": assignment
        }

    def _should_transfer(self, source: str, content: Dict, transfer_type: str) -> bool:
        """Determine if information should be transferred"""

        # Always transfer if explicitly requested
        if transfer_type == "explicit":
            return True

        # Check content type against hemisphere specializations
        content_type = content.get("type", "general")

        if source == "left":
            # Left hemisphere content that might benefit right
            if content_type in ["spatial", "pattern", "emotional"]:
                return True
        else:
            # Right hemisphere content that might benefit left
            if content_type in ["linguistic", "sequential", "logical"]:
                return True

        # Transfer if integration strength is high
        return self.integration_strength > 0.7

    def _determine_hemisphere_assignment(self, task_type: str, complexity: float) -> Dict:
        """Determine optimal hemisphere assignment for task"""

        assignment = {
            "primary": "balanced",
            "support": "both",
            "integration_level": 0.5
        }

        # Task-specific assignments
        task_mappings = {
            "language": ("left", 0.8),
            "logic": ("left", 0.7),
            "math": ("left", 0.6),
            "spatial": ("right", 0.8),
            "creative": ("right", 0.7),
            "pattern": ("right", 0.6),
            "music": ("right", 0.7),
            "emotional": ("right", 0.6)
        }

        if task_type in task_mappings:
            primary, weight = task_mappings[task_type]
            assignment["primary"] = primary

            # High complexity requires both hemispheres
            if complexity > 0.7:
                assignment["support"] = "both"
                assignment["integration_level"] = 0.8
            else:
                assignment["support"] = "opposite" if primary else "both"
                assignment["integration_level"] = weight
        else:
            # Unknown task - use balanced approach
            assignment["integration_level"] = 0.7

        return assignment

    async def _coordination_loop(self):
        """Main coordination loop"""

        while True:
            try:
                # Update integration based on activity
                await self._update_integration()

                # Check for imbalances
                await self._check_hemispheric_balance()

                # Process pending transfers
                await self._process_transfers()

                await asyncio.sleep(0.5)  # 2Hz update

            except Exception as e:
                logger.error("corpus_callosum_error", error=str(e))

    async def _update_integration(self):
        """Update integration strength based on activity"""

        # Integration increases with balanced activity
        activity_diff = abs(
            self.left_hemisphere["activity_level"] -
            self.right_hemisphere["activity_level"]
        )

        if activity_diff < 0.2:
            # Balanced - increase integration
            self.integration_strength = min(
                self.integration_strength * 1.05,
                1.0
            )
        else:
            # Imbalanced - decrease integration
            self.integration_strength = max(
                self.integration_strength * 0.95,
                0.1
            )

    async def _check_hemispheric_balance(self):
        """Check and correct hemispheric imbalances"""

        left_activity = self.left_hemisphere["activity_level"]
        right_activity = self.right_hemisphere["activity_level"]

        # Detect severe imbalance
        if abs(left_activity - right_activity) > 0.5:
            logger.warning("hemispheric_imbalance",
                           left=left_activity,
                           right=right_activity)

            # Request rebalancing
            await self.event_bus.emit("rebalance_hemispheres", {
                "left_activity": left_activity,
                "right_activity": right_activity,
                "recommended_mode": "balanced"
            })

    async def _process_transfers(self):
        """Process queued inter-hemispheric transfers"""

        # Process recent transfers
        recent_transfers = list(self.transfer_queue)[-5:]

        if recent_transfers:
            # Analyze transfer patterns
            transfer_types = [t["type"] for t in recent_transfers]

            # Adjust integration based on transfer success
            if len(set(transfer_types)) > 3:
                # Diverse transfers - increase integration
                self.integration_strength = min(
                    self.integration_strength + 0.02,
                    1.0
                )

    async def _on_left_activity(self, data: Dict):
        """Handle left hemisphere activity reports"""

        self.left_hemisphere["activity_level"] = data.get("activity_level", 0.5)
        self.left_hemisphere["dominant_process"] = data.get("process")

        # Check if should share with right
        if data.get("shareable", False):
            await self._transfer_information({
                "source_hemisphere": "left",
                "content": data,
                "type": "activity_share"
            })

    async def _on_right_activity(self, data: Dict):
        """Handle right hemisphere activity reports"""

        self.right_hemisphere["activity_level"] = data.get("activity_level", 0.5)
        self.right_hemisphere["dominant_process"] = data.get("process")

        # Check if should share with left
        if data.get("shareable", False):
            await self._transfer_information({
                "source_hemisphere": "right",
                "content": data,
                "type": "activity_share"
            })

    async def _on_integration_request(self, data: Dict):
        """Handle explicit integration requests"""

        # Force synchronization
        await self._synchronize_hemispheres()

        # Increase integration temporarily
        old_strength = self.integration_strength
        self.integration_strength = min(self.integration_strength * 1.2, 1.0)

        logger.info("forced_integration",
                    old_strength=old_strength,
                    new_strength=self.integration_strength)

    def get_state(self) -> Dict[str, Any]:
        return {
            "left_hemisphere": self.left_hemisphere.copy(),
            "right_hemisphere": self.right_hemisphere.copy(),
            "integration_strength": self.integration_strength,
            "synchrony": self.synchrony,
            "coordination_mode": self.coordination_mode,
            "transfer_queue_size": len(self.transfer_queue)
        }