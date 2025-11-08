import asyncio
from typing import Dict, List, Any, Set
from collections import defaultdict
import numpy as np
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class GlobalWorkspace(BrainRegion):
    """Global Workspace Theory implementation for consciousness-like integration"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.workspace = {}
        self.attention_competition = {}
        self.broadcast_threshold = 0.7
        self.integration_history = []

        # Track which regions are competing for attention
        self.competing_regions: Set[str] = set()

    async def initialize(self):
        """Initialize global workspace"""
        logger.info("initializing_global_workspace")

        # Subscribe to all brain region outputs
        self.event_bus.subscribe("language_comprehension_complete",
                                 lambda d: self._add_to_competition("language", d))
        self.event_bus.subscribe("working_memory_updated",
                                 lambda d: self._add_to_competition("working_memory", d))
        self.event_bus.subscribe("logical_reasoning_complete",
                                 lambda d: self._add_to_competition("logical_reasoning", d))
        self.event_bus.subscribe("episodic_retrieval_complete",
                                 lambda d: self._add_to_competition("episodic", d))

        # Start integration loop
        asyncio.create_task(self._integration_loop())

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process is handled by the integration loop"""
        return self.get_state()

    async def _integration_loop(self):
        """Main consciousness loop - integrate and broadcast"""

        while True:
            try:
                # Wait for enough regions to compete
                if len(self.attention_competition) >= 2:
                    # Run competition
                    winners = await self._run_attention_competition()

                    if winners:
                        # Integrate winning representations
                        integrated = await self._integrate_representations(winners)

                        # Update workspace
                        self.workspace = integrated

                        # Broadcast to all regions
                        await self._broadcast_global_state(integrated)

                        # Store in history
                        self.integration_history.append({
                            "timestamp": asyncio.get_event_loop().time(),
                            "winners": list(winners.keys()),
                            "integrated_state": integrated
                        })

                        # Clear competition for next cycle
                        self.attention_competition.clear()

                # Small delay for next cycle
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error("global_workspace_error", error=str(e))
                await asyncio.sleep(0.5)

    async def _add_to_competition(self, region: str, data: Dict):
        """Add region output to attention competition"""

        # Compute salience score
        salience = await self._compute_salience(region, data)

        self.attention_competition[region] = {
            "data": data,
            "salience": salience,
            "timestamp": asyncio.get_event_loop().time()
        }

        self.competing_regions.add(region)

    async def _run_attention_competition(self) -> Dict:
        """Select winning representations for global broadcast"""

        # Sort by salience
        sorted_regions = sorted(
            self.attention_competition.items(),
            key=lambda x: x[1]["salience"],
            reverse=True
        )

        # Select winners above threshold
        winners = {}
        total_salience = 0

        for region, info in sorted_regions:
            if info["salience"] >= self.broadcast_threshold:
                winners[region] = info
                total_salience += info["salience"]

                # Limit to top 3 to prevent overload
                if len(winners) >= 3:
                    break

        # Normalize salience scores
        if total_salience > 0:
            for region in winners:
                winners[region]["normalized_salience"] = (
                        winners[region]["salience"] / total_salience
                )

        return winners

    async def _integrate_representations(self, winners: Dict) -> Dict:
        """Integrate multiple representations into coherent global state"""

        integrated = {
            "primary_focus": None,
            "secondary_elements": [],
            "integrated_meaning": None,
            "action_implications": [],
            "confidence": 0.0
        }

        # Identify primary focus (highest salience)
        if winners:
            primary_region = max(winners, key=lambda r: winners[r]["salience"])
            integrated["primary_focus"] = {
                "region": primary_region,
                "content": winners[primary_region]["data"]
            }

            # Add secondary elements
            for region, info in winners.items():
                if region != primary_region:
                    integrated["secondary_elements"].append({
                        "region": region,
                        "content": info["data"],
                        "relevance": info["normalized_salience"]
                    })

        # Compute integrated meaning
        integrated["integrated_meaning"] = await self._compute_integrated_meaning(winners)

        # Extract action implications
        integrated["action_implications"] = self._extract_action_implications(winners)

        # Overall confidence
        integrated["confidence"] = np.mean([w["salience"] for w in winners.values()])

        return integrated

    async def _broadcast_global_state(self, integrated_state: Dict):
        """Broadcast integrated state to all brain regions"""

        await self.event_bus.emit("global_workspace_broadcast", {
            "integrated_state": integrated_state,
            "participating_regions": list(self.competing_regions),
            "integration_strength": integrated_state["confidence"]
        })

        logger.info("global_broadcast_sent",
                    regions=list(self.competing_regions),
                    confidence=integrated_state["confidence"])

    async def _compute_salience(self, region: str, data: Dict) -> float:
        """Compute salience score for attention competition"""

        base_salience = 0.5

        # Region-specific boosts
        region_weights = {
            "language": 0.8,
            "working_memory": 0.9,
            "logical_reasoning": 0.85,
            "episodic": 0.7
        }

        base_salience *= region_weights.get(region, 1.0)

        # Boost for high confidence
        if "confidence" in data:
            base_salience *= (0.5 + 0.5 * data["confidence"])

        # Boost for novelty
        if self._is_novel(data):
            base_salience *= 1.2

        # Boost for emotional salience
        if "emotional_valence" in data and abs(data["emotional_valence"]) > 0.5:
            base_salience *= 1.3

        return min(base_salience, 1.0)

    async def _compute_integrated_meaning(self, winners: Dict) -> str:
        """Compute integrated meaning across representations"""

        # Simple integration - could be enhanced with more sophisticated binding
        meanings = []

        for region, info in winners.items():
            data = info["data"]

            if region == "language" and "original_text" in data:
                meanings.append(f"Language: {data['original_text']}")
            elif region == "logical_reasoning" and "conclusion" in data:
                meanings.append(f"Logic: {data['conclusion']}")
            elif region == "working_memory" and "current_contents" in data:
                if data["current_contents"]:
                    recent = data["current_contents"][-1]
                    meanings.append(f"WM: {recent.get('summary', 'active memory')}")

        return " | ".join(meanings) if meanings else "Integrated state"

    def _extract_action_implications(self, winners: Dict) -> List[str]:
        """Extract potential actions from integrated state"""

        actions = []

        for region, info in winners.items():
            data = info["data"]

            if "action_required" in data:
                actions.append(data["action_required"])

            if region == "logical_reasoning" and "conclusion" in data:
                actions.append(f"Apply conclusion: {data['conclusion']}")

        return actions

    def _is_novel(self, data: Dict) -> bool:
        """Check if data represents novel information"""

        # Simple novelty check against recent history
        data_str = str(data)

        for hist in self.integration_history[-10:]:
            if data_str in str(hist["integrated_state"]):
                return False

        return True

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_workspace": self.workspace,
            "competing_regions": list(self.competing_regions),
            "competition_size": len(self.attention_competition),
            "history_length": len(self.integration_history),
            "last_integration": self.integration_history[-1] if self.integration_history else None
        }