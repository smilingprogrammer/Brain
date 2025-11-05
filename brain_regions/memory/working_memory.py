from collections import deque
from typing import Dict, Any, List, Optional
from core.interfaces import MemorySystem
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog
import time

logger = structlog.get_logger()


class WorkingMemory(MemorySystem):
    """Prefrontal cortex working memory with Gemini augmentation"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService, capacity: int = 7):
        self.event_bus = event_bus
        self.gemini = gemini
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.compressed_history = []
        self.attention_weights = {}

    async def initialize(self):
        """Initialize working memory"""
        logger.info("initializing_working_memory", capacity=self.capacity)

        # Subscribe to relevant events
        self.event_bus.subscribe("language_comprehension_complete", self._on_comprehension)
        self.event_bus.subscribe("episodic_retrieval_complete", self._on_episodic_retrieval)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and maintain information in working memory"""

        # Compute salience
        salience = await self._compute_salience(input_data)

        # Create memory item
        item = {
            "content": input_data,
            "timestamp": time.time(),
            "salience": salience,
            "access_count": 0
        }

        # Store with potential compression
        await self.store(item)

        # Update attention
        await self._update_attention()

        return {
            "current_contents": list(self.buffer),
            "attention_focus": self.get_focus(),
            "capacity_used": len(self.buffer) / self.capacity
        }

    async def store(self, data: Any, metadata: Optional[Dict] = None):
        """Store item in working memory"""

        # Check if compression needed
        if len(self.buffer) >= self.capacity:
            await self._compress_oldest()

        self.buffer.append(data)

        # Emit update event
        await self.event_bus.emit("working_memory_updated", {
            "new_item": data,
            "current_size": len(self.buffer)
        })

    async def retrieve(self, query: Any, k: int = 3) -> List:
        """Retrieve relevant items from working memory"""

        # Use Gemini to find relevant items
        prompt = f"""Given this query: {query}

        And these working memory contents:
        {self._format_buffer_contents()}
        
        Identify and rank the {k} most relevant items. For each:
        1. Item index
        2. Relevance score (0-1)
        3. Why it's relevant
        
        Output as JSON list."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "relevant_items": [
                    {
                        "index": "int",
                        "score": "float",
                        "reason": "string"
                    }
                ]
            }
        )

        if response["success"] and response["parsed"]:
            indices = [item["index"] for item in response["parsed"]["relevant_items"]]
            return [list(self.buffer)[i] for i in indices if i < len(self.buffer)]

        # Fallback to recency
        return list(self.buffer)[-k:]

    async def consolidate(self):
        """Consolidate working memory through compression"""

        if len(self.compressed_history) > 10:
            # Meta-compression of history
            prompt = f"""Compress these memory summaries into key insights:

            {self._format_compressed_history()}
            
            Extract:
            1. Recurring patterns
            2. Important relationships
            3. Key facts to remember"""

            response = await self.gemini.generate(prompt, config_name="balanced")

            if response["success"]:
                self.compressed_history = [{
                    "meta_summary": response["text"],
                    "timestamp": time.time(),
                    "original_count": len(self.compressed_history)
                }]

    async def _compress_oldest(self):
        """Compress oldest items using Gemini"""

        # Take oldest 40% of buffer
        compress_count = int(self.capacity * 0.4)
        to_compress = list(self.buffer)[:compress_count]

        prompt = f"""Compress these working memory items into a concise summary:

        Items:
        {self._format_items(to_compress)}
        
        Create a summary that:
        1. Preserves causal relationships
        2. Maintains temporal markers
        3. Keeps critical information
        4. Removes redundancy
        
        Be concise but complete."""

        response = await self.gemini.generate(prompt, config_name="fast")

        if response["success"]:
            compressed = {
                "summary": response["text"],
                "original_items": to_compress,
                "compression_time": time.time(),
                "item_count": len(to_compress)
            }

            self.compressed_history.append(compressed)

            # Remove compressed items
            for _ in range(compress_count):
                self.buffer.popleft()

    async def _compute_salience(self, data: Dict) -> float:
        """Compute salience score for new information"""

        # Factors: novelty, relevance to current context, emotional valence
        base_score = 0.5

        # Novelty bonus
        if self._is_novel(data):
            base_score += 0.2

        # Relevance to current focus
        if self.attention_weights:
            base_score += 0.3 * self._compute_relevance(data)

        return min(base_score, 1.0)

    async def _update_attention(self):
        """Update attention weights across buffer items"""

        # Decay existing weights
        for key in self.attention_weights:
            self.attention_weights[key] *= 0.9

        # Boost recently accessed
        for i, item in enumerate(self.buffer):
            key = f"item_{i}"
            access_boost = item.get("access_count", 0) * 0.1
            recency_boost = (1.0 - i / len(self.buffer)) * 0.2
            self.attention_weights[key] = min(
                self.attention_weights.get(key, 0) + access_boost + recency_boost,
                1.0
            )

    def get_focus(self) -> Dict:
        """Get current attention focus"""
        if not self.attention_weights:
            return {}

        # Get top attended item
        top_key = max(self.attention_weights, key=self.attention_weights.get)
        idx = int(top_key.split("_")[1])

        if idx < len(self.buffer):
            return {
                "focused_item": list(self.buffer)[idx],
                "attention_weight": self.attention_weights[top_key]
            }
        return {}

    def get_state(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "compressed_count": len(self.compressed_history),
            "attention_distribution": self.attention_weights,
            "capacity_usage": len(self.buffer) / self.capacity
        }

    # Helper methods
    def _format_buffer_contents(self) -> str:
        return "\n".join([
            f"{i}: {item.get('content', {}).get('summary', str(item)[:100])}"
            for i, item in enumerate(self.buffer)
        ])

    def _format_items(self, items: List) -> str:
        return "\n".join([
            f"- {item.get('content', {}).get('summary', str(item)[:100])}"
            for item in items
        ])

    def _format_compressed_history(self) -> str:
        return "\n".join([
            f"[{i}] {comp['summary'][:200]}..."
            for i, comp in enumerate(self.compressed_history[-5:])
        ])

    def _is_novel(self, data: Dict) -> bool:
        # Simple novelty check - could be enhanced
        content_str = str(data)
        for item in self.buffer:
            if content_str in str(item):
                return False
        return True

    def _compute_relevance(self, data: Dict) -> float:
        # Compute relevance to current focus
        focus = self.get_focus()
        if not focus:
            return 0.5

        # Simple string overlap - could use embeddings
        focus_str = str(focus.get("focused_item", ""))
        data_str = str(data)

        overlap = len(set(focus_str.split()) & set(data_str.split()))
        return min(overlap / 10, 1.0)

    # Event handlers
    async def _on_comprehension(self, data: Dict):
        """Handle language comprehension results"""
        await self.store({
            "type": "language_comprehension",
            "content": data,
            "summary": data.get("original_text", "")[:100]
        })

    async def _on_episodic_retrieval(self, data: Dict):
        """Handle episodic memory retrieval"""
        for memory in data.get("memories", [])[:2]:  # Only top 2
            await self.store({
                "type": "episodic_memory",
                "content": memory,
                "summary": f"Retrieved: {memory.get('summary', '')}"
            })