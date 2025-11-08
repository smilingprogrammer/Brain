import numpy as np
from typing import Dict, Any, List, Optional
import time
from collections import deque
import hashlib
from core.interfaces import MemorySystem
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class Hippocampus(MemorySystem):
    """Episodic memory formation and retrieval with pattern separation/completion"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService, capacity: int = 10000):
        self.event_bus = event_bus
        self.gemini = gemini
        self.capacity = capacity

        # Episodic memory store
        self.episodes = deque(maxlen=capacity)
        self.episode_index = {}  # Fast lookup by content hash

        # Pattern separation parameters
        self.separation_threshold = 0.7
        self.ca3_patterns = {}  # Autoassociative patterns
        self.ca1_encodings = {}  # Separated patterns

        # Replay buffer for consolidation
        self.replay_buffer = deque(maxlen=100)

        self.state = {
            "episode_count": 0,
            "last_retrieval": None,
            "replay_count": 0
        }

    async def initialize(self):
        """Initialize hippocampus"""
        logger.info("initializing_hippocampus", capacity=self.capacity)

        # Subscribe to events
        self.event_bus.subscribe("encode_episode", self._on_encode_request)
        self.event_bus.subscribe("working_memory_updated", self._on_working_memory_update)
        self.event_bus.subscribe("consolidate_memories", self._on_consolidation_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process hippocampal operations"""

        operation = input_data.get("operation", "encode")

        if operation == "encode":
            return await self._encode_episode(input_data)
        elif operation == "retrieve":
            return await self._retrieve_episodes(input_data)
        elif operation == "replay":
            return await self._replay_episodes()
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def store(self, data: Any, metadata: Optional[Dict] = None):
        """Store new episode"""

        episode = {
            "content": data,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "access_count": 0,
            "importance": await self._calculate_importance(data)
        }

        # Pattern separation
        separated_pattern = await self._pattern_separation(data)
        episode["pattern"] = separated_pattern

        # Generate content hash for fast lookup
        content_hash = self._generate_hash(data)
        episode["hash"] = content_hash

        # Store episode
        self.episodes.append(episode)
        self.episode_index[content_hash] = len(self.episodes) - 1

        # Update state
        self.state["episode_count"] = len(self.episodes)

        # Add to replay buffer if important
        if episode["importance"] > 0.7:
            self.replay_buffer.append(episode)

        logger.info("episode_encoded",
                    importance=episode["importance"],
                    pattern_dims=len(separated_pattern))

    async def retrieve(self, query: Any, k: int = 5) -> List[Dict]:
        """Retrieve relevant episodes using pattern completion"""

        # Encode query
        query_pattern = await self._pattern_separation(query)

        # Pattern completion in CA3
        completed_pattern = await self._pattern_completion(query_pattern)

        # Find similar episodes
        similarities = []
        for i, episode in enumerate(self.episodes):
            similarity = self._compute_similarity(
                completed_pattern,
                episode.get("pattern", [])
            )
            if similarity > self.separation_threshold:
                similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k episodes
        retrieved = []
        for idx, sim in similarities[:k]:
            episode = self.episodes[idx]
            episode["retrieval_similarity"] = sim
            episode["access_count"] += 1
            retrieved.append(episode)

        self.state["last_retrieval"] = {
            "query": str(query)[:100],
            "retrieved_count": len(retrieved)
        }

        # Emit retrieval event
        await self.event_bus.emit("episodic_retrieval_complete", {
            "memories": retrieved,
            "query": query
        })

        return retrieved

    async def consolidate(self):
        """Consolidate memories through replay"""

        if not self.replay_buffer:
            return

        logger.info("starting_memory_consolidation",
                    replay_count=len(self.replay_buffer))

        # Sample episodes for replay
        replay_episodes = list(self.replay_buffer)[-20:]  # Recent 20

        consolidated_insights = []

        for episode in replay_episodes:
            # Extract patterns and relationships
            insight = await self._extract_insight(episode)
            if insight:
                consolidated_insights.append(insight)

        # Synthesize insights
        if consolidated_insights:
            synthesis = await self._synthesize_insights(consolidated_insights)

            # Store as new semantic memory
            await self.event_bus.emit("store_semantic_memory", {
                "content": synthesis,
                "source": "hippocampal_consolidation"
            })

        self.state["replay_count"] += len(replay_episodes)

        # Clear old replay buffer entries
        self.replay_buffer.clear()

    async def _encode_episode(self, data: Dict) -> Dict:
        """Encode a new episode"""

        await self.store(data.get("content"), data.get("metadata"))

        return {
            "success": True,
            "episode_count": self.state["episode_count"]
        }

    async def _retrieve_episodes(self, data: Dict) -> Dict:
        """Retrieve relevant episodes"""

        query = data.get("query")
        k = data.get("k", 5)

        episodes = await self.retrieve(query, k)

        return {
            "success": True,
            "episodes": episodes,
            "count": len(episodes)
        }

    async def _replay_episodes(self) -> Dict:
        """Replay episodes for consolidation"""

        await self.consolidate()

        return {
            "success": True,
            "replayed_count": self.state["replay_count"]
        }

    async def _pattern_separation(self, data: Any) -> List[float]:
        """Perform pattern separation (DG/CA1 function)"""

        # Convert to string representation
        data_str = str(data)

        # Use Gemini to extract key features
        prompt = f"""Extract 10 key semantic features from this content:
{data_str[:500]}

List features as single words or short phrases."""

        response = await self.gemini.generate(prompt, config_name="fast")

        if response["success"]:
            # Convert features to sparse vector
            features = response["text"].split('\n')[:10]

            # Create sparse high-dimensional representation
            pattern = np.zeros(1024)  # High dimensional

            for i, feature in enumerate(features):
                # Hash feature to multiple indices (sparse coding)
                indices = self._hash_to_indices(feature, n_indices=5, max_index=1024)
                for idx in indices:
                    pattern[idx] = 1.0

            # Add noise for separation
            noise = np.random.normal(0, 0.1, 1024)
            pattern += noise

            # Normalize
            pattern = pattern / (np.linalg.norm(pattern) + 1e-8)

            return pattern.tolist()

        # Fallback to random pattern
        return np.random.randn(1024).tolist()

    async def _pattern_completion(self, partial_pattern: List[float]) -> List[float]:
        """Complete partial pattern (CA3 function)"""

        pattern_np = np.array(partial_pattern)

        # Find most similar stored pattern
        best_match = None
        best_similarity = -1

        for stored_pattern in self.ca3_patterns.values():
            similarity = np.dot(pattern_np, stored_pattern) / (
                    np.linalg.norm(pattern_np) * np.linalg.norm(stored_pattern) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = stored_pattern

        if best_match is not None and best_similarity > 0.5:
            # Blend partial with best match (pattern completion)
            completed = 0.7 * best_match + 0.3 * pattern_np
            completed = completed / (np.linalg.norm(completed) + 1e-8)
            return completed.tolist()

        # No good match, return original
        return partial_pattern

    async def _calculate_importance(self, data: Any) -> float:
        """Calculate importance score for episode"""

        # Factors: novelty, emotional salience, relevance
        base_importance = 0.5

        # Check novelty
        data_hash = self._generate_hash(data)
        if data_hash not in self.episode_index:
            base_importance += 0.2  # Novel

        # Check for emotional or significant content
        data_str = str(data).lower()
        significant_words = [
            "important", "critical", "danger", "success", "failure",
            "breakthrough", "discovery", "error", "warning"
        ]

        if any(word in data_str for word in significant_words):
            base_importance += 0.2

        # Recency bias
        base_importance += 0.1  # Recent events more important

        return min(base_importance, 1.0)

    async def _extract_insight(self, episode: Dict) -> Optional[Dict]:
        """Extract insight from episode during consolidation"""

        content = episode.get("content", {})

        prompt = f"""Analyze this episode and extract a key insight or pattern:

Episode: {str(content)[:300]}
Importance: {episode.get('importance', 0)}

Extract:
1. Key pattern or relationship
2. General principle that could apply elsewhere
3. Connection to other knowledge

Be concise."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            return {
                "insight": response["text"],
                "source_episode": episode.get("hash"),
                "importance": episode.get("importance", 0)
            }

        return None

    async def _synthesize_insights(self, insights: List[Dict]) -> Dict:
        """Synthesize multiple insights into general knowledge"""

        insights_text = "\n".join([
            f"- {ins['insight']}" for ins in insights[:10]
        ])

        prompt = f"""Synthesize these insights into general principles:

{insights_text}

Create 2-3 general rules or patterns that capture the essence of these insights."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        return {
            "synthesis": response["text"] if response["success"] else "Consolidated patterns",
            "source_count": len(insights),
            "timestamp": time.time()
        }

    def _compute_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """Compute cosine similarity between patterns"""

        if not pattern1 or not pattern2:
            return 0.0

        p1 = np.array(pattern1)
        p2 = np.array(pattern2)

        similarity = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)

        return float(similarity)

    def _generate_hash(self, data: Any) -> str:
        """Generate hash for content"""

        content_str = str(data)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _hash_to_indices(self, feature: str, n_indices: int, max_index: int) -> List[int]:
        """Hash feature to multiple indices for sparse coding"""

        indices = []
        for i in range(n_indices):
            # Create unique hash for each index
            hash_str = f"{feature}_{i}"
            hash_val = int(hashlib.md5(hash_str.encode()).hexdigest(), 16)
            index = hash_val % max_index
            indices.append(index)

        return indices

    async def _on_encode_request(self, data: Dict):
        """Handle episode encoding requests"""
        await self.process({"operation": "encode", **data})

    async def _on_working_memory_update(self, data: Dict):
        """Monitor working memory for important episodes"""

        # Check if this should be encoded as episode
        new_item = data.get("new_item", {})

        if new_item.get("salience", 0) > 0.8:
            # High salience items become episodes
            await self.store(new_item, {"source": "working_memory"})

    async def _on_consolidation_request(self, data: Dict):
        """Handle consolidation requests"""
        await self.consolidate()

    def get_state(self) -> Dict[str, Any]:
        return self.state