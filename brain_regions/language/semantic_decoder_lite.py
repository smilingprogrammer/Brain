"""
Lightweight Semantic Decoder using Gemini embeddings
Replaces sentence-transformers with Gemini API
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.embedding_service import GeminiEmbeddingService
import structlog

logger = structlog.get_logger()


class SemanticDecoderLite(BrainRegion):
    """Decode semantic meaning using Gemini embeddings"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.encoder = GeminiEmbeddingService()
        self.semantic_cache = {}
        self.concept_embeddings = {}
        self.state = {}

    async def initialize(self):
        """Initialize semantic decoder"""
        logger.info("initializing_semantic_decoder_lite")

        # Skip pre-computing embeddings for faster startup
        # Concepts will be loaded on-demand if needed
        logger.info("skipping_concept_preload_for_fast_startup")

        # Subscribe to events
        self.event_bus.subscribe("decode_semantics", self._on_decode_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process semantic decoding request"""

        text = input_data.get("text", "")
        mode = input_data.get("mode", "encode")  # encode or decode

        if mode == "encode":
            result = await self._encode_semantics(text)
        else:  # decode
            vector = input_data.get("vector", [])
            result = await self._decode_semantics(vector)

        # Update state
        self.state = {
            "last_operation": mode,
            "cache_size": len(self.semantic_cache)
        }

        return result

    async def _encode_semantics(self, text: str) -> Dict[str, Any]:
        """Encode text into semantic vector using Gemini"""

        # Check cache
        if text in self.semantic_cache:
            return {
                "success": True,
                "vector": self.semantic_cache[text],
                "cached": True
            }

        # Encode with Gemini
        try:
            vector = await self.encoder.encode_async(text)

            # Cache result
            self.semantic_cache[text] = vector

            # Find nearest concepts
            nearest_concepts = self._find_nearest_concepts(vector, k=5)

            return {
                "success": True,
                "vector": vector.tolist(),
                "nearest_concepts": nearest_concepts,
                "cached": False
            }

        except Exception as e:
            logger.error("semantic_encoding_error", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }

    async def _decode_semantics(self, vector: List[float]) -> Dict[str, Any]:
        """Decode semantic vector to nearest concepts"""

        vector_np = np.array(vector)

        # Find nearest concepts
        nearest_concepts = self._find_nearest_concepts(vector_np, k=10)

        # Generate description based on nearest concepts
        description = self._generate_semantic_description(nearest_concepts)

        return {
            "success": True,
            "concepts": nearest_concepts,
            "description": description
        }

    async def _initialize_concept_space(self):
        """Initialize embeddings for common concepts using Gemini"""

        # Core concepts for reasoning
        concepts = [
            # Physical concepts
            "density", "pressure", "gravity", "mass", "volume", "force",
            "temperature", "energy", "motion", "equilibrium",

            # Biological concepts
            "life", "organism", "adaptation", "evolution", "ecosystem",
            "survival", "metabolism", "reproduction", "habitat",

            # Abstract concepts
            "cause", "effect", "change", "stability", "system",
            "relationship", "pattern", "structure", "function",

            # Logical concepts
            "if", "then", "all", "some", "none", "implies",
            "therefore", "because", "however", "unless"
        ]

        # Encode all concepts with Gemini
        try:
            embeddings = await self.encoder.encode_async(concepts)
            for concept, embedding in zip(concepts, embeddings):
                self.concept_embeddings[concept] = embedding

            logger.info("concept_space_initialized", concept_count=len(concepts))
        except Exception as e:
            logger.error("concept_initialization_error", error=str(e))

    def _find_nearest_concepts(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest concepts to given vector"""

        similarities = []

        for concept, concept_vec in self.concept_embeddings.items():
            # Cosine similarity
            similarity = np.dot(vector, concept_vec) / (
                    np.linalg.norm(vector) * np.linalg.norm(concept_vec) + 1e-8
            )
            similarities.append((concept, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def _generate_semantic_description(self, concepts: List[Tuple[str, float]]) -> str:
        """Generate description from concept similarities"""

        if not concepts:
            return "No clear semantic interpretation"

        # Primary concept
        primary = concepts[0][0]

        # Related concepts with high similarity
        related = [c[0] for c in concepts[1:4] if c[1] > 0.7]

        if related:
            return f"Primarily about '{primary}', related to: {', '.join(related)}"
        else:
            return f"Primarily about '{primary}'"

    async def _on_decode_request(self, data: Dict):
        """Handle semantic decoding requests"""
        result = await self.process(data)
        await self.event_bus.emit("semantic_decode_complete", result)

    def get_state(self) -> Dict[str, Any]:
        return self.state
