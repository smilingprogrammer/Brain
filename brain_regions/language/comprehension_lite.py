"""
Lightweight Language Comprehension using Gemini API
Replaces spacy and sentence-transformers with Gemini API calls
"""
from typing import Dict, List, Any
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
from brain_regions.gemini.embedding_service import GeminiEmbeddingService
import structlog

logger = structlog.get_logger()


class LanguageComprehensionLite(BrainRegion):
    """Wernicke's area - language understanding using Gemini"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.embedder = GeminiEmbeddingService()
        self.state = {}

    async def initialize(self):
        """Initialize language models"""
        logger.info("initializing_language_comprehension_lite")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input into semantic representations"""

        text = input_data.get("text", "")

        # Use Gemini for linguistic analysis
        analysis = await self._analyze_with_gemini(text)

        # Generate embeddings with Gemini
        embedding = self.embedder.encode(text)

        # Sentence segmentation
        sentences = self._simple_sentence_split(text)
        sentence_embeddings = self.embedder.encode(sentences) if sentences else []

        result = {
            "original_text": text,
            "tokens": analysis.get("tokens", []),
            "lemmas": analysis.get("lemmas", []),
            "pos_tags": analysis.get("pos_tags", []),
            "entities": analysis.get("entities", []),
            "dependencies": analysis.get("dependencies", []),
            "embedding": embedding.tolist(),
            "sentences": sentences,
            "sentence_embeddings": [emb.tolist() for emb in sentence_embeddings] if len(sentence_embeddings) > 0 else [],
            "complexity_score": self._compute_complexity(analysis)
        }

        # Update state
        self.state = {
            "last_processed": text,
            "entity_count": len(analysis.get("entities", [])),
            "token_count": len(analysis.get("tokens", []))
        }

        # Emit comprehension complete event
        await self.event_bus.emit("language_comprehension_complete", result)

        return result

    async def _analyze_with_gemini(self, text: str) -> Dict:
        """Use Gemini to perform linguistic analysis"""

        prompt = f"""Analyze this text linguistically:
"{text}"

Extract:
1. tokens: List of word tokens
2. lemmas: Base forms of words
3. pos_tags: Part-of-speech tags (NOUN, VERB, ADJ, etc.)
4. entities: Named entities with their types (e.g., [["New York", "LOCATION"], ["IBM", "ORGANIZATION"]])
5. dependencies: Dependency relationships (e.g., [["ran", "nsubj", "John"]])

Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "tokens": ["string"],
                "lemmas": ["string"],
                "pos_tags": ["string"],
                "entities": [["string", "string"]],
                "dependencies": [["string", "string", "string"]]
            }
        )

        if response["success"] and response["parsed"]:
            return response["parsed"]

        # Fallback: basic tokenization
        return {
            "tokens": text.split(),
            "lemmas": text.lower().split(),
            "pos_tags": ["UNKNOWN"] * len(text.split()),
            "entities": [],
            "dependencies": []
        }

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_complexity(self, analysis: Dict) -> float:
        """Compute text complexity score"""
        token_count = len(analysis.get("tokens", []))
        entity_count = len(analysis.get("entities", []))

        # Simple complexity based on length and entities
        complexity = (token_count * 0.1 + entity_count * 0.3)
        return min(complexity, 1.0)

    def get_state(self) -> Dict[str, Any]:
        return self.state
