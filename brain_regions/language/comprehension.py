from sentence_transformers import SentenceTransformer
import spacy
from typing import Dict, List, Any
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class LanguageComprehension(BrainRegion):
    """Wernicke's area - language understanding"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.state = {}

    async def initialize(self):
        """Initialize language models"""
        logger.info("initializing_language_comprehension")
        # Warm up models
        _ = self.embedder.encode("test")
        _ = self.nlp("test")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input into semantic representations"""

        text = input_data.get("text", "")

        # Spacy processing
        doc = self.nlp(text)

        # Extract linguistic features
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

        # Semantic embedding
        embedding = self.embedder.encode(text)

        # Sentence segmentation
        sentences = [sent.text for sent in doc.sents]
        sentence_embeddings = self.embedder.encode(sentences) if sentences else []

        result = {
            "original_text": text,
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "entities": entities,
            "dependencies": dependencies,
            "embedding": embedding.tolist(),
            "sentences": sentences,
            "sentence_embeddings": sentence_embeddings.tolist() if len(sentence_embeddings) > 0 else [],
            "complexity_score": self._compute_complexity(doc)
        }

        # Update state
        self.state = {
            "last_processed": text,
            "entity_count": len(entities),
            "token_count": len(tokens)
        }

        # Emit comprehension complete event
        await self.event_bus.emit("language_comprehension_complete", result)

        return result

    def _compute_complexity(self, doc) -> float:
        """Compute text complexity score"""
        # Simple complexity based on sentence length and dependency depth
        avg_sent_length = sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents)) if doc.sents else 0
        max_dep_depth = max((self._get_depth(token) for token in doc), default=0)

        return (avg_sent_length * 0.1 + max_dep_depth * 0.3)

    def _get_depth(self, token, depth=0) -> int:
        """Get dependency tree depth"""
        if list(token.children):
            return max(self._get_depth(child, depth + 1) for child in token.children)
        return depth

    def get_state(self) -> Dict[str, Any]:
        return self.state