import spacy
from typing import Dict, List, Any
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from config.settings import settings
import structlog
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

logger = structlog.get_logger()


class SimpleEmbedder:
    """Small deterministic embedder for demos when model downloads are unavailable."""

    def encode(self, text_or_texts):
        import numpy as np

        texts = text_or_texts if isinstance(text_or_texts, list) else [text_or_texts]
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [(byte / 255.0) for byte in digest[:16]]
            vectors.append(values)
        arr = np.array(vectors, dtype=float)
        return arr if isinstance(text_or_texts, list) else arr[0]


class LanguageComprehension(BrainRegion):
    """Wernicke's area - language understanding"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.nlp = self._load_spacy_model()
        self.embedder = self._load_embedder()
        self.state = {}

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spacy_model_missing_using_blank_pipeline", model="en_core_web_sm")
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    def _load_embedder(self):
        if settings.brain_demo_mode or settings.use_local_embedder or SentenceTransformer is None:
            return SimpleEmbedder()

        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("sentence_transformer_unavailable_using_simple_embedder", error=str(exc))
            return SimpleEmbedder()

    async def initialize(self):
        """Initialize language models"""
        logger.info("initializing_language_comprehension")
        self.event_bus.subscribe("input_received", self._on_input_received)
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

    async def _on_input_received(self, data: Dict[str, Any]):
        """Handle raw text input events."""
        await self.process(data)
