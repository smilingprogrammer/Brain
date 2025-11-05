"""
Gemini-based text embedding service using new google-genai library
Replaces sentence-transformers with Gemini API calls
Uses the new Client-based API (google-genai v0.2.0+)
"""
from typing import List, Union
import asyncio
from google import genai
from google.genai import types
from config.settings import settings
import structlog
import numpy as np

logger = structlog.get_logger()


class GeminiEmbeddingService:
    """Generate embeddings using Gemini API"""

    def __init__(self):
        # New API: Client automatically uses GEMINI_API_KEY environment variable
        self.client = genai.Client(api_key=settings.gemini_api_key)
        # Use Gemini's embedding model
        self.model_name = "models/text-embedding-004"
        self.cache = {}

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True):
        """
        Encode text(s) to embeddings using Gemini API

        Args:
            texts: Single text or list of texts
            convert_to_numpy: If True, return numpy array

        Returns:
            Embedding(s) as numpy array or list
        """
        # Handle single string
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        embeddings = []
        for text in texts:
            # Check cache
            if text in self.cache:
                embeddings.append(self.cache[text])
                continue

            try:
                # Call Gemini embedding API using new API
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )

                # Extract embedding from result
                # New API returns result with embeddings attribute
                if hasattr(result, 'embeddings') and result.embeddings:
                    embedding = result.embeddings[0].values
                elif hasattr(result, 'embedding'):
                    embedding = result.embedding
                else:
                    # Fallback structure
                    embedding = result

                # Cache result
                self.cache[text] = embedding
                embeddings.append(embedding)

            except Exception as e:
                logger.error("gemini_embedding_error", error=str(e), text=text[:100], error_type=type(e).__name__)
                # Fallback: create zero embedding of standard dimension
                embeddings.append([0.0] * 768)

        if convert_to_numpy:
            embeddings = [np.array(emb) for emb in embeddings]

        # Return single embedding if input was single string
        if is_single:
            return embeddings[0]

        return np.array(embeddings) if convert_to_numpy else embeddings

    async def encode_async(self, texts: Union[str, List[str]], convert_to_numpy: bool = True):
        """
        Async version of encode

        Args:
            texts: Single text or list of texts
            convert_to_numpy: If True, return numpy array

        Returns:
            Embedding(s) as numpy array or list
        """
        return await asyncio.to_thread(self.encode, texts, convert_to_numpy)

    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("embedding_cache_cleared")
