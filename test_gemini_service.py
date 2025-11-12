"""Quick test of the updated Gemini service"""
import asyncio
from brain_regions.gemini.gemini_service import GeminiService
from brain_regions.gemini.embedding_service import GeminiEmbeddingService

async def test_service():
    print("Testing GeminiService initialization...")
    service = GeminiService()
    print(f"[OK] GeminiService initialized with model: {service.model_name}")

    print("\nTesting text generation...")
    result = await service.generate("Say hello in one sentence", config_name="fast")
    if result["success"]:
        print(f"[OK] Generation successful: {result['text'][:100]}")
    else:
        print(f"[FAIL] Generation failed: {result['error']}")

    print("\nTesting embedding service...")
    embedder = GeminiEmbeddingService()
    print(f"[OK] EmbeddingService initialized with model: {embedder.model_name}")

    print("\nTesting embedding generation...")
    try:
        embedding = embedder.encode("Hello world")
        print(f"[OK] Embedding generated, shape: {embedding.shape}, first 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"[FAIL] Embedding failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_service())
