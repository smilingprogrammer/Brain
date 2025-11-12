import pytest
import asyncio
from main import CognitiveTextBrain


@pytest.mark.asyncio
async def test_simple_reasoning():
    """Test simple logical reasoning"""
    brain = CognitiveTextBrain()
    await brain.initialize()

    try:
        response = await brain.process_text(
            "If all cats are animals and all animals need food, do cats need food?"
        )

        assert response is not None
        assert "yes" in response.lower() or "cats need food" in response.lower()

    finally:
        await brain.shutdown()


@pytest.mark.asyncio
async def test_complex_reasoning():
    """Test complex multi-step reasoning"""
    brain = CognitiveTextBrain()
    await brain.initialize()

    try:
        response = await brain.process_text(
            "What would be the environmental impact if all cars suddenly became electric?"
        )

        assert response is not None
        assert len(response) > 100  # Should be a detailed response

    finally:
        await brain.shutdown()


@pytest.mark.asyncio
async def test_working_memory():
    """Test working memory integration"""
    brain = CognitiveTextBrain()
    await brain.initialize()

    try:
        # First statement
        await brain.process_text("Remember that John is a doctor.")

        # Query using context
        response = await brain.process_text("What is John's profession?")

        assert "doctor" in response.lower()

    finally:
        await brain.shutdown()