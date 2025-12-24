"""
Lightweight Cognitive Brain - Using only Gemini API
No heavy ML dependencies required (torch, transformers, sentence-transformers, spacy)
"""
import asyncio
import sys
from typing import Optional
import structlog
from dotenv import load_dotenv

from config.settings import settings
from core.event_bus import EventBus
from core.logging import setup_logging

# Import brain regions (lightweight versions)
# NOTE: Import directly from modules to avoid heavy dependencies in __init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from brain_regions.gemini.gemini_service import GeminiService
from brain_regions.language.comprehension_lite import LanguageComprehensionLite
from brain_regions.language.production import LanguageProduction
from brain_regions.language.semantic_decoder_lite import SemanticDecoderLite
from brain_regions.memory.working_memory import WorkingMemory
from brain_regions.memory.hippocampus import Hippocampus
from brain_regions.reasoning.logical_reasoning_lite import LogicalReasoningLite
from brain_regions.reasoning.analogical_reasoning import AnalogicalReasoning
from brain_regions.reasoning.causal_reasoning import CausalReasoning
from brain_regions.reasoning.creative_reasoning_lite import CreativeReasoningLite
from brain_regions.integration.global_workspace import GlobalWorkspace
from brain_regions.executive.prefrontal_cortex_lite import PrefrontalCortexLite

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(settings.log_level)
logger = structlog.get_logger()


class CognitiveTextBrainLite:
    """Lightweight cognitive brain using only Gemini API"""

    def __init__(self):
        self.event_bus = EventBus()
        self.gemini = GeminiService()
        self.regions = {}
        self.initialized = False

    async def initialize(self):
        """Initialize all brain regions"""

        logger.info("initializing_lightweight_cognitive_brain")

        # Create brain regions (using lite versions where applicable)
        self.regions = {
            "language": LanguageComprehensionLite(self.event_bus, self.gemini),
            "language_production": LanguageProduction(self.event_bus, self.gemini),
            "semantic_decoder": SemanticDecoderLite(self.event_bus),
            "working_memory": WorkingMemory(self.event_bus, self.gemini),
            "hippocampus": Hippocampus(self.event_bus, self.gemini),
            "logical_reasoning": LogicalReasoningLite(self.event_bus, self.gemini),
            "analogical_reasoning": AnalogicalReasoning(self.event_bus, self.gemini),
            "causal_reasoning": CausalReasoning(self.event_bus, self.gemini),
            "creative_reasoning": CreativeReasoningLite(self.event_bus, self.gemini),
            "global_workspace": GlobalWorkspace(self.event_bus),
            "prefrontal_cortex": PrefrontalCortexLite(self.event_bus, self.gemini)
        }

        # Initialize all regions
        for name, region in self.regions.items():
            logger.info("initializing_region", region=name)
            await region.initialize()

        # Start event bus processing
        asyncio.create_task(self.event_bus.process_events())

        self.initialized = True
        logger.info("lightweight_cognitive_brain_initialized")

    async def process_text(self, text: str) -> str:
        """Process text input through the cognitive system"""

        if not self.initialized:
            await self.initialize()

        logger.info("processing_text", text_length=len(text))

        # Create response future
        response_future = asyncio.Future()

        # Handler for task completion
        async def on_task_complete(data):
            response_future.set_result(data)

        # Subscribe to completion
        self.event_bus.subscribe("task_complete", on_task_complete)

        try:
            # Start processing pipeline
            # 1. Language comprehension
            await self.event_bus.emit("input_received", {"text": text})

            # 2. Trigger executive processing
            await self.event_bus.emit("new_task", {
                "task": text,
                "context": {
                    "source": "direct_input"
                }
            })

            # Wait for completion
            result = await asyncio.wait_for(response_future, timeout=settings.reasoning_timeout)

            # Extract final output
            if result.get("success") and result.get("final_output"):
                return result["final_output"]
            else:
                return "I encountered an issue processing your request. Please try rephrasing."

        except asyncio.TimeoutError:
            logger.error("processing_timeout", text=text[:100])
            return "Processing took too long. Please try a simpler question."

        except Exception as e:
            logger.error("processing_error", error=str(e))
            return f"An error occurred: {str(e)}"

        finally:
            # Cleanup
            self.event_bus.unsubscribe("task_complete", on_task_complete)

    async def shutdown(self):
        """Shutdown the cognitive system"""

        logger.info("shutting_down_lightweight_cognitive_brain")

        # Stop event bus
        self.event_bus.stop()

        # Any other cleanup
        await asyncio.sleep(0.5)  # Allow pending events to complete


async def main():
    """Main entry point"""

    # Create brain
    brain = CognitiveTextBrainLite()

    try:
        # Initialize
        await brain.initialize()

        # Interactive mode or single query
        if len(sys.argv) > 1:
            # Process command line argument
            query = " ".join(sys.argv[1:])
            print(f"\nProcessing: {query}\n")

            response = await brain.process_text(query)
            print(f"Response: {response}\n")
        else:
            # Interactive mode
            print("\nLightweight Cognitive Text Brain")
            print("=" * 60)
            print("Using ONLY Gemini API - No heavy ML libraries!")
            print("=" * 60)
            print("\nAvailable brain regions:")
            print("- Language (Comprehension, Production, Semantic)")
            print("- Memory (Working, Hippocampus)")
            print("- Reasoning (Logical, Causal, Analogical, Creative)")
            print("- Integration (Global Workspace)")
            print("- Executive (Prefrontal Cortex)")
            print("=" * 60)
            print("\nType 'exit' to quit, 'help' for examples\n")

            while True:
                try:
                    query = input("You: ").strip()

                    if query.lower() == 'exit':
                        break
                    elif query.lower() == 'help':
                        print_help()
                        continue
                    elif not query:
                        continue

                    print("\nThinking...\n")
                    response = await brain.process_text(query)
                    print(f"Brain: {response}\n")

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"\nError: {e}\n")

    finally:
        # Cleanup
        await brain.shutdown()


def print_help():
    """Print help information"""
    print("""
    Examples of queries you can try:

    1. Logical reasoning:
       - "If all birds can fly and penguins are birds, can penguins fly?"
       - "What follows from: All humans are mortal. Socrates is human."

    2. Complex reasoning:
       - "What would happen if gravity was twice as strong?"
       - "Explain the paradox: This statement is false."

    3. Creative problems:
       - "How would society change if humans could photosynthesize?"
       - "Design a new sport that combines chess and basketball."

    4. Analytical tasks:
       - "Compare and contrast democracy and republic."
       - "What are the implications of quantum computing for cryptography?"
    """)


if __name__ == "__main__":
    asyncio.run(main())
