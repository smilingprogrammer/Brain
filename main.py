import asyncio
import sys
from typing import Optional
import structlog
from dotenv import load_dotenv

from config.settings import settings
from core.event_bus import EventBus
from core.logging import setup_logging

# Import brain regions
from brain_regions.gemini.gemini_service import GeminiService
from brain_regions.language.comprehension import LanguageComprehension
from brain_regions.memory.working_memory import WorkingMemory
from brain_regions.reasoning.logical_reasoning import LogicalReasoning
from brain_regions.integration.global_workspace import GlobalWorkspace
from brain_regions.executive.prefrontal_cortex import PrefrontalCortex

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(settings.log_level)
logger = structlog.get_logger()


class CognitiveTextBrain:
    """Main cognitive brain system for text reasoning"""

    def __init__(self):
        self.event_bus = EventBus()
        self.gemini = GeminiService()
        self.regions = {}
        self.initialized = False

    async def initialize(self):
        """Initialize all brain regions"""

        logger.info("initializing_cognitive_brain")

        # Create brain regions
        self.regions = {
            "language": LanguageComprehension(self.event_bus),
            "working_memory": WorkingMemory(self.event_bus, self.gemini),
            "logical_reasoning": LogicalReasoning(self.event_bus, self.gemini),
            "global_workspace": GlobalWorkspace(self.event_bus),
            "prefrontal_cortex": PrefrontalCortex(self.event_bus, self.gemini)
        }

        # Initialize all regions
        for name, region in self.regions.items():
            logger.info("initializing_region", region=name)
            await region.initialize()

        # Start event bus processing
        asyncio.create_task(self.event_bus.process_events())

        self.initialized = True
        logger.info("cognitive_brain_initialized")

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

        logger.info("shutting_down_cognitive_brain")

        # Stop event bus
        self.event_bus.stop()

        # Any other cleanup
        await asyncio.sleep(0.5)  # Allow pending events to complete


async def main():
    """Main entry point"""

    # Create brain
    brain = CognitiveTextBrain()

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
            print("\n🧠 Cognitive Text Brain - Interactive Mode")
            print("Type 'exit' to quit, 'help' for examples\n")

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


# Expected Output
# Handler received: {'message': 'Hello'}
# Handler received: {'message': 'World'}
# Received 2 events
# Data: [{'message': 'Hello'}, {'message': 'World'}]
# Final count: 2 (should be 2)