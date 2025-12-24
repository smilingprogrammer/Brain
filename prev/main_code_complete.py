import asyncio
import time
from typing import Dict, Any, Optional, List
import structlog
from dotenv import load_dotenv
import re

from config.settings import settings
from core.event_bus import EventBus
from core.logging import setup_logging, cognitive_logger
from core.metrics import metrics_collector

# Import ALL brain regions
from brain_regions.gemini.gemini_service import GeminiService
from brain_regions.gemini.knowledge_base import KnowledgeBase
from brain_regions.gemini.reasoning_engine import ReasoningEngine

# Language regions
from brain_regions.language.comprehension import LanguageComprehension
from brain_regions.language.production import LanguageProduction
from brain_regions.language.semantic_decoder import SemanticDecoder

# Memory regions
from brain_regions.memory.working_memory import WorkingMemory
from brain_regions.memory.hippocampus import Hippocampus
from brain_regions.memory.semantic_cortex import SemanticCortex
from brain_regions.memory.memory_consolidation import MemoryConsolidation

# Executive regions
from brain_regions.executive.prefrontal_cortex import PrefrontalCortex
from brain_regions.executive.attention import AttentionController
from brain_regions.executive.meta_cognition import MetaCognition

# Reasoning regions
from brain_regions.reasoning.logical_reasoning import LogicalReasoning
from brain_regions.reasoning.analogical_reasoning import AnalogicalReasoning
from brain_regions.reasoning.causal_reasoning import CausalReasoning
from brain_regions.reasoning.creative_reasoning import CreativeReasoning

# Integration regions
from brain_regions.integration.global_workspace import GlobalWorkspace
from brain_regions.integration.thalamus import Thalamus
from brain_regions.integration.corpus_callosum import CorpusCallosum

# Coding regions (NEW)
from brain_regions.coding.syntax_validator import SyntaxValidator
from brain_regions.coding.code_review import CodeReviewModule

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(settings.log_level)
logger = structlog.get_logger()


class CompleteCognitiveTextBrain:
    """Complete implementation with all brain regions properly connected"""

    def __init__(self):
        self.event_bus = EventBus()
        self.gemini = GeminiService()
        self.regions = {}
        self.initialized = False

    async def initialize(self):
        """Initialize all brain regions in proper order"""

        logger.info("initializing_complete_cognitive_brain")

        # Initialize Gemini services first
        self.knowledge_base = KnowledgeBase(self.gemini, self.event_bus)
        self.reasoning_engine = ReasoningEngine(self.gemini, self.event_bus)

        # Initialize brain regions
        self.regions = {
            # Language processing
            "language_comprehension": LanguageComprehension(self.event_bus),
            "language_production": LanguageProduction(self.event_bus, self.gemini),
            "semantic_decoder": SemanticDecoder(self.event_bus),

            # Memory systems
            "working_memory": WorkingMemory(self.event_bus, self.gemini),
            "hippocampus": Hippocampus(self.event_bus, self.gemini),
            "semantic_cortex": SemanticCortex(self.event_bus, self.gemini),
            "memory_consolidation": MemoryConsolidation(self.event_bus),

            # Executive control
            "prefrontal_cortex": PrefrontalCortex(self.event_bus, self.gemini),
            "attention_controller": AttentionController(self.event_bus),
            "meta_cognition": MetaCognition(self.event_bus, self.gemini),

            # Reasoning modules
            "logical_reasoning": LogicalReasoning(self.event_bus, self.gemini),
            "analogical_reasoning": AnalogicalReasoning(self.event_bus, self.gemini),
            "causal_reasoning": CausalReasoning(self.event_bus, self.gemini),
            "creative_reasoning": CreativeReasoning(self.event_bus, self.gemini),

            # Integration systems
            "global_workspace": GlobalWorkspace(self.event_bus),
            "thalamus": Thalamus(self.event_bus),
            "corpus_callosum": CorpusCallosum(self.event_bus),

            # Coding modules (NEW)
            "syntax_validator": SyntaxValidator(self.event_bus),
            "code_review": CodeReviewModule(
                self.event_bus,
                self.gemini,
                None  # Will be set after meta_cognition is initialized
            )
        }

        # Set meta_cognition reference for code_review
        self.regions["code_review"].meta_cognition = self.regions["meta_cognition"]

        # Initialize all regions
        for name, region in self.regions.items():
            logger.info("initializing_region", region=name)
            await region.initialize()

        # Initialize Gemini services
        await self.knowledge_base.initialize()
        await self.reasoning_engine.initialize()

        # Set up critical event connections
        self._setup_event_connections()

        # Start event bus processing
        asyncio.create_task(self.event_bus.process_events())

        # Start background processes
        asyncio.create_task(self._background_processes())

        self.initialized = True
        logger.info("complete_cognitive_brain_initialized")

    def _setup_event_connections(self):
        """Set up critical event connections between regions"""

        # Language → Working Memory flow
        self.event_bus.subscribe(
            "language_comprehension_complete",
            self._route_to_working_memory
        )

        # Working Memory → Executive flow
        self.event_bus.subscribe(
            "working_memory_updated",
            self._trigger_executive_processing
        )

        # Executive → Reasoning flow
        self.event_bus.subscribe(
            "action_planned",
            self._route_to_reasoning
        )

        # Reasoning → Global Workspace flow
        self.event_bus.subscribe(
            "reasoning_complete",
            self._send_to_global_workspace
        )

        # Global Workspace → Production flow
        self.event_bus.subscribe(
            "global_workspace_broadcast",
            self._trigger_response_generation
        )

        # Memory consolidation triggers
        self.event_bus.subscribe(
            "high_memory_pressure",
            self._trigger_consolidation
        )

    async def process_text(self, text: str) -> str:
        """Process text through complete cognitive pipeline"""

        if not self.initialized:
            await self.initialize()

        start_time = time.time()
        logger.info("processing_text_input", text_length=len(text))

        # Create processing context
        context = {
            "input_text": text,
            "timestamp": start_time,
            "processing_id": f"proc_{int(start_time * 1000)}"
        }

        # Step 1: Language Comprehension
        language_result = await self._process_language_input(text, context)

        # Step 2: Store in Working Memory
        await self._store_in_working_memory(language_result, context)

        # Step 3: Executive Planning
        plan = await self._executive_planning(context)

        # Step 4: Execute Reasoning (parallel)
        reasoning_results = await self._execute_reasoning(plan, context)

        # Step 5: Global Integration
        integrated_state = await self._global_integration(reasoning_results, context)

        # Step 6: Update Semantic Memory
        await self._update_semantic_memory(integrated_state, context)

        # Step 7: Generate Response
        response = await self._generate_response(integrated_state, context)

        # Step 8: Update Episodic Memory
        await self._update_episodic_memory(context, response)

        # Log metrics
        processing_time = time.time() - start_time
        cognitive_logger.log_brain_event(
            event_type="processing_complete",
            region="system",
            data={"processing_time": processing_time},
            latency_ms=processing_time * 1000
        )

        return response

    async def process_code(self, code: str, instruction: Optional[str] = None) -> Dict:
        """Process code with optional instruction"""

        if not self.initialized:
            await self.initialize()

        # Detect language
        language = self._detect_language(code)

        # Step 1: Syntax validation
        syntax_result = await self.regions["syntax_validator"].process({
            "code": code,
            "language": language,
            "mode": "validate"
        })

        if not syntax_result["valid"]:
            # Try to fix syntax errors
            fix_result = await self.regions["syntax_validator"].process({
                "code": code,
                "language": language,
                "mode": "fix"
            })

            # Return early with syntax fixes
            return {
                "type": "syntax_fix",
                "original_errors": syntax_result["errors"],
                "fixes": fix_result["fixes"],
                "suggestion": "Please fix syntax errors before proceeding"
            }

        # Step 2: Code review
        review_result = await self.regions["code_review"].process({
            "code": code,
            "language": language,
            "review_type": "comprehensive"
        })

        # Step 3: Process any instruction
        if instruction:
            # Combine code and instruction for processing
            combined_input = f"""
            Instruction: {instruction}
            
            Code:
            ```{language}
            {code}
            """
            response = await self.process_text(combined_input)
            # Step 4: Validate any generated code
            if "```" in response:
                # Extract and validate generated code
                generated_code = self._extract_code_from_response(response)
                if generated_code:
                    validation = await self.regions["syntax_validator"].process({
                        "code": generated_code,
                        "language": language,
                        "mode": "validate"
                    })

                    return {
                        "type": "code_update",
                        "instruction": instruction,
                        "original_review": review_result,
                        "updated_code": generated_code,
                        "validation": validation,
                        "explanation": response
                    }

            return {
                "type": "instruction_response",
                "instruction": instruction,
                "review": review_result,
                "response": response
            }

        # Just review if no instruction
        return {
            "type": "code_review",
            "review": review_result,
            "overall_score": review_result["overall_score"],
            "improvements": review_result["improvements"]
        }


    async def _process_language_input(self, text: str, context: Dict) -> Dict:
        """Step 1: Process text through language comprehension"""

        logger.info("step_1_language_comprehension", text_preview=text[:50])

        language = self.regions["language_comprehension"]
        result = await language.process({"text": text})

        # Enhance with semantic decoding
        semantic = self.regions["semantic_decoder"]
        semantic_result = await semantic.process({
            "text": text,
            "mode": "encode"
        })

        result["semantic_vector"] = semantic_result.get("vector", [])
        result["nearest_concepts"] = semantic_result.get("nearest_concepts", [])

        context["language_result"] = result
        return result


    async def _store_in_working_memory(self, language_result: Dict, context: Dict):
        """Step 2: Store comprehension results in working memory"""

        logger.info("step_2_working_memory_storage")

        wm = self.regions["working_memory"]

        # Calculate salience based on complexity and novelty
        salience = language_result.get("complexity_score", 0.5)
        if language_result.get("entities"):
            salience += 0.1 * len(language_result["entities"])

        await wm.store({
            "type": "language_input",
            "content": language_result,
            "context": context,
            "salience": min(salience, 1.0)
        })

        # Update attention
        attention = self.regions["attention_controller"]
        await attention.process({
            "command": "focus",
            "target": "language_processing",
            "intensity": salience
        })


    async def _executive_planning(self, context: Dict) -> Dict:
        """Step 3: Executive planning based on input"""

        logger.info("step_3_executive_planning")

        executive = self.regions["prefrontal_cortex"]

        # Get current working memory state
        wm = self.regions["working_memory"]
        wm_state = wm.get_state()

        # Generate plan
        plan = await executive.process({
            "task": context["input_text"],
            "context": {
                "working_memory_summary": wm_state,
                "language_analysis": context.get("language_result", {})
            }
        })

        context["executive_plan"] = plan
        return plan


    async def _execute_reasoning(self, plan: Dict, context: Dict) -> Dict:
        """Step 4: Execute parallel reasoning based on plan"""

        logger.info("step_4_parallel_reasoning")

        reasoning_tasks = []
        reasoning_types = []

        # Map plan strategies to reasoning modules
        strategy_mapping = {
            "logical": self.regions["logical_reasoning"],
            "causal": self.regions["causal_reasoning"],
            "creative": self.regions["creative_reasoning"],
            "analogical": self.regions["analogical_reasoning"]
        }

        # Execute sub-goals in parallel
        for sub_goal in plan.get("sub_goals", [])[:3]:  # Limit to top 3
            strategy = sub_goal.get("strategy", "logical")
            if strategy in strategy_mapping:
                reasoning_module = strategy_mapping[strategy]
                task = reasoning_module.reason(
                    sub_goal.get("goal", context["input_text"]),
                    context
                )
                reasoning_tasks.append(task)
                reasoning_types.append(strategy)

        # Wait for all reasoning to complete
        results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)

        # Package results
        reasoning_results = {}
        for strategy, result in zip(reasoning_types, results):
            if not isinstance(result, Exception):
                reasoning_results[strategy] = result
            else:
                logger.error("reasoning_failed", strategy=strategy, error=str(result))

        context["reasoning_results"] = reasoning_results
        return reasoning_results


    async def _global_integration(self, reasoning_results: Dict, context: Dict) -> Dict:
        """Step 5: Integrate results in global workspace"""

        logger.info("step_5_global_integration")

        gw = self.regions["global_workspace"]

        # Add all reasoning results to competition
        for reasoning_type, result in reasoning_results.items():
            await gw._add_to_competition(
                f"reasoning_{reasoning_type}",
                {
                    "type": reasoning_type,
                    "result": result,
                    "confidence": result.get("confidence", 0.5)
                }
            )

        # Add working memory context
        wm = self.regions["working_memory"]
        wm_focus = wm.get_focus()

        if wm_focus:
            await gw._add_to_competition("working_memory", wm_focus)

        # Wait for integration
        await asyncio.sleep(0.2)  # Allow competition to run

        # Get integrated state
        integrated_state = gw.get_state().get("current_workspace", {})

        context["integrated_state"] = integrated_state
        return integrated_state

    async def _update_semantic_memory(self, integrated_state: Dict, context: Dict):
        """Step 6: Update long-term semantic memory"""

        logger.info("step_6_semantic_memory_update")

        semantic = self.regions["semantic_cortex"]

        # Extract key insights
        if integrated_state.get("integrated_meaning"):
            await semantic.store({
                "concept": f"query_response_{context['processing_id']}",
                "meaning": integrated_state["integrated_meaning"],
                "confidence": integrated_state.get("confidence", 0.5),
                "timestamp": time.time()
            })

        # Store any new relationships discovered
        if integrated_state.get("action_implications"):
            for implication in integrated_state["action_implications"]:
                await semantic.process({
                    "operation": "relate",
                    "source": context["input_text"][:50],
                    "relation": "implies",
                    "target": implication
                })

    async def _generate_response(self, integrated_state: Dict, context: Dict) -> str:
        """Step 7: Generate natural language response"""

        logger.info("step_7_response_generation")

        production = self.regions["language_production"]

        # Prepare content for production
        content = {
            "integrated_state": integrated_state,
            "reasoning_results": context.get("reasoning_results", {}),
            "original_query": context["input_text"]
        }

        # Generate response
        response_result = await production.process({
            "intent": "respond",
            "content": content,
            "style": "clear"
        })

        if response_result["success"]:
            return response_result["text"]
        else:
            return "I encountered an issue generating a response. Please try rephrasing your question."

    async def _update_episodic_memory(self, context: Dict, response: str):
        """Step 8: Store episode in hippocampus"""

        logger.info("step_8_episodic_memory_update")

        hippocampus = self.regions["hippocampus"]

        episode = {
            "query": context["input_text"],
            "response": response,
            "reasoning_paths": list(context.get("reasoning_results", {}).keys()),
            "confidence": context.get("integrated_state", {}).get("confidence", 0.5),
            "timestamp": time.time(),
            "processing_time": time.time() - context["timestamp"]
        }

        await hippocampus.store(episode, metadata={"type": "query_response"})

        # Update metrics
        metrics_collector.update_working_memory_usage(
            len(self.regions["working_memory"].buffer),
            self.regions["working_memory"].capacity
        )

    async def _background_processes(self):
        """Run background cognitive processes"""

        while True:
            try:
                # Meta-cognitive monitoring every 5 seconds
                if hasattr(self, 'regions') and 'meta_cognition' in self.regions:
                    meta = self.regions["meta_cognition"]
                    await meta.process({"operation": "assess"})

                # Check for memory pressure
                wm = self.regions.get("working_memory")
                if wm and len(wm.buffer) / wm.capacity > 0.8:
                    await self.event_bus.emit("high_memory_pressure", {
                        "level": "high",
                        "usage": len(wm.buffer) / wm.capacity
                    })

                await asyncio.sleep(5.0)

            except Exception as e:
                logger.error("background_process_error", error=str(e))
                await asyncio.sleep(10.0)

    # Event handlers for proper flow
    async def _route_to_working_memory(self, data: Dict):
        """Route language comprehension to working memory"""
        wm = self.regions["working_memory"]
        await wm.store({
            "type": "language_comprehension",
            "content": data,
            "salience": data.get("complexity_score", 0.5)
        })

    async def _trigger_executive_processing(self, data: Dict):
        """Trigger executive processing on working memory updates"""
        # Only process if significant update
        if data.get("new_item", {}).get("salience", 0) > 0.6:
            executive = self.regions["prefrontal_cortex"]
            asyncio.create_task(executive.process({
                "operation": "evaluate",
                "working_memory_update": data
            }))

    async def _route_to_reasoning(self, data: Dict):
        """Route executive plans to appropriate reasoning modules"""
        plan = data.get("plan", {})
        for sub_goal in plan.get("sub_goals", []):
            strategy = sub_goal.get("strategy")

            # Route to appropriate reasoning module
            if strategy == "logical":
                await self.event_bus.emit("logical_reasoning_request", sub_goal)
            elif strategy == "causal":
                await self.event_bus.emit("causal_reasoning_request", sub_goal)
            elif strategy == "creative":
                await self.event_bus.emit("creative_reasoning_request", sub_goal)
            elif strategy == "analogical":
                await self.event_bus.emit("analogical_reasoning_request", sub_goal)

    async def _send_to_global_workspace(self, data: Dict):
        """Send reasoning results to global workspace"""
        gw = self.regions["global_workspace"]
        reasoning_type = data.get("type", "unknown")

        await gw._add_to_competition(
            f"reasoning_{reasoning_type}",
            data
        )

    async def _trigger_response_generation(self, data: Dict):
        """Trigger response generation from global workspace state"""
        integrated_state = data.get("integrated_state", {})

        # Only generate response if we have high confidence integration
        if integrated_state.get("confidence", 0) > 0.6:
            production = self.regions["language_production"]
            asyncio.create_task(production.process({
                "intent": "explain",
                "content": integrated_state,
                "style": "clear"
            }))

    async def _trigger_consolidation(self, data: Dict):
        """Trigger memory consolidation on high pressure"""
        consolidation = self.regions["memory_consolidation"]
        await consolidation.process({"command": "consolidate_now"})

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""

        # Simple heuristics
        if "def " in code and "import " in code:
            return "python"
        elif "function" in code or "const " in code or "=>" in code:
            return "javascript"
        elif "public class" in code or "public static void" in code:
            return "java"
        elif "#include" in code or "int main(" in code:
            return "cpp"
        else:
            return "python"  # default

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code block from response"""

        code_pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()
        return None

    async def shutdown(self):
        """Gracefully shutdown the cognitive system"""

        logger.info("shutting_down_complete_cognitive_brain")

        # Stop background processes
        self.event_bus.stop()

        # Save any pending memories
        if "hippocampus" in self.regions:
            await self.regions["hippocampus"].consolidate()

        # Export cognitive logs
        cognitive_logger.export_session_log(f"session_{int(time.time())}.json")

        await asyncio.sleep(0.5)  # Allow pending events to complete

        logger.info("shutdown_complete")


# Main execution
async def main():
    """Main entry point with complete brain"""

    import sys

    # Create complete brain
    brain = CompleteCognitiveTextBrain()

    try:
        # Initialize
        await brain.initialize()

        # Interactive mode or single query
        if len(sys.argv) > 1:
            # Process command line argument
            query = " ".join(sys.argv[1:])
            print(f"\n🧠 Processing: {query}\n")

            response = await brain.process_text(query)
            print(f"💭 Response: {response}\n")
        else:
            # Interactive mode
            print("\n🧠 Complete Cognitive Text Brain - Interactive Mode")
            print("=" * 60)
            print("This implementation includes ALL brain regions:")
            print("- Language (Comprehension, Production, Semantic)")
            print("- Memory (Working, Hippocampus, Semantic, Consolidation)")
            print("- Executive (Prefrontal, Attention, Meta-cognition)")
            print("- Reasoning (Logical, Causal, Analogical, Creative)")
            print("- Integration (Global Workspace, Thalamus, Corpus Callosum)")
            print("- Coding (Syntax Validation, Code Review)")
            print("=" * 60)
            print("\nCommands:")
            print("- 'code:' prefix to process code")
            print("- 'status' to show brain state")
            print("- 'help' for examples")
            print("- 'exit' to quit\n")

            while True:
                try:
                    query = input("You: ").strip()

                    if query.lower() == 'exit':
                        break
                    elif query.lower() == 'help':
                        print_help()
                        continue
                    elif query.lower() == 'status':
                        print_status(brain)
                        continue
                    elif query.lower().startswith('code:'):
                        # Handle code input
                        code_input = query[5:].strip()
                        if '\n' in code_input:
                            # Multi-line code
                            code, instruction = code_input.split('\n', 1)
                            result = await brain.process_code(code, instruction)
                        else:
                            # Just code review
                            result = await brain.process_code(code_input)
                        print(f"\nCode Analysis:\n{format_code_result(result)}\n")
                        continue
                    elif not query:
                        continue

                    print("\n🤔 Thinking...\n")
                    response = await brain.process_text(query)
                    print(f"Brain: {response}\n")

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    logger.error("processing_error", error=str(e), exc_info=True)

    finally:
        # Cleanup
        await brain.shutdown()


def print_help():
    """Print help information"""
    print("""
    🧠 Complete Cognitive Brain - Help
    
    This system processes text through a full cognitive pipeline:
    1. Language Comprehension (tokenization, entities, embeddings)
    2. Working Memory (context maintenance with compression)
    3. Executive Planning (goal decomposition and strategy)
    4. Parallel Reasoning (logical, causal, creative, analogical)
    5. Global Workspace (consciousness-like integration)
    6. Semantic Memory (knowledge storage and retrieval)
    7. Language Production (natural response generation)
    8. Episodic Memory (experience storage)
    
    Example queries:
    - "What would happen if gravity was twice as strong?"
    - "Explain the relationship between consciousness and attention"
    - "Design a city for people who can fly"
    - "If all insects disappeared, what would be the consequences?"
    - "How is the human brain similar to a computer network?"
    
    Code processing:
    - code: def add(a, b): return a + b
    - code: [paste code]\\n[instruction]
    
    Special commands:
    - status: Show current brain state
    - help: Show this help
    - exit: Quit the program
    """)


def print_status(brain):
    """Print current brain status"""
    print("\n🧠 Brain Status:")
    print("=" * 40)

    for name, region in brain.regions.items():
        state = region.get_state()
        print(f"\n{name.upper()}:")

        # Show key metrics for each region type
        if "working_memory" in name:
            print(f"  Capacity: {state.get('buffer_size', 0)}/{state.get('capacity_usage', 0) * 7:.0f}")
            print(f"  Compressed: {state.get('compressed_count', 0)} items")
        elif "global_workspace" in name:
            print(f"  Competition size: {state.get('competition_size', 0)}")
            print(f"  History: {state.get('history_length', 0)} integrations")
        elif "prefrontal" in name:
            print(f"  Active goals: {state.get('active_goals', 0)}")
            print(f"  Success rate: {state.get('recent_success_rate', 0):.1%}")
        elif "hippocampus" in name:
            print(f"  Episodes: {state.get('episode_count', 0)}")
            print(f"  Replays: {state.get('replay_count', 0)}")
        elif "attention" in name:
            print(f"  Mode: {state.get('attention_mode', 'N/A')}")
            print(f"  Allocated: {state.get('total_allocated', 0):.2f}")
        else:
            # Show first two state items for other regions
            items = list(state.items())[:2]
            for key, value in items:
                print(f"  {key}: {value}")


def format_code_result(result: Dict) -> str:
    """Format code processing result for display"""

    if result["type"] == "syntax_fix":
        output = "Syntax Errors Found:\n"
        for error in result["original_errors"]:
            output += f"  Line {error['line']}: {error['message']}\n"
        output += "\nSuggested Fixes:\n"
        for fix in result["fixes"]:
            output += f"  - {fix['suggestion']}\n"

    elif result["type"] == "code_review":
        output = f"Code Review Score: {result['overall_score']:.1f}/100\n\n"
        review = result["review"]

        if review.get("issues"):
            output += "Issues Found:\n"
            for issue in review["issues"][:5]:
                output += f"  [{issue['severity']}] {issue['message']}\n"

        if result.get("improvements"):
            output += "\nTop Improvements:\n"
            for imp in result["improvements"][:3]:
                output += f"  - {imp['suggestion']}\n"

    elif result["type"] == "code_update":
        output = f"Code Updated Successfully!\n"
        output += f"Instruction: {result['instruction']}\n"
        output += f"Validation: {'✓ Valid' if result['validation']['valid'] else '✗ Invalid'}\n"
        output += f"\nUpdated Code:\n{result['updated_code']}\n"

    else:
        output = str(result)

    return output

if __name__ == "__main__":
    asyncio.run(main())