from typing import Dict, List, Any, Optional
import asyncio
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class PrefrontalCortexLite(BrainRegion):
    """Lightweight executive control with simplified planning"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.goal_stack = []
        self.current_plan = None
        self.strategy_history = []

    async def initialize(self):
        """Initialize prefrontal cortex"""
        logger.info("initializing_prefrontal_cortex_lite")

        # Subscribe to global workspace broadcasts
        self.event_bus.subscribe("global_workspace_broadcast", self._on_global_broadcast)

        # Subscribe to task inputs
        self.event_bus.subscribe("new_task", self._on_new_task)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process executive control request"""

        task = input_data.get("task", "")
        context = input_data.get("context", {})

        # Simplified plan generation - prefer direct strategy
        plan = self._create_simple_plan(task)

        self.current_plan = plan

        # Execute plan
        result = await self._execute_plan(plan)

        return result

    def _create_simple_plan(self, task: str) -> Dict:
        """Create a simple, fast execution plan"""

        # Use direct strategy by default for speed
        return {
            "main_goal": task,
            "sub_goals": [
                {
                    "goal": task,
                    "strategy": "direct",  # Prefer direct/fast approach
                    "required_regions": ["reasoning"],
                    "priority": 1
                }
            ],
            "resource_allocation": {
                "primary_regions": ["reasoning"],
                "support_regions": ["memory"]
            },
            "success_criteria": ["Task addressed"],
            "contingency_plans": ["Use creative reasoning if direct fails"],
            "estimated_steps": 1
        }

    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute the generated plan"""

        results = {
            "plan": plan,
            "execution_results": [],
            "success": False,
            "final_output": None
        }

        # Execute sub-goals
        for sub_goal in plan["sub_goals"]:
            logger.info("executing_subgoal_lite", goal=sub_goal["goal"])

            # Route to appropriate brain regions
            result = await self._route_to_regions(sub_goal)

            results["execution_results"].append({
                "sub_goal": sub_goal["goal"],
                "result": result,
                "success": result.get("success", False)
            })

            # If direct fails, try creative as fallback
            if not result.get("success", False):
                logger.info("direct_failed_trying_creative", goal=sub_goal["goal"])

                creative_goal = {
                    **sub_goal,
                    "strategy": "creative"
                }

                creative_result = await self._route_to_regions(creative_goal)

                results["execution_results"].append({
                    "sub_goal": sub_goal["goal"] + " (creative fallback)",
                    "result": creative_result,
                    "success": creative_result.get("success", False),
                    "is_fallback": True
                })

        # Synthesize final result
        results["final_output"] = await self._synthesize_results(results["execution_results"])
        results["success"] = self._evaluate_success(results)

        # Update strategy history
        self.strategy_history.append({
            "plan": plan["main_goal"],
            "success": results["success"]
        })

        return results

    async def _route_to_regions(self, sub_goal: Dict) -> Dict:
        """Route sub-goal to appropriate brain regions"""

        strategy = sub_goal["strategy"]

        # Map strategies to events (prefer faster ones)
        strategy_events = {
            "direct": ("reasoning_request", {"problem": sub_goal["goal"], "type": "general"}),
            "logical": ("reasoning_request", {"problem": sub_goal["goal"], "type": "logical"}),
            "creative": ("creative_reasoning_request", {"problem": sub_goal["goal"]}),
        }

        if strategy not in strategy_events:
            strategy = "direct"  # Fallback to direct

        event_name, event_data = strategy_events[strategy]

        # Create a future to wait for response
        response_future = asyncio.Future()

        # Temporary handler to capture response
        async def response_handler(data):
            if not response_future.done():
                response_future.set_result(data)

        # Subscribe to response event
        response_event = f"{event_name}_complete"
        self.event_bus.subscribe(response_event, response_handler)

        try:
            # Emit request
            await self.event_bus.emit(event_name, event_data)

            # Wait for response with timeout
            result = await asyncio.wait_for(response_future, timeout=30.0)
            return result

        except asyncio.TimeoutError:
            logger.warning("region_response_timeout", strategy=strategy, goal=sub_goal["goal"][:50])
            return {
                "success": False,
                "error": f"Timeout waiting for {strategy} response",
                "conclusion": "Unable to complete reasoning in time"
            }

        finally:
            # Cleanup
            self.event_bus.unsubscribe(response_event, response_handler)

    async def _synthesize_results(self, execution_results: List[Dict]) -> str:
        """Synthesize execution results into final output"""

        successful_results = [r for r in execution_results if r.get("success", False)]

        if not successful_results:
            return "Unable to generate a complete response. Please try rephrasing your question."

        # Extract conclusions from successful results
        conclusions = []
        for r in successful_results:
            result_data = r.get("result", {})

            # Try different keys where conclusion might be
            if "conclusion" in result_data:
                conclusions.append(result_data["conclusion"])
            elif "text" in result_data:
                conclusions.append(result_data["text"])
            elif "solutions" in result_data and result_data["solutions"]:
                # Creative reasoning returns solutions
                for sol in result_data["solutions"][:2]:
                    if isinstance(sol, dict) and "idea" in sol:
                        conclusions.append(sol["idea"])

        if conclusions:
            # Return the best/first conclusion
            return conclusions[0] if len(conclusions) == 1 else "\n\n".join(conclusions[:2])

        return "Task processed successfully."

    def _evaluate_success(self, results: Dict) -> bool:
        """Evaluate if execution met success criteria"""

        has_results = len(results["execution_results"]) > 0
        has_success = any(r.get("success", False) for r in results["execution_results"])

        return has_results and has_success

    async def _on_global_broadcast(self, data: Dict):
        """Handle global workspace broadcasts"""
        # Simplified - just log for now
        logger.debug("global_broadcast_received", data_keys=list(data.keys()))

    async def _on_new_task(self, data: Dict):
        """Handle new task request - launch as separate task to avoid blocking event bus"""

        async def process_task():
            task = data.get("task", "")
            context = data.get("context", {})

            # Process the task
            result = await self.process({"task": task, "context": context})

            # Emit completion
            await self.event_bus.emit("task_complete", result)

        # Launch as separate task so event bus can continue processing events
        asyncio.create_task(process_task())

    def get_state(self) -> Dict[str, Any]:
        return {
            "active_goals": len(self.goal_stack),
            "current_plan": self.current_plan["main_goal"] if self.current_plan else None,
            "strategy_history_length": len(self.strategy_history),
            "mode": "lite"
        }
