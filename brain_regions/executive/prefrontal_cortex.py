from typing import Dict, List, Any, Optional
import asyncio
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class PrefrontalCortex(BrainRegion):
    """Executive control and orchestration"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.goal_stack = []
        self.current_plan = None
        self.strategy_history = []
        self.monitoring_active = False

    async def initialize(self):
        """Initialize prefrontal cortex"""
        logger.info("initializing_prefrontal_cortex")

        # Subscribe to global workspace broadcasts
        self.event_bus.subscribe("global_workspace_broadcast", self._on_global_broadcast)

        # Subscribe to task inputs
        self.event_bus.subscribe("new_task", self._on_new_task)

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process executive control request"""

        task = input_data.get("task", "")
        context = input_data.get("context", {})

        # Generate executive plan
        plan = await self._generate_plan(task, context)
        self.current_plan = plan

        # Execute plan
        result = await self._execute_plan(plan)

        return result

    async def _generate_plan(self, task: str, context: Dict) -> Dict:
        """Generate executive plan for task"""

        prompt = f"""As the prefrontal cortex executive controller, create a detailed plan:

        Task: {task}
        
        Current Context:
        - Working Memory: {context.get('working_memory', 'Empty')}
        - Available Knowledge: {context.get('knowledge', 'None')}
        - Recent History: {self.strategy_history[-3:] if self.strategy_history else 'None'}
        
        Generate a plan with:
        1. goal_decomposition: Break down into sub-goals
        2. strategy_selection: Choose approach for each sub-goal
        3. resource_allocation: What brain regions to engage
        4. success_criteria: How to measure completion
        5. contingency_plans: Fallback strategies
        
        Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "main_goal": "string",
                "sub_goals": [
                    {
                        "goal": "string",
                        "strategy": "string",
                        "required_regions": ["string"],
                        "priority": "int"
                    }
                ],
                "resource_allocation": {
                    "primary_regions": ["string"],
                    "support_regions": ["string"]
                },
                "success_criteria": ["string"],
                "contingency_plans": ["string"],
                "estimated_steps": "int"
            }
        )

        if response["success"] and response["parsed"]:
            plan = response["parsed"]

            # Add to goal stack
            for sub_goal in plan["sub_goals"]:
                self.goal_stack.append(sub_goal)

            return plan

        # Fallback plan
        return {
            "main_goal": task,
            "sub_goals": [{"goal": task, "strategy": "direct", "required_regions": ["reasoning"], "priority": 1}],
            "resource_allocation": {"primary_regions": ["reasoning"], "support_regions": ["memory"]},
            "success_criteria": ["Task completed"],
            "contingency_plans": ["Retry with different approach"],
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

        # Execute sub-goals in priority order
        sorted_goals = sorted(plan["sub_goals"], key=lambda g: g["priority"])

        for sub_goal in sorted_goals:
            logger.info("executing_subgoal", goal=sub_goal["goal"])

            # Route to appropriate brain regions
            result = await self._route_to_regions(sub_goal)

            results["execution_results"].append({
                "sub_goal": sub_goal["goal"],
                "result": result,
                "success": result.get("success", False)
            })

            # Check if we should continue or adapt
            if not result.get("success", False):
                # Try contingency
                contingency_result = await self._execute_contingency(sub_goal, plan)
                if contingency_result:
                    results["execution_results"].append(contingency_result)

        # Synthesize final result
        results["final_output"] = await self._synthesize_results(results["execution_results"])
        results["success"] = self._evaluate_success(results, plan["success_criteria"])

        # Update strategy history
        self.strategy_history.append({
            "plan": plan["main_goal"],
            "success": results["success"],
            "strategies_used": [g["strategy"] for g in plan["sub_goals"]]
        })

        return results

    async def _route_to_regions(self, sub_goal: Dict) -> Dict:
        """Route sub-goal to appropriate brain regions"""

        regions = sub_goal["required_regions"]
        strategy = sub_goal["strategy"]

        # Map strategies to events
        strategy_events = {
            "logical": ("reasoning_request", {"problem": sub_goal["goal"], "type": "logical"}),
            "creative": ("creative_reasoning_request", {"problem": sub_goal["goal"]}),
            "memory": ("memory_query", {"query": sub_goal["goal"]}),
            "linguistic": ("language_analysis_request", {"text": sub_goal["goal"]}),
            "direct": ("reasoning_request", {"problem": sub_goal["goal"], "type": "general"})
        }

        if strategy in strategy_events:
            event_name, event_data = strategy_events[strategy]

            # Create a future to wait for response
            response_future = asyncio.Future()

            # Temporary handler to capture response
            async def response_handler(data):
                response_future.set_result(data)

            # Subscribe to response event
            response_event = f"{event_name}_complete"
            self.event_bus.subscribe(response_event, response_handler)

            # Emit request
            await self.event_bus.emit(event_name, event_data)

            try:
                # Wait for response with timeout (increased for complex reasoning)
                result = await asyncio.wait_for(response_future, timeout=60.0)
                return result
            except asyncio.TimeoutError:
                return {"success": False, "error": "Timeout waiting for region response"}
            finally:
                # Cleanup
                self.event_bus.unsubscribe(response_event, response_handler)

        return {"success": False, "error": f"Unknown strategy: {strategy}"}

    async def _monitoring_loop(self):
        """Monitor execution and adapt strategies"""

        self.monitoring_active = True

        while self.monitoring_active:
            try:
                if self.current_plan and self.goal_stack:
                    # Check progress
                    progress = await self._assess_progress()

                    if progress["needs_adaptation"]:
                        await self._adapt_strategy(progress)

                await asyncio.sleep(1.0)  # Monitor every second

            except Exception as e:
                logger.error("monitoring_error", error=str(e))

    async def _assess_progress(self) -> Dict:
        """Assess progress on current goals"""

        # Simple progress assessment
        completed_goals = len([h for h in self.strategy_history if h.get("success", False)])
        total_goals = len(self.strategy_history)

        if total_goals > 0:
            success_rate = completed_goals / total_goals
            needs_adaptation = success_rate < 0.5 and total_goals > 3
        else:
            success_rate = 0
            needs_adaptation = False

        return {
            "success_rate": success_rate,
            "needs_adaptation": needs_adaptation,
            "active_goals": len(self.goal_stack),
            "completed_goals": completed_goals
        }

    async def _adapt_strategy(self, progress: Dict):
        """Adapt strategy based on progress"""

        prompt = f"""The executive controller needs to adapt its strategy.

                Current Progress:
                - Success Rate: {progress['success_rate']:.2%}
                - Active Goals: {progress['active_goals']}
                - Recent Failures: {[h for h in self.strategy_history[-3:] if not h.get('success', False)]}

                Suggest adaptations:
                1. Alternative strategies to try
                2. Whether to simplify or decompose goals differently
                3. Which brain regions to engage more/less

                Be specific and actionable."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            # Log adaptation suggestion
            logger.info("strategy_adaptation", suggestion=response["text"][:200])

            # Could implement automatic adaptation here
            await self.event_bus.emit("strategy_adaptation_suggested", {
                "suggestion": response["text"],
                "progress": progress
            })

    async def _execute_contingency(self, failed_goal: Dict, plan: Dict) -> Optional[Dict]:
        """Execute contingency plan for failed goal"""

        if plan["contingency_plans"]:
            contingency = plan["contingency_plans"][0]

            logger.info("executing_contingency",
                        failed_goal=failed_goal["goal"],
                        contingency=contingency)

            # Modify goal with contingency strategy
            modified_goal = {
                **failed_goal,
                "strategy": "creative",  # Try creative approach
                "goal": f"{failed_goal['goal']} (contingency: {contingency})"
            }

            result = await self._route_to_regions(modified_goal)

            return {
                "sub_goal": modified_goal["goal"],
                "result": result,
                "success": result.get("success", False),
                "is_contingency": True
            }

        return None

    async def _synthesize_results(self, execution_results: List[Dict]) -> str:
        """Synthesize execution results into final output"""

        successful_results = [r for r in execution_results if r.get("success", False)]

        if not successful_results:
            return "Unable to complete task successfully."

        # Use Gemini to synthesize
        prompt = f"""Synthesize these execution results into a coherent final answer:

                Results:
                {self._format_results(successful_results)}

                Create a clear, comprehensive response that integrates all successful results."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        return response["text"] if response["success"] else "Results synthesized."

    def _evaluate_success(self, results: Dict, criteria: List[str]) -> bool:
        """Evaluate if execution met success criteria"""

        # Simple evaluation - check if any results and no critical failures
        has_results = len(results["execution_results"]) > 0
        has_success = any(r.get("success", False) for r in results["execution_results"])

        return has_results and has_success

    def _format_results(self, results: List[Dict]) -> str:
        """Format results for prompts"""

        formatted = []
        for r in results:
            formatted.append(f"Goal: {r['sub_goal']}\nResult: {r.get('result', {}).get('conclusion', 'Completed')}")

        return "\n\n".join(formatted)

    async def _on_global_broadcast(self, data: Dict):
        """Handle global workspace broadcasts"""

        integrated_state = data.get("integrated_state", {})

        # Check if action is needed
        if integrated_state.get("action_implications"):
            for action in integrated_state["action_implications"]:
                # Add to goal stack if not already present
                if not any(g["goal"] == action for g in self.goal_stack):
                    self.goal_stack.append({
                        "goal": action,
                        "strategy": "direct",
                        "required_regions": ["reasoning"],
                        "priority": 2
                    })

    async def _on_new_task(self, data: Dict):
        """Handle new task request"""

        task = data.get("task", "")
        context = data.get("context", {})

        # Process the task
        result = await self.process({"task": task, "context": context})

        # Emit completion
        await self.event_bus.emit("task_complete", result)

    def get_state(self) -> Dict[str, Any]:
        return {
            "active_goals": len(self.goal_stack),
            "current_plan": self.current_plan["main_goal"] if self.current_plan else None,
            "strategy_history_length": len(self.strategy_history),
            "recent_success_rate": self._calculate_recent_success_rate(),
            "monitoring_active": self.monitoring_active
        }

    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate of recent strategies"""

        recent = self.strategy_history[-10:]
        if not recent:
            return 0.0

        successes = sum(1 for h in recent if h.get("success", False))
        return successes / len(recent)