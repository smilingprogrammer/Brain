from typing import Dict, Any, List, Optional
import time
from collections import deque, defaultdict
import numpy as np
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class MetaCognition(BrainRegion):
    """Self-monitoring and meta-cognitive processes"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini

        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.error_patterns = defaultdict(int)
        self.success_strategies = defaultdict(float)

        # Self-model
        self.cognitive_state = {
            "confidence": 0.5,
            "cognitive_load": 0.5,
            "fatigue": 0.0,
            "clarity": 0.5
        }

        # Meta-cognitive strategies
        self.active_strategies = set()
        self.strategy_effectiveness = {}

        # Monitoring metrics
        self.metrics = {
            "response_times": deque(maxlen=50),
            "error_rate": 0.0,
            "learning_rate": 0.0
        }

        self.state = {}

    async def initialize(self):
        """Initialize meta-cognition"""
        logger.info("initializing_meta_cognition")

        # Subscribe to cognitive events
        self.event_bus.subscribe("task_complete", self._on_task_complete)
        self.event_bus.subscribe("error_occurred", self._on_error)
        self.event_bus.subscribe("reasoning_complete", self._on_reasoning_complete)

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process meta-cognitive operations"""

        operation = input_data.get("operation", "assess")

        if operation == "assess":
            return await self._assess_cognitive_state()
        elif operation == "reflect":
            return await self._reflect_on_performance()
        elif operation == "adapt":
            return await self._adapt_strategies()
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _monitoring_loop(self):
        """Continuous self-monitoring"""

        while True:
            try:
                # Update cognitive state
                await self._update_cognitive_state()

                # Check for concerning patterns
                await self._check_patterns()

                # Adjust strategies if needed
                if self.cognitive_state["confidence"] < 0.3:
                    await self._adapt_strategies()

                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error("metacognition_error", error=str(e))

    async def _assess_cognitive_state(self) -> Dict:
        """Assess current cognitive state"""

        # Analyze recent performance
        recent_performance = list(self.performance_history)[-10:]

        if recent_performance:
            # Calculate metrics
            success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
            avg_confidence = np.mean([p.get("confidence", 0.5) for p in recent_performance])

            # Update cognitive state
            self.cognitive_state["confidence"] = 0.7 * self.cognitive_state["confidence"] + 0.3 * avg_confidence

            # Assess cognitive load
            response_times = [p.get("duration", 1.0) for p in recent_performance]
            normalized_time = np.mean(response_times) / 5.0  # Normalize to 5 second baseline
            self.cognitive_state["cognitive_load"] = min(normalized_time, 1.0)

            # Assess clarity
            error_rate = 1.0 - success_rate
            self.cognitive_state["clarity"] = 1.0 - error_rate

        # Use Gemini for qualitative assessment
        assessment = await self._gemini_self_assessment()

        return {
            "success": True,
            "cognitive_state": self.cognitive_state.copy(),
            "qualitative_assessment": assessment,
            "active_strategies": list(self.active_strategies)
        }

    async def _reflect_on_performance(self) -> Dict:
        """Reflect on recent performance and extract insights"""

        recent = list(self.performance_history)[-20:]

        if not recent:
            return {"success": True, "insights": "No recent performance to analyze"}

        # Prepare performance summary
        summary = self._summarize_performance(recent)

        # Use Gemini for deep reflection
        prompt = f"""Reflect on this cognitive performance data:

        {summary}
        
        Analyze:
        1. What patterns indicate strong performance?
        2. What patterns suggest areas for improvement?
        3. Are there specific types of problems causing difficulty?
        4. What cognitive strategies might help?
        
        Provide actionable insights."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        insights = response["text"] if response["success"] else "Reflection unavailable"

        # Extract patterns
        patterns = self._extract_patterns(recent)

        return {
            "success": True,
            "insights": insights,
            "patterns": patterns,
            "recommendations": self._generate_recommendations(patterns)
        }

    async def _adapt_strategies(self) -> Dict:
        """Adapt cognitive strategies based on performance"""

        current_state = self.cognitive_state.copy()

        adaptations = []

        # Low confidence - increase verification
        if current_state["confidence"] < 0.4:
            adaptations.append("enable_double_checking")
            self.active_strategies.add("double_check")

            await self.event_bus.emit("enable_verification_mode", {
                "reason": "low_confidence",
                "threshold": 0.8
            })

        # High cognitive load - simplify
        if current_state["cognitive_load"] > 0.8:
            adaptations.append("simplify_processing")
            self.active_strategies.add("decomposition")

            await self.event_bus.emit("enable_decomposition_mode", {
                "reason": "high_load",
                "max_chunk_size": 3
            })

        # Low clarity - request clarification
        if current_state["clarity"] < 0.5:
            adaptations.append("increase_analysis_depth")
            self.active_strategies.add("deep_analysis")

        # High fatigue - reduce complexity
        if current_state["fatigue"] > 0.7:
            adaptations.append("reduce_complexity")
            self.active_strategies.discard("deep_analysis")

        logger.info("strategies_adapted",
                   adaptations=adaptations,
                   active_strategies=list(self.active_strategies))

        return {
            "success": True,
            "adaptations": adaptations,
            "active_strategies": list(self.active_strategies),
            "cognitive_state": current_state
        }

    async def _update_cognitive_state(self):
        """Update cognitive state based on recent activity"""

        # Fatigue increases over time, decreases with success
        self.cognitive_state["fatigue"] = min(
            self.cognitive_state["fatigue"] + 0.01,  # Gradual increase
            1.0
        )

        # Recent success reduces fatigue
        if self.performance_history and self.performance_history[-1].get("success"):
            self.cognitive_state["fatigue"] *= 0.95

        # Update metrics
        if self.performance_history:
            recent = list(self.performance_history)[-10:]

            # Error rate
            errors = sum(1 for p in recent if not p.get("success", True))
            self.metrics["error_rate"] = errors / len(recent)

            # Learning rate (improvement over time)
            if len(self.performance_history) > 20:
                old_success = sum(1 for p in list(self.performance_history)[-20:-10] if p.get("success"))
                new_success = sum(1 for p in recent if p.get("success"))
                self.metrics["learning_rate"] = (new_success - old_success) / 10.0

    async def _check_patterns(self):
        """Check for concerning patterns in performance"""

        if self.metrics["error_rate"] > 0.5:
            logger.warning("high_error_rate_detected", rate=self.metrics["error_rate"])

            # Trigger adaptation
            await self._adapt_strategies()

        # Check for repeated errors
        for error_type, count in self.error_patterns.items():
            if count > 3:
                logger.warning("repeated_error_pattern",
                             error_type=error_type,
                             count=count)

                # Request focused learning
                await self.event_bus.emit("focus_learning", {
                    "error_type": error_type,
                    "occurrences": count
                })

    async def _gemini_self_assessment(self) -> str:
        """Use Gemini for qualitative self-assessment"""

        state_description = f"""
        Confidence: {self.cognitive_state['confidence']:.2f}
        Cognitive Load: {self.cognitive_state['cognitive_load']:.2f}
        Fatigue: {self.cognitive_state['fatigue']:.2f}
        Clarity: {self.cognitive_state['clarity']:.2f}
        
        Recent Error Rate: {self.metrics['error_rate']:.2%}
        Learning Rate: {self.metrics['learning_rate']:+.2f}
        Active Strategies: {', '.join(self.active_strategies) or 'None'}
        """

        prompt = f"""Assess this cognitive state:

        {state_description}
        
        Provide a brief qualitative assessment of:
        1. Overall cognitive health
        2. Most concerning metric
        3. Recommended immediate action"""

        response = await self.gemini.generate(prompt, config_name="fast")

        return response["text"] if response["success"] else "Assessment unavailable"

    def _summarize_performance(self, performance_list: List[Dict]) -> str:
        """Summarize performance data"""

        total = len(performance_list)
        successes = sum(1 for p in performance_list if p.get("success"))

        avg_duration = np.mean([p.get("duration", 0) for p in performance_list])
        avg_confidence = np.mean([p.get("confidence", 0.5) for p in performance_list])

        task_types = defaultdict(int)
        for p in performance_list:
            task_type = p.get("task_type", "unknown")
            task_types[task_type] += 1

        summary = f"""Performance Summary:
        - Total Tasks: {total}
        - Success Rate: {successes/total:.1%}
        - Average Duration: {avg_duration:.1f}s
        - Average Confidence: {avg_confidence:.2f}
        
        Task Distribution:
        """
        for task_type, count in task_types.items():
            summary += f"- {task_type}: {count} ({count/total:.1%})\n"

        return summary

    def _extract_patterns(self, performance_list: List[Dict]) -> Dict:
        """Extract patterns from performance data"""

        patterns = {
            "success_by_type": defaultdict(list),
            "duration_by_type": defaultdict(list),
            "error_sequences": [],
            "improvement_areas": []
        }

        # Analyze by task type
        for p in performance_list:
            task_type = p.get("task_type", "unknown")
            patterns["success_by_type"][task_type].append(p.get("success", False))
            patterns["duration_by_type"][task_type].append(p.get("duration", 0))

        # Find error sequences
        error_streak = 0
        for p in performance_list:
            if not p.get("success", True):
                error_streak += 1
            else:
                if error_streak >= 2:
                    patterns["error_sequences"].append(error_streak)
                error_streak = 0

        # Identify improvement areas
        for task_type, successes in patterns["success_by_type"].items():
            if successes and sum(successes) / len(successes) < 0.6:
                patterns["improvement_areas"].append(task_type)

        return patterns

    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on patterns"""

        recommendations = []

        # Task-specific recommendations
        for task_type in patterns["improvement_areas"]:
            recommendations.append(f"Focus practice on {task_type} tasks")

        # Error sequence recommendations
        if patterns["error_sequences"]:
            max_streak = max(patterns["error_sequences"])
            if max_streak >= 3:
                recommendations.append("Take breaks after 2 consecutive errors")

        # Duration-based recommendations
        for task_type, durations in patterns["duration_by_type"].items():
            if durations and np.mean(durations) > 10:
                recommendations.append(f"Break down {task_type} tasks into smaller steps")

        return recommendations

    async def _on_task_complete(self, data: Dict):
        """Handle task completion events"""

        performance_record = {
            "timestamp": time.time(),
            "success": data.get("success", True),
            "duration": data.get("duration", 0),
            "confidence": data.get("confidence", 0.5),
            "task_type": data.get("task_type", "general")
        }

        self.performance_history.append(performance_record)

        # Update response times
        self.metrics["response_times"].append(performance_record["duration"])

        # Track strategy effectiveness
        for strategy in self.active_strategies:
            if strategy not in self.strategy_effectiveness:
                self.strategy_effectiveness[strategy] = []
            self.strategy_effectiveness[strategy].append(performance_record["success"])

    async def _on_error(self, data: Dict):
        """Handle error events"""

        error_type = data.get("error_type", "unknown")
        self.error_patterns[error_type] += 1

        # Log for analysis
        logger.warning("cognitive_error_tracked",
                      error_type=error_type,
                      count=self.error_patterns[error_type])

    async def _on_reasoning_complete(self, data: Dict):
        """Handle reasoning completion events"""

        # Extract reasoning quality metrics
        confidence = data.get("confidence", 0.5)
        reasoning_paths = data.get("reasoning_paths", [])

        # Update cognitive state based on reasoning quality
        if len(reasoning_paths) > 1:
            # Multiple paths indicate thorough reasoning
            self.cognitive_state["clarity"] = 0.8 * self.cognitive_state["clarity"] + 0.2 * 0.8

    def get_state(self) -> Dict[str, Any]:
        return {
            "cognitive_state": self.cognitive_state.copy(),
            "active_strategies": list(self.active_strategies),
            "metrics": self.metrics.copy(),
            "error_patterns": dict(self.error_patterns),
            "performance_history_size": len(self.performance_history)
        }