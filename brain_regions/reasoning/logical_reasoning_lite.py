from typing import Dict, Any
from core.interfaces import ReasoningModule
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class LogicalReasoningLite(ReasoningModule):
    """Lightweight logical reasoning with single API call"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.confidence = 0.0

    async def initialize(self):
        """Initialize logical reasoning"""
        logger.info("initializing_logical_reasoning_lite")

        # Subscribe to reasoning requests
        self.event_bus.subscribe("reasoning_request", self._on_reasoning_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process logical reasoning request"""

        problem = input_data.get("problem", "")
        context = input_data.get("context", {})

        result = await self.reason(problem, context)

        # Emit reasoning complete
        await self.event_bus.emit("reasoning_request_complete", result)

        return result

    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform logical reasoning with a single API call"""

        # Single consolidated prompt for reasoning
        prompt = f"""Answer this question clearly and concisely: {problem}

Context: {context if context else 'None'}

Provide a direct, factual answer. If it requires reasoning, show your logic briefly."""

        response = await self.gemini.generate(prompt, config_name="fast")

        if not response["success"]:
            return {
                "success": False,
                "error": "Reasoning failed",
                "confidence": 0.0
            }

        # Simple confidence based on response
        self.confidence = 0.8

        return {
            "success": True,
            "conclusion": response["text"],
            "proof_steps": ["Direct answer provided"],
            "proof_type": "direct",
            "confidence": self.confidence
        }

    async def _on_reasoning_request(self, data: Dict):
        """Handle reasoning requests"""
        logger.info("reasoning_request_received_lite", problem=data.get("problem", "")[:50])
        result = await self.process(data)
        logger.info("reasoning_request_processed_lite", success=result.get("success"))

    def get_confidence(self) -> float:
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_confidence": self.confidence,
            "mode": "lite"
        }
