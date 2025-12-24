import numpy as np
from typing import Dict, Any, List
from core.interfaces import ReasoningModule
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class CreativeReasoningLite(ReasoningModule):
    """Lightweight creative reasoning with minimal API calls"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.confidence = 0.0

    async def initialize(self):
        """Initialize creative reasoning"""
        logger.info("initializing_creative_reasoning_lite")

        # Subscribe to reasoning requests
        self.event_bus.subscribe("creative_reasoning_request", self._on_reasoning_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process creative reasoning request"""

        problem = input_data.get("problem", "")
        context = input_data.get("context", {})

        result = await self.reason(problem, context)

        # Emit completion
        await self.event_bus.emit("creative_reasoning_request_complete", result)

        return result

    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform creative reasoning with a single consolidated API call"""

        # Single comprehensive prompt for creative thinking
        prompt = f"""Think creatively about this problem: {problem}

Context: {context}

Provide a creative, thoughtful response that:
1. Analyzes the core challenge
2. Explores multiple perspectives and approaches
3. Offers 2-3 innovative solutions or insights
4. Considers both conventional and unconventional angles

Be thorough but concise. Focus on practical creativity."""

        response = await self.gemini.generate(prompt, config_name="creative")

        if not response["success"]:
            return {
                "success": False,
                "error": "Creative reasoning failed",
                "confidence": 0.0
            }

        # Parse the response into structured solutions
        solutions = self._parse_solutions(response["text"])

        # Simple confidence based on response quality
        self.confidence = 0.7 if len(solutions) > 0 else 0.3

        return {
            "success": True,
            "problem": problem,
            "conclusion": response["text"],
            "solutions": solutions,
            "ideas_generated": len(solutions),
            "confidence": self.confidence,
            "creative_process": {
                "method": "consolidated_single_call",
                "api_calls": 1
            }
        }

    def _parse_solutions(self, text: str) -> List[Dict]:
        """Parse solutions from the response text"""

        solutions = []
        lines = text.split('\n')

        current_solution = ""
        solution_num = 0

        for line in lines:
            line = line.strip()

            if not line:
                if current_solution and len(current_solution) > 20:
                    solutions.append({
                        "rank": solution_num,
                        "idea": current_solution,
                        "strategy": "creative_thinking",
                        "creativity_score": 0.7,
                        "source": "consolidated"
                    })
                    solution_num += 1
                    current_solution = ""
                continue

            # Look for solution markers
            if any(marker in line.lower() for marker in ['solution', 'approach', 'insight', 'idea']):
                if current_solution and len(current_solution) > 20:
                    solutions.append({
                        "rank": solution_num,
                        "idea": current_solution,
                        "strategy": "creative_thinking",
                        "creativity_score": 0.7,
                        "source": "consolidated"
                    })
                    solution_num += 1

                current_solution = line.lstrip('0123456789.-• ')
            else:
                current_solution += " " + line

        # Add last solution
        if current_solution and len(current_solution) > 20:
            solutions.append({
                "rank": solution_num,
                "idea": current_solution,
                "strategy": "creative_thinking",
                "creativity_score": 0.7,
                "source": "consolidated"
            })

        # If no structured solutions found, use the whole text as one solution
        if not solutions:
            solutions.append({
                "rank": 1,
                "idea": text[:500],  # First 500 chars
                "strategy": "holistic",
                "creativity_score": 0.6,
                "source": "full_response"
            })

        return solutions[:3]  # Return top 3

    async def _on_reasoning_request(self, data: Dict):
        """Handle creative reasoning requests"""
        result = await self.process(data)

    def get_confidence(self) -> float:
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_confidence": self.confidence,
            "mode": "lite"
        }
