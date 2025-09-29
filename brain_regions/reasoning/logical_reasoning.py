from typing import Dict, List, Any, Optional
from core.interfaces import ReasoningModule
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class LogicalReasoning(ReasoningModule):
    """Formal logical reasoning module"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.confidence = 0.0
        self.last_proof = None

    async def initialize(self):
        """Initialize logical reasoning"""
        logger.info("initializing_logical_reasoning")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process logical reasoning request"""

        problem = input_data.get("problem", "")
        context = input_data.get("context", {})

        result = await self.reason(problem, context)

        # Emit reasoning complete
        await self.event_bus.emit("logical_reasoning_complete", result)

        return result

    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform logical reasoning"""

        # Extract premises and conclusion
        analysis = await self._analyze_logical_structure(problem)

        if not analysis["success"]:
            return {
                "success": False,
                "error": "Could not analyze logical structure",
                "confidence": 0.0
            }

        # Attempt formal proof
        proof = await self._construct_proof(
            analysis["premises"],
            analysis["conclusion"],
            context
        )

        # Validate proof
        validation = await self._validate_proof(proof)

        self.confidence = validation["confidence"]
        self.last_proof = proof

        return {
            "success": True,
            "conclusion": proof["conclusion"],
            "proof_steps": proof["steps"],
            "proof_type": proof["type"],
            "confidence": self.confidence,
            "validation": validation
        }

    async def _analyze_logical_structure(self, problem: str) -> Dict:
        """Extract logical structure from natural language"""

        prompt = f"""Analyze the logical structure of this problem:

        Problem: {problem}
        
        Extract:
        1. premises: List of logical premises
        2. conclusion: What needs to be proven/concluded
        3. logical_form: Type of reasoning needed (deductive/inductive/abductive)
        4. variables: Key variables or entities
        
        Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "premises": ["string"],
                "conclusion": "string",
                "logical_form": "string",
                "variables": ["string"]
            }
        )

        if response["success"] and response["parsed"]:
            return {
                "success": True,
                **response["parsed"]
            }

        return {"success": False}

    async def _construct_proof(self, premises: List[str], conclusion: str, context: Dict) -> Dict:
        """Construct a logical proof"""

        prompt = f"""Construct a formal logical proof:
        
        Premises:
        {chr(10).join(f'{i + 1}. {p}' for i, p in enumerate(premises))}
        
        Conclusion to prove: {conclusion}
        
        Context: {context.get('working_memory_summary', 'None')}
        
        Provide:
        1. Step-by-step proof
        2. Logical rules used at each step
        3. Type of proof (direct, contradiction, induction, etc.)
        
        Be rigorous and explicit about each logical step."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            # Parse proof steps
            steps = self._parse_proof_steps(response["text"])

            return {
                "premises": premises,
                "conclusion": conclusion,
                "steps": steps,
                "type": self._identify_proof_type(response["text"]),
                "raw_proof": response["text"]
            }

        return {
            "premises": premises,
            "conclusion": conclusion,
            "steps": [],
            "type": "failed",
            "error": "Could not construct proof"
        }

    async def _validate_proof(self, proof: Dict) -> Dict:
        """Validate the logical proof"""

        prompt = f"""Validate this logical proof for correctness:

        Proof:
        {proof['raw_proof']}
        
        Check for:
        1. Logical fallacies
        2. Invalid inferences
        3. Missing steps
        4. Circular reasoning
        5. Unsupported assumptions
        
        Rate confidence (0-1) and explain any issues."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            # Extract confidence and issues
            confidence = self._extract_confidence(response["text"])
            issues = self._extract_issues(response["text"])

            return {
                "valid": confidence > 0.7,
                "confidence": confidence,
                "issues": issues,
                "validation_details": response["text"]
            }

        return {
            "valid": False,
            "confidence": 0.0,
            "issues": ["Validation failed"],
            "validation_details": ""
        }

    def get_confidence(self) -> float:
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_confidence": self.confidence,
            "has_active_proof": self.last_proof is not None
        }

    # Helper methods
    def _parse_proof_steps(self, proof_text: str) -> List[Dict]:
        """Parse proof text into structured steps"""
        steps = []
        lines = proof_text.split('\n')

        current_step = None
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    "statement": line,
                    "justification": "",
                    "rule": ""
                }
            elif current_step and line:
                current_step["justification"] += line + " "

        if current_step:
            steps.append(current_step)

        return steps

    def _identify_proof_type(self, proof_text: str) -> str:
        """Identify the type of proof used"""
        proof_text_lower = proof_text.lower()

        if "contradiction" in proof_text_lower:
            return "proof_by_contradiction"
        elif "induction" in proof_text_lower:
            return "mathematical_induction"
        elif "cases" in proof_text_lower:
            return "proof_by_cases"
        elif "contrapositive" in proof_text_lower:
            return "proof_by_contrapositive"
        else:
            return "direct_proof"

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        import re

        # Look for confidence mentions
        patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)\s*confidence',
            r'confident[:\s]+([0-9.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass

        # Heuristic based on keywords
        if "highly confident" in text.lower():
            return 0.9
        elif "confident" in text.lower():
            return 0.7
        elif "uncertain" in text.lower():
            return 0.3

        return 0.5

    def _extract_issues(self, text: str) -> List[str]:
        """Extract validation issues from text"""
        issues = []

        issue_keywords = [
            "fallacy", "invalid", "missing", "circular",
            "unsupported", "assumption", "error", "incorrect"
        ]

        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in issue_keywords):
                issues.append(line.strip())

        return issues[:5]  # Limit to top 5 issues