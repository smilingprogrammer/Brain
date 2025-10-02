from typing import Dict, Any, List, Optional
import asyncio
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class ReasoningEngine:
    """Gemini-powered multi-path reasoning engine"""

    def __init__(self, gemini_service: GeminiService, event_bus: EventBus):
        self.gemini = gemini_service
        self.event_bus = event_bus

        # Reasoning strategies
        self.strategies = {
            "deductive": self._deductive_reasoning,
            "inductive": self._inductive_reasoning,
            "abductive": self._abductive_reasoning,
            "analogical": self._analogical_reasoning,
            "causal": self._causal_reasoning,
            "counterfactual": self._counterfactual_reasoning
        }

        # Reasoning cache
        self.reasoning_cache = {}

    async def initialize(self):
        """Initialize reasoning engine"""
        logger.info("initializing_gemini_reasoning_engine")

        # Subscribe to reasoning requests
        self.event_bus.subscribe("multi_path_reasoning", self._on_reasoning_request)

    async def reason(self, problem: str, context: Dict, strategies: Optional[List[str]] = None) -> Dict:
        """Perform multi-path reasoning"""

        # Use all strategies if none specified
        if not strategies:
            strategies = list(self.strategies.keys())

        # Check cache
        cache_key = f"{problem}:{':'.join(sorted(strategies))}"
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]

        # Run strategies in parallel
        tasks = []
        for strategy in strategies:
            if strategy in self.strategies:
                tasks.append(self.strategies[strategy](problem, context))

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        reasoning_paths = []
        for strategy, result in zip(strategies, results):
            if isinstance(result, Exception):
                logger.error("reasoning_strategy_failed",
                             strategy=strategy,
                             error=str(result))
            else:
                reasoning_paths.append({
                    "strategy": strategy,
                    "result": result
                })

        # Synthesize results
        synthesis = await self._synthesize_reasoning(reasoning_paths, problem)

        final_result = {
            "success": True,
            "problem": problem,
            "reasoning_paths": reasoning_paths,
            "synthesis": synthesis,
            "confidence": synthesis.get("confidence", 0.5)
        }

        # Cache result
        self.reasoning_cache[cache_key] = final_result

        return final_result

    async def _deductive_reasoning(self, problem: str, context: Dict) -> Dict:
        """Apply deductive reasoning"""

        prompt = f"""Apply deductive reasoning to solve:

Problem: {problem}
Context: {context}

Use formal logic:
1. Identify premises
2. Apply logical rules
3. Derive conclusion
4. Show each step clearly

Format:
Premise 1: ...
Premise 2: ...
Rule: ...
Therefore: ..."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            return {
                "approach": "deductive",
                "reasoning": response["text"],
                "confidence": self._extract_confidence(response["text"], default=0.8)
            }

        return {"approach": "deductive", "error": "Failed"}

    async def _inductive_reasoning(self, problem: str, context: Dict) -> Dict:
        """Apply inductive reasoning"""

        prompt = f"""Apply inductive reasoning to solve:

Problem: {problem}
Context: {context}

Use pattern recognition:
1. Identify specific examples
2. Find patterns or regularities
3. Generalize to broader principle
4. State confidence in generalization

Show your reasoning process."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            return {
                "approach": "inductive",
                "reasoning": response["text"],
                "confidence": self._extract_confidence(response["text"], default=0.7)
            }

        return {"approach": "inductive", "error": "Failed"}

    async def _abductive_reasoning(self, problem: str, context: Dict) -> Dict:
        """Apply abductive reasoning"""

        prompt = f"""Apply abductive reasoning to solve:

        Problem: {problem}
        Context: {context}
        
        Find the best explanation:
        1. List observations
        2. Generate possible explanations
        3. Evaluate each explanation
        4. Select most likely explanation
        5. State reasoning for selection"""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            return {
                "approach": "abductive",
                "reasoning": response["text"],
                "confidence": self._extract_confidence(response["text"], default=0.6)
            }

        return {"approach": "abductive", "error": "Failed"}

    async def _analogical_reasoning(self, problem: str, context: Dict) -> Dict:
        """Apply analogical reasoning"""

        prompt = f"""Apply analogical reasoning to solve:

        Problem: {problem}
        Context: {context}
        
        Use analogies:
        1. Find similar problem or situation
        2. Map relationships between domains
        3. Transfer solution approach
        4. Adapt to current problem
        5. Evaluate fit"""

        response = await self.gemini.generate(prompt, config_name="creative")

        if response["success"]:
            return {
                "approach": "analogical",
                "reasoning": response["text"],
                "confidence": self._extract_confidence(response["text"], default=0.65)
            }

        return {"approach": "analogical", "error": "Failed"}

    async def _causal_reasoning(self, problem: str, context: Dict) -> Dict:
        """Apply causal reasoning"""

        prompt = f"""Apply causal reasoning to solve:

Problem: {problem}
Context: {context}

Trace cause and effect:
1. Identify initial causes
2. Map causal chains
3. Consider feedback loops
4. Predict outcomes
5. Identify intervention points"""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            return {
                "approach": "causal",
                "reasoning": response["text"],
                "confidence": self._extract_confidence(response["text"], default=0.75)
            }

        return {"approach": "causal", "error": "Failed"}

    async def _counterfactual_reasoning(self, problem: str, context: Dict) -> Dict:
        """Apply counterfactual reasoning"""

        prompt = f"""Apply counterfactual reasoning to solve:

Problem: {problem}
Context: {context}

Explore alternatives:
1. Identify key assumptions
2. Consider "what if" scenarios
3. Trace alternative outcomes
4. Compare with actual situation
5. Draw insights"""

        response = await self.gemini.generate(prompt, config_name="creative")

        if response["success"]:
            return {
                "approach": "counterfactual",
                "reasoning": response["text"],
                "confidence": self._extract_confidence(response["text"], default=0.6)
            }

        return {"approach": "counterfactual", "error": "Failed"}

    async def _synthesize_reasoning(self, paths: List[Dict], problem: str) -> Dict:
        """Synthesize multiple reasoning paths"""

        if not paths:
            return {
                "conclusion": "No valid reasoning paths",
                "confidence": 0.0
            }

        # Format paths for synthesis
        path_summaries = []
        for path in paths:
            if "error" not in path.get("result", {}):
                summary = f"{path['strategy']}: {path['result'].get('reasoning', '')[:200]}..."
                path_summaries.append(summary)

        prompt = f"""Synthesize these different reasoning approaches for the problem:
        "{problem}"
        
        Reasoning paths:
        {chr(10).join(path_summaries)}
        
        Create a unified conclusion that:
        1. Identifies points of agreement
        2. Resolves contradictions
        3. Combines insights from each approach
        4. States overall confidence (0-1)
        5. Provides final answer"""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            return self._parse_synthesis(response["text"])

        # Fallback to highest confidence path
        best_path = max(paths, key=lambda p: p.get("result", {}).get("confidence", 0))
        return {
            "conclusion": best_path.get("result", {}).get("reasoning", ""),
            "confidence": best_path.get("result", {}).get("confidence", 0.5),
            "primary_approach": best_path.get("strategy", "unknown")
        }

    def _parse_synthesis(self, text: str) -> Dict:
        """Parse synthesis from text"""

        synthesis = {
            "conclusion": "",
            "confidence": 0.7,
            "agreements": [],
            "insights": []
        }

        lines = text.split('\n')
        current_section = "conclusion"

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            if "agreement" in line.lower():
                current_section = "agreements"
            elif "insight" in line.lower():
                current_section = "insights"
            elif "confidence" in line.lower():
                # Extract confidence
                import re
                numbers = re.findall(r'0?\.\d+|1\.0', line)
                if numbers:
                    synthesis["confidence"] = float(numbers[0])
            elif "conclusion" in line.lower() or "answer" in line.lower():
                current_section = "conclusion"
            else:
                # Add to current section
                if current_section == "conclusion":
                    synthesis["conclusion"] += line + " "
                elif current_section == "agreements" and line.strip():
                    synthesis["agreements"].append(line.strip('•-123456789. '))
                elif current_section == "insights" and line.strip():
                    synthesis["insights"].append(line.strip('•-123456789. '))

        synthesis["conclusion"] = synthesis["conclusion"].strip()

        return synthesis

    def _extract_confidence(self, text: str, default: float = 0.5) -> float:
        """Extract confidence from reasoning text"""

        import re

        # Look for explicit confidence statements
        confidence_patterns = [
            r'confidence:?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*confidence',
            r'confident:?\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, text.lower())
            if match:
                conf = float(match.group(1))
                if conf <= 1.0:
                    return conf
                elif conf <= 100:
                    return conf / 100

        # Heuristic based on certainty words
        certainty_words = {
            "certain": 0.95,
            "highly confident": 0.9,
            "confident": 0.8,
            "likely": 0.7,
            "probable": 0.7,
            "possible": 0.5,
            "uncertain": 0.3,
            "unlikely": 0.2
        }

        text_lower = text.lower()
        for word, conf in certainty_words.items():
            if word in text_lower:
                return conf

        return default

    async def _on_reasoning_request(self, data: Dict):
        """Handle multi-path reasoning requests"""

        problem = data.get("problem", "")
        context = data.get("context", {})
        strategies = data.get("strategies")

        result = await self.reason(problem, context, strategies)

        # Emit result
        await self.event_bus.emit("multi_path_reasoning_complete", result)