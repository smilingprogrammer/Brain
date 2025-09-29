import random
import numpy as np
from typing import Dict, Any, List, Optional
from core.interfaces import ReasoningModule
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class CreativeReasoning(ReasoningModule):
    """Creative and divergent thinking processes"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.confidence = 0.0
        self.creativity_temperature = 0.8

    async def initialize(self):
        """Initialize creative reasoning"""
        logger.info("initializing_creative_reasoning")

        # Subscribe to reasoning requests
        self.event_bus.subscribe("creative_reasoning_request", self._on_reasoning_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process creative reasoning request"""

        problem = input_data.get("problem", "")
        constraints = input_data.get("constraints", [])
        context = input_data.get("context", {})

        result = await self.reason(problem, context)

        # Apply constraints if any
        if constraints:
            result = await self._apply_constraints(result, constraints)

        # Emit completion
        await self.event_bus.emit("creative_reasoning_complete", result)

        return result

    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform creative reasoning"""

        # Analyze problem space
        problem_space = await self._analyze_problem_space(problem, context)

        # Generate diverse ideas
        ideas = await self._generate_ideas(problem_space)

        # Combine ideas creatively
        combinations = await self._combine_ideas(ideas)

        # Apply creative transformations
        transformed = await self._apply_transformations(ideas + combinations)

        # Evaluate novelty and usefulness
        evaluated = await self._evaluate_creativity(transformed)

        # Select best creative solutions
        best_solutions = self._select_best(evaluated, n=3)

        self.confidence = self._compute_creative_confidence(best_solutions)

        return {
            "success": True,
            "problem": problem,
            "problem_space": problem_space,
            "ideas_generated": len(ideas),
            "solutions": best_solutions,
            "creative_process": {
                "divergent_ideas": len(ideas),
                "combinations": len(combinations),
                "transformations": len(transformed)
            },
            "confidence": self.confidence
        }

    async def _analyze_problem_space(self, problem: str, context: Dict) -> Dict:
        """Analyze the creative problem space"""

        prompt = f"""Analyze this creative problem:

Problem: {problem}
Context: {context}

Identify:
1. core_challenge: What needs to be solved/created
2. constraints: Explicit or implicit limitations
3. resources: Available elements to work with
4. assumptions: What can be challenged
5. dimensions: Different aspects to explore

Think broadly and identify opportunities for creativity."""

        response = await self.gemini.generate(prompt, config_name="creative")

        if response["success"]:
            # Parse problem space
            return self._parse_problem_space(response["text"])

        return {
            "core_challenge": problem,
            "constraints": [],
            "resources": [],
            "assumptions": [],
            "dimensions": []
        }

    async def _generate_ideas(self, problem_space: Dict) -> List[Dict]:
        """Generate diverse creative ideas"""

        ideas = []

        # Use different creative strategies
        strategies = [
            "random_association",
            "opposite_thinking",
            "analogy",
            "combination",
            "elimination",
            "exaggeration"
        ]

        for strategy in strategies:
            idea = await self._apply_strategy(strategy, problem_space)
            if idea:
                ideas.append(idea)

        # Add some wild ideas with high temperature
        wild_prompt = f"""Generate 3 wild, unconventional ideas for:
{problem_space.get('core_challenge')}

Be extremely creative, ignore practical constraints.
Think outside all boxes."""

        wild_response = await self.gemini.generate(
            wild_prompt,
            config_name="creative"
        )

        if wild_response["success"]:
            wild_ideas = self._parse_ideas(wild_response["text"], source="wild")
            ideas.extend(wild_ideas)

        return ideas

    async def _apply_strategy(self, strategy: str, problem_space: Dict) -> Optional[Dict]:
        """Apply specific creative strategy"""

        strategies_prompts = {
            "random_association": f"""
            Use random association to solve: {problem_space.get('core_challenge')}
            Connect with completely unrelated concepts: {random.choice(['ocean', 'music', 'cooking', 'space', 'dreams'])}
            """,
                        "opposite_thinking": f"""
            Solve by doing the opposite: {problem_space.get('core_challenge')}
            What if we tried to achieve the reverse? How might that lead to a solution?
            """,
                        "analogy": f"""
            Find an analogy for: {problem_space.get('core_challenge')}
            What similar problem exists in nature, art, or other fields?
            """,
                        "combination": f"""
            Combine unexpected elements for: {problem_space.get('core_challenge')}
            Merge concepts from different domains.
            """,
                        "elimination": f"""
            Solve by elimination for: {problem_space.get('core_challenge')}
            What if we removed a key assumption or component?
            """,
                        "exaggeration": f"""
            Solve by exaggeration for: {problem_space.get('core_challenge')}
            What if we made something 100x bigger/smaller/faster?
            """
        }

        prompt = strategies_prompts.get(strategy, "")
        if not prompt:
            return None

        response = await self.gemini.generate(prompt, config_name="creative")

        if response["success"]:
            return {
                "idea": response["text"],
                "strategy": strategy,
                "source": "strategic",
                "novelty": 0.7,
                "raw_text": response["text"]
            }

        return None

    async def _combine_ideas(self, ideas: List[Dict]) -> List[Dict]:
        """Combine ideas to create new ones"""

        combinations = []

        if len(ideas) < 2:
            return combinations

        # Select random pairs to combine
        num_combinations = min(len(ideas) // 2, 5)

        for _ in range(num_combinations):
            # Randomly select two different ideas
            idea1, idea2 = random.sample(ideas, 2)

            prompt = f"""Creatively combine these two ideas:

        Idea 1: {idea1.get('idea', '')}
        Idea 2: {idea2.get('idea', '')}

        Create a novel synthesis that incorporates elements of both."""

            response = await self.gemini.generate(prompt, config_name="creative")

            if response["success"]:
                combinations.append({
                    "idea": response["text"],
                    "strategy": "combination",
                    "source": "combined",
                    "parents": [idea1.get('idea', ''), idea2.get('idea', '')],
                    "novelty": 0.8
                })

        return combinations

    async def _apply_transformations(self, ideas: List[Dict]) -> List[Dict]:
        """Apply creative transformations to ideas"""

        transformed = []

        transformations = [
            "reverse", "fragment", "multiply",
            "substitute", "adapt", "modify"
        ]

        # Transform a subset of ideas
        ideas_to_transform = random.sample(
            ideas,
            min(len(ideas), len(ideas) // 2 + 1)
        )

        for idea in ideas_to_transform:
            transformation = random.choice(transformations)

            prompt = f"""Apply '{transformation}' transformation to this idea:

        Original: {idea.get('idea', '')}

        Transform it creatively while maintaining its essence."""

            response = await self.gemini.generate(prompt, config_name="creative")

            if response["success"]:
                transformed.append({
                    "idea": response["text"],
                    "strategy": f"transform_{transformation}",
                    "source": "transformed",
                    "original": idea.get('idea', ''),
                    "novelty": 0.85
                })

        # Add original ideas too
        transformed.extend(ideas)

        return transformed

    async def _evaluate_creativity(self, ideas: List[Dict]) -> List[Dict]:
        """Evaluate ideas for creativity"""

        evaluated = []

        for idea in ideas:
            # Skip if already evaluated
            if "creativity_score" in idea:
                evaluated.append(idea)
                continue

            prompt = f"""Evaluate this creative idea:

        Idea: {idea.get('idea', '')}

        Rate on:
        1. Novelty (0-10): How original and unexpected?
        2. Usefulness (0-10): How well does it solve the problem?
        3. Elegance (0-10): How simple yet effective?
        4. Feasibility (0-10): How implementable?

        Provide brief reasoning for each score."""

            response = await self.gemini.generate(prompt, config_name="balanced")

            if response["success"]:
                scores = self._parse_creativity_scores(response["text"])

                idea["novelty_score"] = scores.get("novelty", 5) / 10
                idea["usefulness_score"] = scores.get("usefulness", 5) / 10
                idea["elegance_score"] = scores.get("elegance", 5) / 10
                idea["feasibility_score"] = scores.get("feasibility", 5) / 10

                # Weighted creativity score
                idea["creativity_score"] = (
                        0.4 * idea["novelty_score"] +
                        0.3 * idea["usefulness_score"] +
                        0.2 * idea["elegance_score"] +
                        0.1 * idea["feasibility_score"]
                )

                idea["evaluation"] = response["text"]
            else:
                # Default scores
                idea["creativity_score"] = idea.get("novelty", 0.5)

            evaluated.append(idea)

        return evaluated

    async def _apply_constraints(self, result: Dict, constraints: List[str]) -> Dict:
        """Apply constraints to filter or modify solutions"""

        if "solutions" not in result:
            return result

        # Filter solutions based on constraints
        filtered_solutions = []

        for solution in result["solutions"]:
            prompt = f"""Check if this solution violates any constraints:

        Solution: {solution.get('idea', '')}
        Constraints: {', '.join(constraints)}

        If it violates constraints, suggest a modification.
        If it's acceptable, say "PASSES"."""

            response = await self.gemini.generate(prompt, config_name="fast")

            if response["success"]:
                if "PASSES" in response["text"].upper():
                    filtered_solutions.append(solution)
                else:
                    # Try to modify
                    modified = solution.copy()
                    modified["idea"] = response["text"]
                    modified["modified_for_constraints"] = True
                    filtered_solutions.append(modified)

        result["solutions"] = filtered_solutions
        result["constraints_applied"] = constraints

        return result

    def _parse_problem_space(self, text: str) -> Dict:
        """Parse problem space from text"""

        space = {
            "core_challenge": "",
            "constraints": [],
            "resources": [],
            "assumptions": [],
            "dimensions": []
        }

        lines = text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            if "challenge" in line.lower():
                current_section = "core_challenge"
            elif "constraint" in line.lower():
                current_section = "constraints"
            elif "resource" in line.lower():
                current_section = "resources"
            elif "assumption" in line.lower():
                current_section = "assumptions"
            elif "dimension" in line.lower():
                current_section = "dimensions"
            elif current_section:
                # Add to current section
                if current_section == "core_challenge":
                    space["core_challenge"] += line + " "
                elif current_section in ["constraints", "resources", "assumptions", "dimensions"]:
                    # Clean and add
                    cleaned = line.strip('- •·123456789.')
                    if cleaned:
                        space[current_section].append(cleaned)

        # Clean core challenge
        space["core_challenge"] = space["core_challenge"].strip()

        return space

    def _parse_ideas(self, text: str, source: str = "generated") -> List[Dict]:
        """Parse ideas from text"""

        ideas = []
        lines = text.split('\n')

        current_idea = ""

        for line in lines:
            line = line.strip()

            if not line:
                if current_idea:
                    ideas.append({
                        "idea": current_idea,
                        "source": source,
                        "strategy": "generation",
                        "novelty": 0.7
                    })
                    current_idea = ""
                continue

            # Look for numbered or bulleted ideas
            if line[0].isdigit() or line.startswith('-') or line.startswith('•'):
                if current_idea:
                    ideas.append({
                        "idea": current_idea,
                        "source": source,
                        "strategy": "generation",
                        "novelty": 0.7
                    })

                current_idea = line.lstrip('0123456789.-• ')
            else:
                current_idea += " " + line

        # Add last idea
        if current_idea:
            ideas.append({
                "idea": current_idea,
                "source": source,
                "strategy": "generation",
                "novelty": 0.7
            })

        return ideas

    def _parse_creativity_scores(self, text: str) -> Dict[str, float]:
        """Parse creativity scores from evaluation text"""

        scores = {
            "novelty": 5,
            "usefulness": 5,
            "elegance": 5,
            "feasibility": 5
        }

        lines = text.lower().split('\n')

        for line in lines:
            # Look for score patterns
            for metric in scores.keys():
                if metric in line:
                    # Extract number
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = float(numbers[0])
                        if score <= 10:
                            scores[metric] = score

        return scores

    def _select_best(self, ideas: List[Dict], n: int = 3) -> List[Dict]:
        """Select best creative solutions"""

        # Sort by creativity score
        sorted_ideas = sorted(
            ideas,
            key=lambda x: x.get("creativity_score", 0),
            reverse=True
        )

        # Take top n
        best = sorted_ideas[:n]

        # Format for output
        formatted = []
        for i, idea in enumerate(best):
            formatted.append({
                "rank": i + 1,
                "idea": idea.get("idea", ""),
                "strategy": idea.get("strategy", "unknown"),
                "creativity_score": idea.get("creativity_score", 0),
                "novelty": idea.get("novelty_score", 0),
                "usefulness": idea.get("usefulness_score", 0),
                "evaluation": idea.get("evaluation", "")
            })

        return formatted

    def _compute_creative_confidence(self, solutions: List[Dict]) -> float:
        """Compute confidence in creative solutions"""

        if not solutions:
            return 0.0

        # Average creativity score of top solutions
        avg_creativity = np.mean([s.get("creativity_score", 0) for s in solutions])

        # Boost confidence if we have diverse strategies
        strategies = set(s.get("strategy", "") for s in solutions)
        diversity_bonus = min(len(strategies) * 0.1, 0.3)

        confidence = avg_creativity + diversity_bonus

        return min(confidence, 1.0)

    async def _on_reasoning_request(self, data: Dict):
        """Handle creative reasoning requests"""
        result = await self.process(data)

    def get_confidence(self) -> float:
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_confidence": self.confidence,
            "creativity_temperature": self.creativity_temperature
        }