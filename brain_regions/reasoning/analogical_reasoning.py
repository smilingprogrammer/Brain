from typing import Dict, Any, List, Tuple
import numpy as np
from core.interfaces import ReasoningModule
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class AnalogicalReasoning(ReasoningModule):
    """Reasoning by analogy and pattern mapping"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.confidence = 0.0
        self.analogy_cache = {}

    async def initialize(self):
        """Initialize analogical reasoning"""
        logger.info("initializing_analogical_reasoning")

        # Subscribe to reasoning requests
        self.event_bus.subscribe("analogical_reasoning_request", self._on_reasoning_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analogical reasoning request"""

        source = input_data.get("source", "")
        target = input_data.get("target", "")
        context = input_data.get("context", {})

        result = await self.reason(f"Find analogy between {source} and {target}", context)

        # Emit completion
        await self.event_bus.emit("analogical_reasoning_complete", result)

        return result

    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform analogical reasoning"""

        # Extract source and target domains
        domains = await self._extract_domains(problem)

        if not domains["success"]:
            return {
                "success": False,
                "error": "Could not extract domains for analogy"
            }

        # Find mappings between domains
        mappings = await self._find_mappings(
            domains["source"],
            domains["target"],
            context
        )

        # Generate insights from mappings
        insights = await self._generate_insights(mappings, domains)

        # Evaluate analogy quality
        quality = self._evaluate_analogy(mappings)

        self.confidence = quality["score"]

        return {
            "success": True,
            "source_domain": domains["source"],
            "target_domain": domains["target"],
            "mappings": mappings,
            "insights": insights,
            "quality": quality,
            "confidence": self.confidence
        }

    async def _extract_domains(self, problem: str) -> Dict:
        """Extract source and target domains from problem"""

        prompt = f"""Extract the source and target domains for analogical reasoning:

        Problem: {problem}
        
        Identify:
        1. source_domain: The known domain/situation
        2. target_domain: The domain/situation to understand
        3. source_elements: Key elements in source
        4. target_elements: Key elements in target
        
        Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "source_domain": "string",
                "target_domain": "string",
                "source_elements": ["string"],
                "target_elements": ["string"]
            }
        )

        if response["success"] and response["parsed"]:
            return {
                "success": True,
                "source": response["parsed"]["source_domain"],
                "target": response["parsed"]["target_domain"],
                "source_elements": response["parsed"]["source_elements"],
                "target_elements": response["parsed"]["target_elements"]
            }

        return {"success": False}

    async def _find_mappings(self, source: str, target: str, context: Dict) -> List[Dict]:
        """Find mappings between source and target domains"""

        # Check cache
        cache_key = f"{source}:{target}"
        if cache_key in self.analogy_cache:
            return self.analogy_cache[cache_key]

        prompt = f"""Find detailed mappings between these domains:

        Source Domain: {source}
        Target Domain: {target}
        Context: {context.get('background', 'General comparison')}

        For each mapping:
        1. source_element: Element in source domain
        2. target_element: Corresponding element in target
        3. relationship: How they correspond
        4. mapping_type: structural/functional/causal
        5. strength: How strong is this mapping (0-1)
        
        Find at least 5 mappings. Be creative but accurate."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            mappings = self._parse_mappings(response["text"])

            # Cache result
            self.analogy_cache[cache_key] = mappings

            return mappings

        return []

    async def _generate_insights(self, mappings: List[Dict], domains: Dict) -> List[str]:
        """Generate insights from analogical mappings"""

        if not mappings:
            return ["No clear mappings found between domains"]

        # Format mappings for analysis
        mapping_text = "\n".join([
            f"- {m['source_element']} → {m['target_element']} ({m['relationship']})"
            for m in mappings[:10]
        ])

        prompt = f"""Based on these analogical mappings between {domains['source']} and {domains['target']}:

        {mapping_text}
        
        Generate insights:
        1. What can we learn about {domains['target']} from {domains['source']}?
        2. What patterns transfer between domains?
        3. What are the limitations of this analogy?
        4. What predictions can we make?
        
        Be specific and actionable."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            # Parse insights from response
            insights = self._extract_insights(response["text"])
            return insights

        return ["Analogy identified but insights unclear"]

    def _parse_mappings(self, text: str) -> List[Dict]:
        """Parse mappings from text response"""

        mappings = []
        lines = text.split('\n')

        current_mapping = {}

        for line in lines:
            line = line.strip()

            if not line:
                if current_mapping:
                    mappings.append(current_mapping)
                    current_mapping = {}
                continue

            # Parse different formats
            if '→' in line or '->' in line:
                # Format: source → target (relationship)
                parts = line.replace('→', '->').split('->')
                if len(parts) == 2:
                    source = parts[0].strip().strip('-').strip()

                    # Extract target and relationship
                    target_part = parts[1].strip()
                    if '(' in target_part and ')' in target_part:
                        target = target_part[:target_part.index('(')].strip()
                        relationship = target_part[target_part.index('(') + 1:target_part.index(')')].strip()
                    else:
                        target = target_part
                        relationship = "corresponds to"

                    current_mapping = {
                        "source_element": source,
                        "target_element": target,
                        "relationship": relationship,
                        "mapping_type": "structural",
                        "strength": 0.7
                    }

            # Look for strength indicators
            if "strong" in line.lower():
                if current_mapping:
                    current_mapping["strength"] = 0.9
            elif "weak" in line.lower():
                if current_mapping:
                    current_mapping["strength"] = 0.4

            # Look for mapping types
            if "structural" in line.lower():
                if current_mapping:
                    current_mapping["mapping_type"] = "structural"
            elif "functional" in line.lower():
                if current_mapping:
                    current_mapping["mapping_type"] = "functional"
            elif "causal" in line.lower():
                if current_mapping:
                    current_mapping["mapping_type"] = "causal"

        # Add last mapping
        if current_mapping:
            mappings.append(current_mapping)

        return mappings

    def _extract_insights(self, text: str) -> List[str]:
        """Extract insights from text"""

        insights = []
        lines = text.split('\n')

        current_insight = ""

        for line in lines:
            line = line.strip()

            # Look for numbered insights
            if line and (line[0].isdigit() or line.startswith('-')):
                if current_insight:
                    insights.append(current_insight)
                current_insight = line.lstrip('0123456789.-').strip()
            elif line and current_insight:
                # Continuation of current insight
                current_insight += " " + line

        # Add last insight
        if current_insight:
            insights.append(current_insight)

        # If no structured insights found, split by sentences
        if not insights:
            sentences = text.split('.')
            insights = [s.strip() for s in sentences if len(s.strip()) > 20]

        return insights[:5]  # Top 5 insights

    def _evaluate_analogy(self, mappings: List[Dict]) -> Dict:
        """Evaluate the quality of an analogy"""

        if not mappings:
            return {
                "score": 0.0,
                "strength": "none",
                "usefulness": "low"
            }

        # Calculate average mapping strength
        avg_strength = np.mean([m.get("strength", 0.5) for m in mappings])

        # Count mapping types
        type_counts = {}
        for m in mappings:
            mtype = m.get("mapping_type", "unknown")
            type_counts[mtype] = type_counts.get(mtype, 0) + 1

        # Evaluate based on multiple factors
        score = avg_strength

        # Bonus for multiple mapping types
        if len(type_counts) >= 2:
            score += 0.1

        # Bonus for many mappings
        if len(mappings) >= 5:
            score += 0.1

        # Penalty for only weak mappings
        if avg_strength < 0.5:
            score *= 0.7

        score = min(score, 1.0)

        # Determine strength category
        if score >= 0.8:
            strength = "strong"
        elif score >= 0.6:
            strength = "moderate"
        elif score >= 0.4:
            strength = "weak"
        else:
            strength = "very weak"

        # Determine usefulness
        if score >= 0.7 and len(mappings) >= 4:
            usefulness = "high"
        elif score >= 0.5 and len(mappings) >= 3:
            usefulness = "moderate"
        else:
            usefulness = "low"

        return {
            "score": score,
            "strength": strength,
            "usefulness": usefulness,
            "mapping_count": len(mappings),
            "avg_mapping_strength": avg_strength,
            "mapping_types": list(type_counts.keys())
        }

    async def _on_reasoning_request(self, data: Dict):
        """Handle analogical reasoning requests"""
        result = await self.process(data)

    def get_confidence(self) -> float:
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_confidence": self.confidence,
            "cache_size": len(self.analogy_cache)
        }