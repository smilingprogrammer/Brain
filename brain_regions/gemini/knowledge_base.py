from typing import Dict, Any, List, Optional
import json
from brain_regions.gemini.gemini_service import GeminiService
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class KnowledgeBase:
    """Gemini-powered semantic knowledge base"""

    def __init__(self, gemini_service: GeminiService, event_bus: EventBus):
        self.gemini = gemini_service
        self.event_bus = event_bus

        # Knowledge cache
        self.fact_cache = {}
        self.inference_cache = {}

        # Knowledge domains
        self.domains = [
            "science", "mathematics", "history", "culture",
            "technology", "philosophy", "psychology", "biology"
        ]

    async def initialize(self):
        """Initialize knowledge base"""
        logger.info("initializing_gemini_knowledge_base")

        # Subscribe to knowledge queries
        self.event_bus.subscribe("knowledge_query", self._on_knowledge_query)
        self.event_bus.subscribe("fact_check", self._on_fact_check)

    async def query(self, query: str, domain: Optional[str] = None) -> Dict:
        """Query knowledge base"""

        # Check cache
        cache_key = f"{domain}:{query}" if domain else query
        if cache_key in self.fact_cache:
            return self.fact_cache[cache_key]

        # Build knowledge query
        prompt = self._build_knowledge_prompt(query, domain)

        # Query Gemini
        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            result = {
                "success": True,
                "query": query,
                "domain": domain,
                "knowledge": response["text"],
                "structured": await self._extract_structured_knowledge(response["text"])
            }

            # Cache result
            self.fact_cache[cache_key] = result

            return result

        return {
            "success": False,
            "query": query,
            "error": "Knowledge retrieval failed"
        }

    async def infer(self, premises: List[str], query: str) -> Dict:
        """Make inferences from premises"""

        # Check inference cache
        cache_key = f"{':'.join(sorted(premises))}:{query}"
        if cache_key in self.inference_cache:
            return self.inference_cache[cache_key]

        prompt = f"""Given these premises:
{chr(10).join(f'- {p}' for p in premises)}

Question: {query}

Provide:
1. Direct inference from the premises
2. Confidence level (0-1)
3. Reasoning steps
4. Any assumptions made"""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            result = {
                "success": True,
                "premises": premises,
                "query": query,
                "inference": self._parse_inference(response["text"])
            }

            # Cache result
            self.inference_cache[cache_key] = result

            return result

        return {
            "success": False,
            "error": "Inference failed"
        }

    async def check_fact(self, statement: str) -> Dict:
        """Fact-check a statement"""

        prompt = f"""Fact-check this statement:
        "{statement}"
        
        Provide:
        1. Verdict: TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE
        2. Explanation
        3. Confidence (0-1)
        4. Supporting evidence or corrections"""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "verdict": "string",
                "explanation": "string",
                "confidence": "float",
                "evidence": ["string"]
            }
        )

        if response["success"] and response["parsed"]:
            return {
                "success": True,
                "statement": statement,
                "fact_check": response["parsed"]
            }

        return {
            "success": False,
            "statement": statement,
            "error": "Fact check failed"
        }

    async def get_related_concepts(self, concept: str, n: int = 5) -> List[str]:
        """Get related concepts"""

        prompt = f"""List {n} concepts closely related to "{concept}".
        Include different types of relationships:
        - Hierarchical (broader/narrower)
        - Associative (commonly linked)
        - Causal (cause/effect)
        - Functional (similar use/purpose)
        
        Format: one concept per line"""

        response = await self.gemini.generate(prompt, config_name="fast")

        if response["success"]:
            # Parse concepts
            lines = response["text"].strip().split('\n')
            concepts = []

            for line in lines:
                # Clean and extract concept
                cleaned = line.strip().lstrip('•-123456789. ')
                if cleaned and len(cleaned) > 2:
                    concepts.append(cleaned)

            return concepts[:n]

        return []

    def _build_knowledge_prompt(self, query: str, domain: Optional[str]) -> str:
        """Build knowledge retrieval prompt"""

        base_prompt = f"""Provide comprehensive knowledge about: {query}"""

        if domain:
            base_prompt += f"\nFocus on the {domain} perspective."

        base_prompt += """

Include:
1. Definition or core concept
2. Key facts and properties
3. Important relationships
4. Common applications or implications
5. Any important caveats or limitations

Be accurate and informative."""

        return base_prompt

    async def _extract_structured_knowledge(self, text: str) -> Dict:
        """Extract structured knowledge from text"""

        prompt = f"""Extract structured knowledge from this text:

{text[:1000]}

Create a knowledge structure with:
- main_concept: The primary concept
- definition: Clear definition
- properties: List of key properties
- relationships: List of related concepts
- applications: Practical applications

Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "main_concept": "string",
                "definition": "string",
                "properties": ["string"],
                "relationships": ["string"],
                "applications": ["string"]
            }
        )

        if response["success"] and response["parsed"]:
            return response["parsed"]

        return {}

    def _parse_inference(self, text: str) -> Dict:
        """Parse inference from text"""

        inference = {
            "conclusion": "",
            "confidence": 0.7,
            "steps": [],
            "assumptions": []
        }

        lines = text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            if "inference" in line.lower() or "conclusion" in line.lower():
                current_section = "conclusion"
            elif "confidence" in line.lower():
                # Extract confidence
                import re
                numbers = re.findall(r'0?\.\d+|1\.0', line)
                if numbers:
                    inference["confidence"] = float(numbers[0])
            elif "step" in line.lower() or "reasoning" in line.lower():
                current_section = "steps"
            elif "assumption" in line.lower():
                current_section = "assumptions"
            elif current_section:
                # Add to current section
                if current_section == "conclusion":
                    inference["conclusion"] += line + " "
                elif current_section == "steps":
                    if line.strip():
                        inference["steps"].append(line.strip('•-123456789. '))
                elif current_section == "assumptions":
                    if line.strip():
                        inference["assumptions"].append(line.strip('•-123456789. '))

        # Clean conclusion
        inference["conclusion"] = inference["conclusion"].strip()

        return inference

    async def _on_knowledge_query(self, data: Dict):
        """Handle knowledge query requests"""

        query = data.get("query", "")
        domain = data.get("domain")

        result = await self.query(query, domain)

        # Emit result
        await self.event_bus.emit("knowledge_retrieved", result)

    async def _on_fact_check(self, data: Dict):
        """Handle fact check requests"""

        statement = data.get("statement", "")

        result = await self.check_fact(statement)

        # Emit result
        await self.event_bus.emit("fact_check_complete", result)