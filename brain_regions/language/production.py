from typing import Dict, Any, List
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class LanguageProduction(BrainRegion):
    """Broca's area - language generation and production"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.production_templates = {
            "explanation": "Explain {concept} in clear, simple terms.",
            "comparison": "Compare and contrast {item1} and {item2}.",
            "summary": "Summarize the following: {content}",
            "elaboration": "Elaborate on {topic} with examples."
        }
        self.state = {}

    async def initialize(self):
        """Initialize language production module"""
        logger.info("initializing_language_production")

        # Subscribe to requests for language generation
        self.event_bus.subscribe("generate_response", self._on_generate_request)
        self.event_bus.subscribe("global_workspace_broadcast", self._on_global_broadcast)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate language output"""

        intent = input_data.get("intent", "explain")
        content = input_data.get("content", {})
        style = input_data.get("style", "clear")

        # Generate appropriate language
        response = await self._generate_language(intent, content, style)

        # Update state
        self.state = {
            "last_production": response["text"][:100],
            "intent": intent,
            "success": response["success"]
        }

        # Emit completion event
        await self.event_bus.emit("language_production_complete", response)

        return response

    async def _generate_language(self, intent: str, content: Dict, style: str) -> Dict:
        """Generate language based on intent and content"""

        # Build prompt based on intent
        if intent == "explain":
            prompt = self._build_explanation_prompt(content, style)
        elif intent == "summarize":
            prompt = self._build_summary_prompt(content, style)
        elif intent == "respond":
            prompt = self._build_response_prompt(content, style)
        else:
            prompt = self._build_general_prompt(content, style)

        # Use Gemini to generate
        response = await self.gemini.generate(
            prompt,
            config_name="balanced" if style == "detailed" else "fast"
        )

        if response["success"]:
            # Post-process for clarity
            processed_text = self._post_process(response["text"], style)

            return {
                "success": True,
                "text": processed_text,
                "intent": intent,
                "style": style
            }

        return {
            "success": False,
            "text": "Unable to generate appropriate response.",
            "error": response.get("error", "Generation failed")
        }

    def _build_explanation_prompt(self, content: Dict, style: str) -> str:
        """Build prompt for explanations"""
        topic = content.get("topic", "the concept")
        context = content.get("context", "")

        style_instructions = {
            "simple": "Use simple language suitable for a general audience.",
            "technical": "Use precise technical terminology.",
            "detailed": "Provide comprehensive detail with examples.",
            "concise": "Be brief and to the point."
        }

        return f"""Explain {topic}.

        Context: {context}
        
        {style_instructions.get(style, '')}
        
        Structure your explanation clearly with:
        1. Core concept definition
        2. Key components or aspects
        3. Practical implications or examples"""

    def _build_summary_prompt(self, content: Dict, style: str) -> str:
        """Build prompt for summaries"""
        text = content.get("text", "")
        focus = content.get("focus", "main points")

        return f"""Summarize the following text, focusing on {focus}:

        {text}
        
        Style: {style}
        Length: {"1-2 sentences" if style == "concise" else "1-2 paragraphs"}"""

    def _build_response_prompt(self, content: Dict, style: str) -> str:
        """Build prompt for responses"""
        query = content.get("query", "")
        context = content.get("context", {})

        return f"""Respond to this query: {query}

        Context: {context}
        
        Provide a {style} response that directly addresses the question."""

    def _build_general_prompt(self, content: Dict, style: str) -> str:
        """Build general purpose prompt"""
        return f"""Generate text based on: {content}

        Style: {style}"""

    def _post_process(self, text: str, style: str) -> str:
        """Post-process generated text for style consistency"""

        # Remove any prompt artifacts
        text = text.strip()

        # Style-specific processing
        if style == "concise":
            # Take only first paragraph or 3 sentences
            sentences = text.split('. ')
            if len(sentences) > 3:
                text = '. '.join(sentences[:3]) + '.'

        return text

    async def _on_generate_request(self, data: Dict):
        """Handle generation requests"""
        result = await self.process(data)

    async def _on_global_broadcast(self, data: Dict):
        """React to global workspace broadcasts"""
        integrated_state = data.get("integrated_state", {})

        # Check if language output is needed
        if integrated_state.get("action_implications"):
            for action in integrated_state["action_implications"]:
                if "explain" in action.lower() or "describe" in action.lower():
                    await self.process({
                        "intent": "explain",
                        "content": {"topic": integrated_state.get("primary_focus", {})},
                        "style": "clear"
                    })

    def get_state(self) -> Dict[str, Any]:
        return self.state