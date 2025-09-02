from google import genai
from typing import Dict, List, Optional
import asyncio
import json
from config.settings import settings
import structlog

logger = structlog.get_logger()


class GeminiService:
    """Core Gemini 2.0 Flash integration service"""

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

        # Different generation configs for different use cases
        self.configs = {
            "fast": genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=256,
                candidate_count=1
            ),
            "balanced": genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
                candidate_count=2
            ),
            "creative": genai.GenerationConfig(
                temperature=0.9,
                max_output_tokens=2048,
                candidate_count=3,
                top_p=0.95
            ),
            "structured": genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
                candidate_count=1,
                response_mime_type="application/json"
            )
        }

    async def generate(self,
                       prompt: str,
                       config_name: str = "balanced",
                       system_prompt: Optional[str] = None) -> Dict:
        """Generate response from Gemini"""

        try:
            # Build full prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            # Get config
            config = self.configs.get(config_name, self.configs["balanced"])

            # Generate asynchronously
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=config
            )

            return {
                "success": True,
                "text": response.text,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count
                }
            }

        except Exception as e:
            logger.error("gemini_generation_error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }

    async def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        """Generate structured JSON output"""

        structured_prompt = f"""{prompt}

        Output your response as valid JSON matching this schema:
        {json.dumps(schema, indent=2)}"""

        response = await self.generate(
            structured_prompt,
            config_name="structured"
        )

        if response["success"]:
            try:
                response["parsed"] = json.loads(response["text"])
            except json.JSONDecodeError:
                response["parsed"] = None

        return response