"""
Updated Gemini Service for new google-genai library
Uses the new Client-based API (google-genai v0.2.0+)
"""
from google import genai
from google.genai import types
from typing import Dict, List, Optional
import asyncio
import json
from config.settings import settings
import structlog

logger = structlog.get_logger()


class GeminiService:
    """Core Gemini 2.0 Flash integration service"""

    def __init__(self):
        # New API: Client automatically uses GEMINI_API_KEY environment variable
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model

        # Different generation configs for different use cases
        self.configs = {
            "fast": {
                "temperature": 0.3,
                "max_output_tokens": 256,
                "candidate_count": 1
            },
            "balanced": {
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "candidate_count": 1  # New API only supports 1 candidate
            },
            "creative": {
                "temperature": 0.9,
                "max_output_tokens": 2048,
                "candidate_count": 1,
                "top_p": 0.95
            },
            "structured": {
                "temperature": 0.1,
                "max_output_tokens": 1024,
                "candidate_count": 1,
                "response_mime_type": "application/json"
            }
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
            config_dict = self.configs.get(config_name, self.configs["balanced"])

            # Create GenerateContentConfig
            config = types.GenerateContentConfig(
                temperature=config_dict.get("temperature", 0.7),
                max_output_tokens=config_dict.get("max_output_tokens", 1024),
                top_p=config_dict.get("top_p"),
                response_mime_type=config_dict.get("response_mime_type")
            )

            # Generate using new API
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=full_prompt,
                config=config
            )

            # Extract usage metadata
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_meta = response.usage_metadata
                usage["prompt_tokens"] = getattr(usage_meta, 'prompt_token_count', 0)
                usage["completion_tokens"] = getattr(usage_meta, 'candidates_token_count', 0)

            return {
                "success": True,
                "text": response.text,
                "usage": usage
            }

        except Exception as e:
            logger.error("gemini_generation_error", error=str(e), error_type=type(e).__name__)
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
                # Try to parse JSON from response
                text = response["text"].strip()

                # Handle markdown code blocks
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"):
                    text = text.replace("```", "").strip()

                response["parsed"] = json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning("json_parse_error", error=str(e), text=response["text"][:200])
                response["parsed"] = None

        return response
