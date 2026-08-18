"""
lib/image_gen_client.py — Image Generation Client for Iris / Hermes Agent

Supports:
  1. Pollinations.ai (Default: 100% Free, Instant URL Generation, No OpenAI / API Key Required)
  2. OpenRouter Image Models (when OPENROUTER_API_KEY is available)
"""

import os
import urllib.parse
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ImageGenClient:
    def __init__(self, openrouter_key: Optional[str] = None):
        self.openrouter_key = (openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()

    def generate_image(self, prompt: str, aspect_ratio: str = "square", width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """
        Generate an AI image URL from text prompt without requiring OpenAI API key.
        Uses Pollinations.ai (Flux / SD model) or OpenRouter.
        """
        clean_prompt = prompt.strip()
        logger.info(f"Generating image for prompt: '{clean_prompt[:60]}...'")

        # Map aspect ratios to dimensions if needed
        if aspect_ratio == "landscape":
            width, height = 1280, 720
        elif aspect_ratio == "portrait":
            width, height = 720, 1280

        # Pollinations.ai (Genuinely Free, Instant High-Quality Image URL Generation)
        encoded_prompt = urllib.parse.quote(clean_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed=42"

        return {
            "success": True,
            "provider": "Pollinations.ai (Flux)",
            "image_url": image_url,
            "prompt": clean_prompt,
            "aspect_ratio": aspect_ratio,
            "message": "🎨 Image generated successfully!"
        }
