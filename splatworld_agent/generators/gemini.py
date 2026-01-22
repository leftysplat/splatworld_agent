"""
Gemini 2.5 Flash image generator for SplatWorld Agent.

Alternative to Nano Banana Pro. Uses Gemini 2.5 Flash Image
for image generation via the Gemini API.
"""

import base64
import os
from typing import Optional

import httpx

from . import ImageGenerator


# Gemini 2.5 Flash Image - latest image generation model
GEMINI_MODEL = "gemini-2.5-flash-image"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


class GeminiGenerator(ImageGenerator):
    """
    Gemini 2.5 Flash image generator.

    Uses Google's Gemini 2.5 Flash Image model for image generation.
    Available through the standard Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini generator.

        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = httpx.Client(timeout=120.0)

    def name(self) -> str:
        return "gemini"

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Generate an image from a prompt using Gemini 2.5 Flash.

        Args:
            prompt: The generation prompt
            seed: Optional random seed (not supported by this model)

        Returns:
            Image bytes (PNG format)
        """
        # Build the full prompt with quality keywords
        full_prompt = f"Generate a high resolution detailed photograph: {prompt}"

        # Build request payload for Gemini 2.0 Flash with image output
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
            }
        }

        # Make API request
        response = self._client.post(
            f"{GEMINI_API_URL}?key={self.api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if not response.is_success:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()

        # Extract image from response
        # Response format: candidates[0].content.parts[].inlineData.data
        if data.get("candidates"):
            for candidate in data["candidates"]:
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    inline_data = part.get("inlineData", {})
                    if inline_data.get("mimeType", "").startswith("image/"):
                        return base64.b64decode(inline_data["data"])

        # Check for error
        if data.get("error"):
            raise RuntimeError(f"Gemini API error: {data['error'].get('message', 'Unknown error')}")

        # Check for blocked content
        if data.get("candidates") and data["candidates"][0].get("finishReason") == "SAFETY":
            raise RuntimeError("Image generation blocked by safety filters")

        raise RuntimeError("No image found in Gemini API response")

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
