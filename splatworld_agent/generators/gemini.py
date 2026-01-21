"""
Gemini 2.0 Flash image generator for SplatWorld Agent.

Alternative to Nano Banana Pro. Uses the standard Gemini 2.0 Flash model
which is available on the free tier.
"""

import base64
import os
from typing import Optional

import httpx

from . import ImageGenerator


# Gemini 2.0 Flash API configuration
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


class GeminiGenerator(ImageGenerator):
    """
    Gemini 2.0 Flash image generator.

    Uses Google's Gemini 2.0 Flash model. Available on free tier.
    Good quality but not as high-res as Nano Banana Pro.
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
        Generate an image from a prompt.

        Args:
            prompt: The generation prompt
            seed: Optional random seed (not directly supported)

        Returns:
            Image bytes (PNG format)
        """
        # Build the full prompt with quality keywords
        full_prompt = f"high resolution detailed photograph: {prompt}"

        # Make API request
        response = self._client.post(
            f"{GEMINI_API_URL}?key={self.api_key}",
            json={
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                }
            },
            headers={"Content-Type": "application/json"},
        )

        if not response.is_success:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

        data = response.json()

        # Extract image from response
        if data.get("candidates") and data["candidates"][0].get("content", {}).get("parts"):
            for part in data["candidates"][0]["content"]["parts"]:
                if part.get("inlineData", {}).get("mimeType", "").startswith("image/"):
                    return base64.b64decode(part["inlineData"]["data"])

        # Check for error
        if data.get("error"):
            raise RuntimeError(f"Gemini API error: {data['error'].get('message', 'Unknown error')}")

        raise RuntimeError("No image found in Gemini API response")

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
