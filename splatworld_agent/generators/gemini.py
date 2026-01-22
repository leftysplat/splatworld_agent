"""
Imagen 3 image generator for SplatWorld Agent.

Alternative to Nano Banana Pro. Uses Google's Imagen 3 model
for image generation via the Gemini API.
"""

import base64
import os
from typing import Optional

import httpx

from . import ImageGenerator


# Imagen 3 API configuration
IMAGEN_MODEL = "imagen-3.0-generate-002"
IMAGEN_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGEN_MODEL}:predict"


class GeminiGenerator(ImageGenerator):
    """
    Imagen 3 image generator (via Gemini API).

    Uses Google's Imagen 3 model for image generation.
    Good quality, widely available.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini/Imagen generator.

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
        Generate an image from a prompt using Imagen 3.

        Args:
            prompt: The generation prompt
            seed: Optional random seed

        Returns:
            Image bytes (PNG format)
        """
        # Build the full prompt with quality keywords
        full_prompt = f"high resolution detailed photograph: {prompt}"

        # Build request payload for Imagen 3
        payload = {
            "instances": [{"prompt": full_prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "16:9",
            }
        }

        if seed is not None:
            payload["parameters"]["seed"] = seed

        # Make API request
        response = self._client.post(
            f"{IMAGEN_API_URL}?key={self.api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if not response.is_success:
            raise RuntimeError(f"Imagen API error {response.status_code}: {response.text}")

        data = response.json()

        # Extract image from response
        if data.get("predictions"):
            for prediction in data["predictions"]:
                if prediction.get("bytesBase64Encoded"):
                    return base64.b64decode(prediction["bytesBase64Encoded"])

        # Check for error
        if data.get("error"):
            raise RuntimeError(f"Imagen API error: {data['error'].get('message', 'Unknown error')}")

        raise RuntimeError("No image found in Imagen API response")

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
