"""
Nano Banana Pro (Gemini 3 Pro Image) generator for SplatWorld Agent.

Nano Banana Pro is the codename for Google's Gemini 3 Pro Image model.
Uses the NANOBANANA_API_KEY environment variable (which is a Google API key).
"""

import base64
import os
from typing import Optional

import httpx

from . import ImageGenerator


# Gemini 3 Pro Image API configuration
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateContent"


class NanoGenerator(ImageGenerator):
    """
    Nano Banana Pro image generator.

    Uses Google's Gemini 3 Pro Image model for high-quality image generation.
    Best quality results, especially for panoramic/environmental images.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Nano generator.

        Args:
            api_key: Nano/Google API key. If not provided, reads from
                     NANOBANANA_API_KEY or GOOGLE_API_KEY env vars.
        """
        self.api_key = api_key or os.environ.get("NANOBANANA_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Nano API key not provided. Set NANOBANANA_API_KEY or GOOGLE_API_KEY "
                "environment variable or pass api_key parameter."
            )
        self._client = httpx.Client(timeout=120.0)

    def name(self) -> str:
        return "nano"

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        is_panorama: bool = True,
        aspect_ratio: str = "21:9",
    ) -> bytes:
        """
        Generate an image from a prompt.

        Args:
            prompt: The generation prompt
            seed: Optional random seed (not directly supported by Gemini, but included for interface)
            is_panorama: Whether to generate panoramic format (adds panoramic keywords)
            aspect_ratio: Aspect ratio for the image (default "21:9" for panoramic)

        Returns:
            Image bytes (PNG format)
        """
        # Build the full prompt
        if is_panorama:
            full_prompt = f"high resolution 4K ultra detailed equirectangular panoramic still of an empty scene with no humans: {prompt}"
        else:
            full_prompt = f"high resolution 4K ultra detailed photograph of a scene with no humans: {prompt}"

        # Make API request
        response = self._client.post(
            f"{GEMINI_API_URL}?key={self.api_key}",
            json={
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": "4K"
                    }
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

    def generate_standard(
        self,
        prompt: str,
        seed: Optional[int] = None,
        aspect_ratio: str = "16:9",
    ) -> bytes:
        """
        Generate a standard (non-panoramic) image.

        Args:
            prompt: The generation prompt
            seed: Optional random seed
            aspect_ratio: Aspect ratio (default "16:9")

        Returns:
            Image bytes (PNG format)
        """
        return self.generate(prompt, seed=seed, is_panorama=False, aspect_ratio=aspect_ratio)

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Backwards compatibility alias
ImagenGenerator = NanoGenerator
