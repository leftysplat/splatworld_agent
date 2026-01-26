"""
FLUX.2 [pro] generator for SplatWorld Agent.

FLUX.2 [pro] is Black Forest Labs' highest-quality image generation model.
Uses the BFL_API_KEY environment variable for authentication.
"""

import os
import time
from typing import Optional

import httpx

from . import ImageGenerator


# BFL API configuration
BFL_API_URL = "https://api.bfl.ai/v1/flux-2-pro"
POLL_INTERVAL = 0.5  # seconds, per BFL documentation
MAX_POLL_DURATION = 120  # seconds timeout


class FluxGenerator(ImageGenerator):
    """
    FLUX.2 [pro] image generator.

    Uses Black Forest Labs' FLUX.2 [pro] model for highest-quality image generation.
    Best quality results, especially for panoramic/environmental images.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Flux generator.

        Args:
            api_key: BFL API key. If not provided, reads from
                     BFL_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("BFL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "BFL API key not provided. Set BFL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = httpx.Client(timeout=120.0)

    def name(self) -> str:
        return "flux"

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
            seed: Optional random seed for reproducibility
            is_panorama: Whether to generate panoramic format (adds panoramic keywords)
            aspect_ratio: Aspect ratio for the image (default "21:9" for panoramic)

        Returns:
            Image bytes (JPEG format)
        """
        # Build the full prompt
        if is_panorama:
            full_prompt = f"high resolution 4K ultra detailed equirectangular panoramic still of an empty scene with no humans: {prompt}"
        else:
            full_prompt = f"high resolution 4K ultra detailed photograph of a scene with no humans: {prompt}"

        # Step 1: Submit generation request
        request_data = {
            "prompt": full_prompt,
            "output_format": "jpeg",
            "safety_tolerance": 2,
            "aspect_ratio": aspect_ratio,
        }
        if seed is not None:
            request_data["seed"] = seed

        try:
            response = self._client.post(
                BFL_API_URL,
                json=request_data,
                headers={
                    "x-key": self.api_key,
                    "Content-Type": "application/json",
                },
            )

            # Handle specific error codes
            if response.status_code == 402:
                raise RuntimeError("Insufficient BFL credits")
            elif response.status_code == 429:
                raise RuntimeError("Rate limit exceeded")
            elif not response.is_success:
                raise RuntimeError(f"BFL API error {response.status_code}: {response.text}")

            data = response.json()
            polling_url = data.get("polling_url")
            if not polling_url:
                raise RuntimeError("No polling_url in BFL API response")

        except httpx.HTTPError as e:
            raise RuntimeError(f"BFL API request failed: {e}")

        # Step 2: Poll for completion
        max_attempts = int(MAX_POLL_DURATION / POLL_INTERVAL)
        backoff_delay = POLL_INTERVAL

        for attempt in range(max_attempts):
            time.sleep(backoff_delay)

            try:
                poll_response = self._client.get(polling_url)

                # Handle rate limiting during polling with exponential backoff
                if poll_response.status_code == 429:
                    backoff_delay = min(backoff_delay * 2, 10.0)
                    continue

                if not poll_response.is_success:
                    raise RuntimeError(f"Polling error {poll_response.status_code}: {poll_response.text}")

                poll_data = poll_response.json()
                status = poll_data.get("status")

                if status == "Ready":
                    # Step 3: Download image immediately (URL expires after 10 minutes)
                    result = poll_data.get("result", {})
                    image_url = result.get("sample")
                    if not image_url:
                        raise RuntimeError("No sample URL in Ready response")

                    image_response = self._client.get(image_url)
                    if not image_response.is_success:
                        raise RuntimeError(f"Failed to download image: {image_response.status_code}")

                    return image_response.content

                elif status in ("Failed", "Error"):
                    error_msg = poll_data.get("error", "Unknown error")
                    raise RuntimeError(f"Generation failed: {error_msg}")

                elif status in ("Request Moderated", "Content Moderated"):
                    raise RuntimeError("Generation was moderated by content filter")

                elif status == "Pending":
                    # Reset backoff on successful poll
                    backoff_delay = POLL_INTERVAL
                    continue

                else:
                    # Unknown status, continue polling
                    continue

            except httpx.HTTPError as e:
                raise RuntimeError(f"Polling request failed: {e}")

        # Step 4: Timeout
        raise RuntimeError("Generation timeout after 120 seconds")

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
