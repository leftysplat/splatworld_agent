"""
Image generators for SplatWorld Agent.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ImageGenerator(ABC):
    """Abstract base class for image generators."""

    @abstractmethod
    def generate(self, prompt: str, seed: Optional[int] = None) -> bytes:
        """Generate an image from a prompt.

        Args:
            prompt: The generation prompt
            seed: Optional random seed for reproducibility

        Returns:
            Image bytes (PNG or JPEG)
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the generator name."""
        pass
