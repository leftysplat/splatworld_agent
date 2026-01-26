"""
Provider manager for multi-provider image generation.

Implements Strategy pattern for provider selection with:
- Lazy loading of generators
- Retry logic with tenacity for transient failures
- Provider state tracking for mid-session switching (IGEN-04)
- Failure signaling for user consent flows (IGEN-02)
"""

from datetime import datetime
from typing import Optional, Tuple
import logging

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from ..models import ProviderState

logger = logging.getLogger(__name__)


class ProviderFailureError(Exception):
    """Raised when provider fails and fallback is available.

    IGEN-02 requires user consent for automatic fallback.
    Caller should catch this and prompt user before switching.
    """

    def __init__(self, provider: str, fallback_available: str, original_error: Exception):
        self.provider = provider
        self.fallback_available = fallback_available
        self.original_error = original_error
        super().__init__(f"{provider} failed: {original_error}")


class ProviderManager:
    """Manages image generation providers with failover and credit tracking.

    Implements Strategy pattern where the active provider can be switched
    at runtime without losing session state.

    Usage:
        manager = ProviderManager(api_keys={"nano": "...", "google": "..."})

        try:
            image_bytes, metadata = manager.generate("a cozy cabin")
        except ProviderFailureError as e:
            # Ask user for consent to switch
            if user_consents:
                manager.switch_provider(e.fallback_available)
                image_bytes, metadata = manager.generate("a cozy cabin")

        state = manager.get_state()
        if state.should_warn:
            # Ask user about switching to conserve credits
    """

    def __init__(
        self,
        api_keys: dict,
        initial_provider: str = "nano",  # IGEN-01: Nano is default
        credits_limit: Optional[int] = None,
    ):
        """
        Initialize the provider manager.

        Args:
            api_keys: Dict with keys "nano" and/or "google" for API keys
            initial_provider: Starting provider ("nano" or "gemini")
            credits_limit: Optional credit limit for usage warnings
        """
        self.api_keys = api_keys
        self.state = ProviderState(
            current_provider=initial_provider,
            credits_limit=credits_limit,
        )

        # Lazy-loaded generators
        self._nano = None
        self._gemini = None
        self._flux = None

    def _get_provider(self, provider_name: str):
        """Lazy-load and return the specified provider."""
        if provider_name == "nano":
            if self._nano is None:
                from .nano import NanoGenerator
                api_key = self.api_keys.get("nano") or self.api_keys.get("google")
                if not api_key:
                    raise ValueError("No API key available for Nano provider")
                self._nano = NanoGenerator(api_key=api_key)
            return self._nano
        elif provider_name == "flux":
            if self._flux is None:
                from .flux import FluxGenerator
                api_key = self.api_keys.get("bfl")
                if not api_key:
                    raise ValueError("No API key available for Flux provider")
                self._flux = FluxGenerator(api_key=api_key)
            return self._flux
        else:  # gemini
            if self._gemini is None:
                from .gemini import GeminiGenerator
                api_key = self.api_keys.get("google")
                if not api_key:
                    raise ValueError("No API key available for Gemini provider")
                self._gemini = GeminiGenerator(api_key=api_key)
            return self._gemini

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.NetworkError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_with_retry(
        self,
        provider,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> bytes:
        """Generate with automatic retry on transient network failures."""
        return provider.generate(prompt, seed=seed, **kwargs)

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[bytes, dict]:
        """
        Generate image with current provider.

        Args:
            prompt: The generation prompt
            seed: Optional random seed
            **kwargs: Additional args passed to generator (e.g., is_panorama)

        Returns:
            Tuple of (image_bytes, metadata_dict)

        Raises:
            ProviderFailureError: If provider fails and fallback available
                                  (caller should handle user consent)
            RuntimeError: If generation fails with no fallback
        """
        provider = self._get_provider(self.state.current_provider)

        try:
            image_bytes = self._generate_with_retry(provider, prompt, seed, **kwargs)

            # Update state
            self.state.generation_count += 1
            self.state.credits_used += 1  # Simplified: 1 credit per generation

            return image_bytes, {
                "provider": self.state.current_provider,
                "credits_used": self.state.credits_used,
                "generation_count": self.state.generation_count,
                "usage_percentage": self.state.usage_percentage,
            }

        except (httpx.HTTPStatusError, RuntimeError) as e:
            # IGEN-02: Signal failure with available fallback
            if self.state.current_provider == "nano":
                self.state.nano_failures += 1
                # Don't auto-switch - let caller handle user consent
                raise ProviderFailureError(
                    provider="nano",
                    fallback_available="gemini",
                    original_error=e
                )
            else:
                # No fallback available from gemini
                raise

    def switch_provider(self, new_provider: str, reason: str = "user_request") -> None:
        """
        Switch to a different provider (IGEN-04).

        Args:
            new_provider: Provider to switch to ("nano", "gemini", or "flux")
            reason: Why the switch happened (for tracking)
        """
        if new_provider not in ("nano", "gemini", "flux"):
            raise ValueError(f"Unknown provider: {new_provider}")

        old_provider = self.state.current_provider
        self.state.current_provider = new_provider
        self.state.provider_switches.append({
            "from": old_provider,
            "to": new_provider,
            "reason": reason,
            "at": datetime.now().isoformat(),
        })

    def get_state(self) -> ProviderState:
        """Get current provider state for display/persistence."""
        return self.state

    def set_credits_limit(self, limit: Optional[int]) -> None:
        """Set or update the credits limit."""
        self.state.credits_limit = limit

    def close(self):
        """Clean up resources."""
        if self._nano:
            self._nano.close()
        if self._gemini:
            self._gemini.close()
        if self._flux:
            self._flux.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
