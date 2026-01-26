"""
Provider manager for multi-provider image generation.

Implements Strategy pattern for provider selection with:
- Lazy loading of generators
- Retry logic with tenacity for transient failures
- Automatic provider chain failover (FLUX -> Nano -> Gemini)
- Provider state tracking for mid-session switching (IGEN-04)
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


class ProviderManager:
    """Manages image generation providers with automatic failover and credit tracking.

    Implements Strategy pattern with automatic three-tier failover:
    1. FLUX.2 [pro] (if BFL_API_KEY configured) - PROV-01
    2. Nano Banana Pro (if available) - PROV-02
    3. Gemini (if GOOGLE_API_KEY configured) - PROV-03

    Failover is automatic and transparent - no user intervention required.

    Usage:
        manager = ProviderManager(api_keys={"bfl": "...", "nano": "...", "google": "..."})

        try:
            image_bytes, metadata = manager.generate("a cozy cabin")
            # Automatically tries FLUX -> Nano -> Gemini until success
            if metadata.get("failover_occurred"):
                print(f"Failed over to {metadata['provider']}")
        except RuntimeError as e:
            # Only raised if all providers fail
            print(f"Generation failed: {e}")

        state = manager.get_state()
        if state.should_warn:
            # Ask user about switching to conserve credits
    """

    def __init__(
        self,
        api_keys: dict,
        initial_provider: Optional[str] = None,  # Auto-determined from available keys
        credits_limit: Optional[int] = None,
    ):
        """
        Initialize the provider manager.

        Args:
            api_keys: Dict with keys "bfl", "nano", and/or "google" for API keys
            initial_provider: Starting provider (auto-determined if None)
            credits_limit: Optional credit limit for usage warnings
        """
        self.api_keys = api_keys

        # Determine initial provider from available keys (PROV-01: FLUX first if available)
        if initial_provider is None:
            chain = self._get_available_provider_chain()
            initial_provider = chain[0] if chain else "nano"

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

    def _get_available_provider_chain(self) -> list[str]:
        """Build provider chain based on configured API keys.

        Returns providers in priority order:
        - PROV-01: FLUX first if BFL_API_KEY configured
        - PROV-02: Nano second if NANO_API_KEY or GOOGLE_API_KEY configured
        - PROV-03: Gemini third if GOOGLE_API_KEY configured

        Returns:
            List of provider names in priority order
        """
        chain = []
        if self.api_keys.get("bfl"):
            chain.append("flux")
        if self.api_keys.get("nano") or self.api_keys.get("google"):
            chain.append("nano")
        if self.api_keys.get("google"):
            chain.append("gemini")
        return chain

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
        Generate image with automatic provider chain failover.

        Attempts generation with each provider in the chain (FLUX -> Nano -> Gemini)
        until success. Failover is automatic and transparent.

        Args:
            prompt: The generation prompt
            seed: Optional random seed
            **kwargs: Additional args passed to generator (e.g., is_panorama)

        Returns:
            Tuple of (image_bytes, metadata_dict)
            metadata_dict includes:
                - provider: Provider that succeeded
                - failover_occurred: True if not first provider in chain
                - attempted_providers: List of providers tried before success
                - credits_used, generation_count, usage_percentage

        Raises:
            RuntimeError: If all providers in chain fail
        """
        provider_chain = self._get_available_provider_chain()

        if not provider_chain:
            raise RuntimeError("No image generation providers available. Configure BFL_API_KEY, NANO_API_KEY, or GOOGLE_API_KEY.")

        attempted_providers = []
        last_error = None
        first_provider = provider_chain[0]

        for provider_name in provider_chain:
            attempted_providers.append(provider_name)

            try:
                provider = self._get_provider(provider_name)
                image_bytes = self._generate_with_retry(provider, prompt, seed, **kwargs)

                # Success! Update state
                failover_occurred = provider_name != first_provider
                if failover_occurred:
                    # Log failover event
                    logger.info(f"Successfully failed over to {provider_name} after {', '.join(attempted_providers[:-1])} failed")
                    # Track switch in state
                    self.state.provider_switches.append({
                        "from": attempted_providers[0] if attempted_providers else "none",
                        "to": provider_name,
                        "reason": "automatic_failover",
                        "at": datetime.now().isoformat(),
                    })

                # Update current provider to actual provider used
                self.state.current_provider = provider_name
                self.state.generation_count += 1
                self.state.credits_used += 1  # Simplified: 1 credit per generation

                # Track nano failures if nano was attempted and failed
                if "nano" in attempted_providers[:-1]:
                    self.state.nano_failures += 1

                return image_bytes, {
                    "provider": provider_name,
                    "failover_occurred": failover_occurred,
                    "attempted_providers": attempted_providers,
                    "credits_used": self.state.credits_used,
                    "generation_count": self.state.generation_count,
                    "usage_percentage": self.state.usage_percentage,
                }

            except (httpx.HTTPStatusError, RuntimeError, ValueError) as e:
                # Provider failed, log and try next
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}. Trying next provider in chain.")

                # Track nano failures immediately when nano fails
                if provider_name == "nano":
                    self.state.nano_failures += 1

                # Continue to next provider in chain
                continue

        # All providers failed
        raise RuntimeError(
            f"All providers failed. Attempted: {', '.join(attempted_providers)}. "
            f"Last error: {last_error}"
        )

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
