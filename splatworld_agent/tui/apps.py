"""TUI applications for long-running operations.

This module provides Textual-based TUI apps that replace Rich progress
context managers for inline terminal display.

CRITICAL: Do NOT import Rich console or use console.print() in TUI code.
Rich and Textual cannot mix - terminal state will be corrupted.
"""
from textual.app import App
from textual.widgets import Static, ProgressBar
from textual.containers import Vertical

from .results import GenerateResult


class GenerateTUI(App[GenerateResult]):
    """TUI for direct generation pipeline.

    Displays 3-stage progress inline under the command prompt:
    1. Enhancing prompt with taste profile
    2. Generating image
    3. Converting to 3D splat

    Returns GenerateResult with all pipeline output data.
    """

    # Inline mode settings - display under prompt, not fullscreen
    INLINE_PADDING = 0  # No blank line above app

    CSS = """
    Screen {
        height: auto;
        max-height: 15;
    }

    #status {
        height: 1;
        width: 100%;
    }

    #progress-container {
        height: auto;
        width: 100%;
    }

    ProgressBar {
        width: 100%;
    }
    """

    def __init__(
        self,
        prompt: str,
        manager,  # ProfileManager
        config,  # Config
        profile,  # TasteProfile
        provider: str,
        no_download: bool = False,
    ):
        """Initialize GenerateTUI.

        Args:
            prompt: User's original prompt text
            manager: ProfileManager for saving generations
            config: Config with API keys
            profile: TasteProfile for prompt enhancement
            provider: Image provider name (nano/gemini)
            no_download: Skip downloading splat file
        """
        super().__init__()
        self.prompt = prompt
        self.manager = manager
        self.config = config
        self.profile = profile
        self.provider = provider
        self.no_download = no_download

    def compose(self):
        """Create child widgets."""
        with Vertical(id="progress-container"):
            yield Static("Starting direct generation...", id="status")
            yield ProgressBar(total=3, show_eta=False)

    def on_mount(self):
        """Start the pipeline when app mounts."""
        self.run_pipeline()

    def run_pipeline(self):
        """Execute the direct generation pipeline.

        This method runs the 3-stage pipeline:
        1. Enhance prompt with taste profile
        2. Generate image with provider
        3. Convert to 3D with Marble

        Updates are pushed to the TUI via Static widget.
        On completion, calls self.exit(result) to return GenerateResult.
        """
        # Import dependencies inside method to avoid circular imports
        # and keep TUI module isolated from heavy deps at import time
        import base64
        from datetime import datetime
        from pathlib import Path
        import uuid

        from splatworld_agent.generators.manager import ProviderManager, ProviderFailureError
        from splatworld_agent.learning import enhance_prompt, PromptAdapter
        from splatworld_agent.core.marble import MarbleClient, MarbleTimeoutError, MarbleConversionError

        status = self.query_one("#status", Static)
        progress = self.query_one(ProgressBar)

        # Initialize tracking variables
        image_number = None
        flat_image_path = None
        enhanced_prompt = None
        reasoning = None
        modifications = []
        gen_name = self.provider
        result = None

        # Initialize API clients
        api_keys = {
            "nano": self.config.api_keys.nano or self.config.api_keys.google,
            "google": self.config.api_keys.google,
        }
        provider_manager = ProviderManager(
            api_keys=api_keys,
            initial_provider=gen_name,
        )
        adapter = PromptAdapter(api_key=self.config.api_keys.anthropic)
        marble = MarbleClient(api_key=self.config.api_keys.marble)

        try:
            # Stage 1/3: Enhance prompt
            status.update("[cyan]Stage 1/3:[/cyan] Enhancing prompt...")
            progress.update(progress=0)

            try:
                variant = adapter.generate_variant(self.prompt, self.profile)
                enhanced_prompt = variant.variant_prompt
                reasoning = variant.reasoning
                modifications = variant.modifications
            except Exception:
                # Fall back to basic enhancement if adapter fails
                enhanced_prompt = enhance_prompt(self.prompt, self.profile)
                reasoning = None
                modifications = []

            progress.update(progress=1)

            # Stage 2/3: Generate image
            status.update(f"[cyan]Stage 2/3:[/cyan] Generating image with {gen_name}...")

            try:
                image_bytes, gen_metadata = provider_manager.generate(enhanced_prompt)
                gen_name = gen_metadata["provider"]
            except ProviderFailureError as e:
                provider_manager.close()
                marble.close()
                self.exit(GenerateResult(
                    success=False,
                    error=f"Provider {e.provider} failed: {e.original_error}",
                    provider=gen_name,
                ))
                return

            # Save image IMMEDIATELY (prevents data loss on Marble timeout)
            image_number = self.manager.get_next_image_number()
            flat_image_path = self.manager.get_flat_image_path(image_number)
            self.manager.images_dir.mkdir(exist_ok=True)
            with open(flat_image_path, "wb") as f:
                f.write(image_bytes)

            progress.update(progress=2)
            status.update("[cyan]Stage 2/3:[/cyan] Image generated and saved!")

            # Stage 3/3: Convert to 3D
            status.update("[cyan]Stage 3/3:[/cyan] Converting to 3D...")

            try:
                result = marble.generate_and_wait(
                    image_base64=base64.b64encode(image_bytes).decode(),
                    mime_type="image/png",
                    display_name=f"Image {image_number}",
                    is_panorama=True,
                    on_progress=lambda s, d: status.update(f"[cyan]Stage 3/3:[/cyan] {d or s}"),
                )
            except (MarbleTimeoutError, MarbleConversionError) as e:
                # Image was saved - report partial success
                gen_id = f"direct-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                gen_timestamp = datetime.now()

                self.manager.save_image_metadata(image_number, {
                    "id": gen_id,
                    "image_number": image_number,
                    "prompt": self.prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "modifications": modifications if modifications else [],
                    "reasoning": reasoning if reasoning else "",
                    "timestamp": gen_timestamp.isoformat(),
                    "generator": gen_name,
                    "mode": "direct",
                    "marble_error": str(e),
                })
                self.manager.register_image(gen_id, image_number)

                provider_manager.close()
                marble.close()

                self.exit(GenerateResult(
                    success=False,
                    partial=True,
                    image_number=image_number,
                    image_path=str(flat_image_path),
                    enhanced_prompt=enhanced_prompt,
                    reasoning=reasoning,
                    modifications=modifications,
                    provider=gen_name,
                    error=f"3D conversion failed: {e}",
                ))
                return

            progress.update(progress=3)
            status.update("[green]Complete!")

            # Save full metadata
            gen_id = f"direct-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            gen_timestamp = datetime.now()

            self.manager.save_image_metadata(image_number, {
                "id": gen_id,
                "image_number": image_number,
                "prompt": self.prompt,
                "enhanced_prompt": enhanced_prompt,
                "modifications": modifications if modifications else [],
                "reasoning": reasoning if reasoning else "",
                "timestamp": gen_timestamp.isoformat(),
                "generator": gen_name,
                "mode": "direct",
                "world_id": result.world_id,
                "viewer_url": result.viewer_url,
            })
            self.manager.register_image(gen_id, image_number)

            # Download splat file
            splat_path = None
            if result.splat_url and not self.no_download and self.config.defaults.download_splats:
                self.manager.splats_dir.mkdir(exist_ok=True)
                splat_path_obj = self.manager.get_flat_splat_path(image_number)
                try:
                    marble.download_file(result.splat_url, splat_path_obj)
                    splat_path = str(splat_path_obj)
                except Exception:
                    splat_path = None

            provider_manager.close()
            marble.close()

            self.exit(GenerateResult(
                success=True,
                image_number=image_number,
                image_path=str(flat_image_path),
                splat_path=splat_path,
                viewer_url=result.viewer_url,
                enhanced_prompt=enhanced_prompt,
                reasoning=reasoning,
                modifications=modifications,
                provider=gen_name,
            ))

        except Exception as e:
            provider_manager.close()
            marble.close()

            self.exit(GenerateResult(
                success=False,
                image_number=image_number,
                image_path=str(flat_image_path) if flat_image_path else None,
                error=str(e),
                provider=gen_name,
            ))
