"""TUI applications for long-running operations.

This module provides Textual-based TUI apps that replace Rich progress
context managers for inline terminal display.

CRITICAL: Do NOT import Rich console or use console.print() in TUI code.
Rich and Textual cannot mix - terminal state will be corrupted.
"""
from textual.app import App
from textual.widgets import Static
from textual.containers import Vertical
from textual.worker import get_current_worker
from textual import work

from .results import GenerateResult
from .widgets import StageProgress, ResourcePanel, OutputLog


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
        max-height: 20;
    }

    #resource-panel {
        height: 1;
        width: 100%;
        background: $surface;
    }

    #progress-container {
        height: auto;
        width: 100%;
    }

    #progress {
        height: auto;
        width: 100%;
    }

    #current-op {
        height: 1;
        width: 100%;
        color: $text-muted;
    }

    #output-log {
        height: 4;
        width: 100%;
        border-top: solid $primary;
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
        yield ResourcePanel(id="resource-panel")
        with Vertical(id="progress-container"):
            yield StageProgress(total_stages=3, id="progress")
            yield Static("Starting direct generation...", id="current-op")
        yield OutputLog(id="output-log")

    def on_mount(self):
        """Start the pipeline when app mounts."""
        self.run_pipeline()

    @work(thread=True, exclusive=True)
    def run_pipeline(self):
        """Execute the direct generation pipeline in a background thread.

        This method runs the 3-stage pipeline:
        1. Enhance prompt with taste profile
        2. Generate image with provider
        3. Convert to 3D with Marble

        Uses @work(thread=True) for non-blocking execution.
        All UI updates go through call_from_thread for thread safety.
        Checks worker.is_cancelled between stages for graceful cancellation.
        """
        worker = get_current_worker()

        # Import dependencies inside method to avoid circular imports
        # and keep TUI module isolated from heavy deps at import time
        import base64
        from datetime import datetime
        import uuid

        from splatworld_agent.generators.manager import ProviderManager, ProviderFailureError
        from splatworld_agent.learning import enhance_prompt, PromptAdapter
        from splatworld_agent.core.marble import MarbleClient, MarbleTimeoutError, MarbleConversionError

        # Initialize tracking variables
        image_number = None
        flat_image_path = None
        enhanced_prompt = None
        reasoning = None
        modifications = []
        gen_name = self.provider
        marble_result = None

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
            self.call_from_thread(self._update_stage, 1, "running", "Enhancing prompt...")
            self.call_from_thread(self._update_current_op, "Generating variant with taste profile")

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

            self.call_from_thread(self._update_stage, 1, "complete", "Prompt enhanced")

            # Check cancellation before expensive operation
            if worker.is_cancelled:
                self._cleanup_and_exit(provider_manager, marble, GenerateResult(
                    success=False, error="Cancelled by user"
                ))
                return

            # Stage 2/3: Generate image
            self.call_from_thread(self._update_stage, 2, "running", f"Generating with {gen_name}...")
            self.call_from_thread(self._update_current_op, f"Generating image with {gen_name}")

            try:
                image_bytes, gen_metadata = provider_manager.generate(enhanced_prompt)
                gen_name = gen_metadata["provider"]
            except ProviderFailureError as e:
                self.call_from_thread(self._update_stage, 2, "error", str(e))
                self._cleanup_and_exit(provider_manager, marble, GenerateResult(
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

            self.call_from_thread(self._update_stage, 2, "complete", "Image saved")

            # Check cancellation - image is saved, can exit gracefully
            if worker.is_cancelled:
                self._cleanup_and_exit(provider_manager, marble, GenerateResult(
                    success=False,
                    partial=True,
                    image_number=image_number,
                    image_path=str(flat_image_path),
                    error="Cancelled before 3D conversion"
                ))
                return

            # Stage 3/3: Convert to 3D
            self.call_from_thread(self._update_stage, 3, "running", "Converting to 3D...")
            self.call_from_thread(self._update_current_op, "Converting with Marble AI")

            # Marble progress callback - also needs call_from_thread
            def on_marble_progress(status, description):
                self.call_from_thread(self._update_current_op, description or status)

            try:
                marble_result = marble.generate_and_wait(
                    image_base64=base64.b64encode(image_bytes).decode(),
                    mime_type="image/png",
                    display_name=f"Image {image_number}",
                    is_panorama=True,
                    on_progress=on_marble_progress,
                )
            except (MarbleTimeoutError, MarbleConversionError) as e:
                # Image was saved - report partial success
                self.call_from_thread(self._update_stage, 3, "error", str(e))

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

                self._cleanup_and_exit(provider_manager, marble, GenerateResult(
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

            self.call_from_thread(self._update_stage, 3, "complete", "Conversion complete")
            self.call_from_thread(self._stop_timer)
            self.call_from_thread(self._update_current_op, "Complete!")

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
                "world_id": marble_result.world_id,
                "viewer_url": marble_result.viewer_url,
            })
            self.manager.register_image(gen_id, image_number)

            # Download splat file
            splat_path = None
            if marble_result.splat_url and not self.no_download and self.config.defaults.download_splats:
                self.manager.splats_dir.mkdir(exist_ok=True)
                splat_path_obj = self.manager.get_flat_splat_path(image_number)
                try:
                    marble.download_file(marble_result.splat_url, splat_path_obj)
                    splat_path = str(splat_path_obj)
                except Exception:
                    splat_path = None

            self._cleanup_and_exit(provider_manager, marble, GenerateResult(
                success=True,
                image_number=image_number,
                image_path=str(flat_image_path),
                splat_path=splat_path,
                viewer_url=marble_result.viewer_url,
                enhanced_prompt=enhanced_prompt,
                reasoning=reasoning,
                modifications=modifications,
                provider=gen_name,
            ))

        except Exception as e:
            self._cleanup_and_exit(provider_manager, marble, GenerateResult(
                success=False,
                image_number=image_number,
                image_path=str(flat_image_path) if flat_image_path else None,
                error=str(e),
                provider=gen_name,
            ))

    def _update_stage(self, stage: int, status: str, description: str = ""):
        """Update stage progress (called from main thread via call_from_thread)."""
        progress = self.query_one("#progress", StageProgress)
        progress.set_stage(stage, status, description)

    def _update_current_op(self, text: str):
        """Update current operation text (called from main thread via call_from_thread)."""
        current_op = self.query_one("#current-op", Static)
        current_op.update(text)

    def _stop_timer(self):
        """Stop the elapsed time timer (called from main thread via call_from_thread)."""
        progress = self.query_one("#progress", StageProgress)
        progress.stop_timer()

    def _cleanup_and_exit(self, provider_manager, marble, result: GenerateResult):
        """Clean up resources and exit with result."""
        provider_manager.close()
        marble.close()
        self.call_from_thread(self._stop_timer)
        self.call_from_thread(self.exit, result)
