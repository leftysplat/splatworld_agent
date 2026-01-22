"""Terminal inline image display with automatic protocol detection."""

import os
from pathlib import Path
from typing import Optional


class TerminalDisplay:
    """Handles terminal image display with automatic protocol detection."""

    def __init__(self):
        self._can_display_images = self._detect_capabilities()

    def _detect_capabilities(self) -> bool:
        """Detect if terminal supports inline images."""
        # Check for known terminals
        term_program = os.environ.get('TERM_PROGRAM', '')
        term = os.environ.get('TERM', '')

        # iTerm2
        if term_program == 'iTerm.app' or 'ITERM_SESSION_ID' in os.environ:
            return True

        # Kitty
        if term == 'xterm-kitty' or 'KITTY_WINDOW_ID' in os.environ:
            return True

        # WezTerm
        if term_program == 'WezTerm':
            return True

        return False

    def display_image(self, image_path: Path, max_width: Optional[int] = None) -> bool:
        """Display image inline if terminal supports it.

        Args:
            image_path: Path to image file
            max_width: Maximum width in terminal columns (default: 80)

        Returns:
            True if image was displayed inline, False if fallback needed.
        """
        if not self._can_display_images:
            return False

        try:
            from term_image.image import from_file

            image = from_file(str(image_path))

            # Set size constraints if needed
            if max_width:
                image.set_size(columns=max_width)

            # Draw to terminal (uses best available protocol)
            image.draw()

            return True
        except Exception:
            # Fallback silently if display fails
            return False

    @property
    def can_display_images(self) -> bool:
        """Check if terminal can display images inline."""
        return self._can_display_images


# Singleton instance
display = TerminalDisplay()
