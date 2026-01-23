"""TUI components for long-running operations.

This module provides Textual-based TUI apps for displaying inline progress
during long-running operations like image generation and 3D conversion.

CRITICAL: Rich and Textual cannot mix in the same command execution.
Use Rich console.print() ONLY before or after TUI execution, never during.
"""
from .apps import GenerateTUI
from .results import GenerateResult

__all__ = ["GenerateTUI", "GenerateResult"]
