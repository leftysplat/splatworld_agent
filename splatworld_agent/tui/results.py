"""Result types for TUI apps."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerateResult:
    """Result from GenerateTUI execution.

    Contains all output data from the direct generation pipeline:
    - success: Whether the entire pipeline succeeded
    - image_number: Sequential number for the generated image
    - image_path: Path to saved image file
    - splat_path: Path to saved splat file (if downloaded)
    - viewer_url: URL to view 3D splat in browser
    - enhanced_prompt: The taste-profile-enhanced prompt
    - reasoning: LLM reasoning for prompt modifications
    - modifications: List of modifications made to prompt
    - provider: Image generator provider used (nano/gemini)
    - error: Error message if failed
    - partial: True if image saved but Marble conversion failed
    """
    success: bool
    image_number: Optional[int] = None
    image_path: Optional[str] = None
    splat_path: Optional[str] = None
    viewer_url: Optional[str] = None
    enhanced_prompt: Optional[str] = None
    reasoning: Optional[str] = None
    modifications: Optional[list] = field(default_factory=list)
    provider: Optional[str] = None
    error: Optional[str] = None
    partial: bool = False  # True if image saved but Marble failed
