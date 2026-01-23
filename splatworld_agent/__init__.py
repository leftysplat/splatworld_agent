"""
SplatWorld Agent - Claude Code plugin for iterative 3D splat generation with taste learning.

The agent learns your aesthetic preferences over time and applies them to future generations.
"""

from pathlib import Path

# Read version from VERSION file (single source of truth)
_version_file = Path(__file__).parent.parent / "VERSION"
if _version_file.exists():
    __version__ = _version_file.read_text().strip()
else:
    __version__ = "0.1.0"  # Fallback for development

from splatworld_agent.models import TasteProfile, Generation, Feedback
from splatworld_agent.profile import ProfileManager
from splatworld_agent.config import Config

__all__ = [
    "__version__",
    "TasteProfile",
    "Generation",
    "Feedback",
    "ProfileManager",
    "Config",
]
