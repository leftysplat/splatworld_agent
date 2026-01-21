"""
SplatWorld Agent - Claude Code plugin for iterative 3D splat generation with taste learning.

The agent learns your aesthetic preferences over time and applies them to future generations.
"""

__version__ = "0.1.2"

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
