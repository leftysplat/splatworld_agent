"""
Configuration management for SplatWorld Agent.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


GLOBAL_CONFIG_DIR = Path.home() / ".splatworld_agent"
GLOBAL_CONFIG_FILE = GLOBAL_CONFIG_DIR / "config.yaml"
PROJECT_DIR_NAME = ".splatworld"


@dataclass
class APIKeys:
    """API key configuration."""

    marble: str = ""
    nano: str = ""
    google: str = ""
    anthropic: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "APIKeys":
        return cls(
            marble=data.get("marble", ""),
            nano=data.get("nano", ""),
            google=data.get("google", ""),
            anthropic=data.get("anthropic", ""),
        )

    @classmethod
    def from_env(cls) -> "APIKeys":
        """Load API keys from environment variables."""
        return cls(
            marble=os.getenv("WORLDLABS_API_KEY", ""),
            nano=os.getenv("NANO_API_KEY", ""),
            google=os.getenv("GOOGLE_API_KEY", ""),
            anthropic=os.getenv("ANTHROPIC_API_KEY", ""),
        )

    def merge_env(self) -> "APIKeys":
        """Merge with environment variables (env takes precedence)."""
        env_keys = APIKeys.from_env()
        return APIKeys(
            marble=env_keys.marble or self.marble,
            nano=env_keys.nano or self.nano,
            google=env_keys.google or self.google,
            anthropic=env_keys.anthropic or self.anthropic,
        )


@dataclass
class Defaults:
    """Default settings."""

    image_generator: str = "nano"  # "nano" or "gemini"
    auto_learn_threshold: int = 10  # Run learning after this many feedback entries
    download_splats: bool = True
    download_meshes: bool = True
    exploration_mode: str = "explore"  # "explore" or "refine" (MODE-01/MODE-02)

    @classmethod
    def from_dict(cls, data: dict) -> "Defaults":
        return cls(
            image_generator=data.get("image_generator", "nano"),
            auto_learn_threshold=data.get("auto_learn_threshold", 10),
            download_splats=data.get("download_splats", True),
            download_meshes=data.get("download_meshes", True),
            exploration_mode=data.get("exploration_mode", "explore"),
        )


@dataclass
class Config:
    """Complete configuration."""

    api_keys: APIKeys = field(default_factory=APIKeys)
    defaults: Defaults = field(default_factory=Defaults)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file and environment."""
        config_path = config_path or GLOBAL_CONFIG_FILE

        # Start with defaults
        config = cls()

        # Load from file if exists
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
                config.api_keys = APIKeys.from_dict(data.get("api_keys", {}))
                config.defaults = Defaults.from_dict(data.get("defaults", {}))

        # Merge environment variables (they take precedence)
        config.api_keys = config.api_keys.merge_env()

        return config

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        config_path = config_path or GLOBAL_CONFIG_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "api_keys": {
                "marble": self.api_keys.marble,
                "nano": self.api_keys.nano,
                "google": self.api_keys.google,
                "anthropic": self.api_keys.anthropic,
            },
            "defaults": {
                "image_generator": self.defaults.image_generator,
                "auto_learn_threshold": self.defaults.auto_learn_threshold,
                "download_splats": self.defaults.download_splats,
                "download_meshes": self.defaults.download_meshes,
                "exploration_mode": self.defaults.exploration_mode,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.api_keys.marble:
            issues.append("Marble API key not configured (WORLDLABS_API_KEY)")

        if self.defaults.image_generator == "nano" and not self.api_keys.nano:
            issues.append("Nano API key not configured (NANO_API_KEY)")
        elif self.defaults.image_generator == "gemini" and not self.api_keys.google:
            issues.append("Google API key not configured (GOOGLE_API_KEY)")

        if not self.api_keys.anthropic:
            issues.append("Anthropic API key not configured (ANTHROPIC_API_KEY)")

        return issues


def get_project_dir(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find .splatworld directory in current or parent directories."""
    current = start_path or Path.cwd()

    while current != current.parent:
        project_dir = current / PROJECT_DIR_NAME
        if project_dir.exists():
            return project_dir
        current = current.parent

    return None


def ensure_global_config_dir() -> Path:
    """Ensure global config directory exists and return path."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return GLOBAL_CONFIG_DIR
