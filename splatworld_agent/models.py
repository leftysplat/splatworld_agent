"""
Data models for SplatWorld Agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


@dataclass
class StylePreference:
    """A single style preference with confidence score."""

    preference: str = ""
    avoid: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "preference": self.preference,
            "avoid": self.avoid,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StylePreference":
        return cls(
            preference=data.get("preference", ""),
            avoid=data.get("avoid", ""),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class VisualStyle:
    """Visual style preferences."""

    lighting: StylePreference = field(default_factory=StylePreference)
    color_palette: StylePreference = field(default_factory=StylePreference)
    mood: StylePreference = field(default_factory=StylePreference)

    def to_dict(self) -> dict:
        return {
            "lighting": self.lighting.to_dict(),
            "color_palette": self.color_palette.to_dict(),
            "mood": self.mood.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VisualStyle":
        return cls(
            lighting=StylePreference.from_dict(data.get("lighting", {})),
            color_palette=StylePreference.from_dict(data.get("color_palette", {})),
            mood=StylePreference.from_dict(data.get("mood", {})),
        )


@dataclass
class CompositionPrefs:
    """Composition preferences."""

    density: StylePreference = field(default_factory=StylePreference)
    perspective: StylePreference = field(default_factory=StylePreference)
    foreground: StylePreference = field(default_factory=StylePreference)

    def to_dict(self) -> dict:
        return {
            "density": self.density.to_dict(),
            "perspective": self.perspective.to_dict(),
            "foreground": self.foreground.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompositionPrefs":
        return cls(
            density=StylePreference.from_dict(data.get("density", {})),
            perspective=StylePreference.from_dict(data.get("perspective", {})),
            foreground=StylePreference.from_dict(data.get("foreground", {})),
        )


@dataclass
class DomainPrefs:
    """Domain/environment preferences."""

    environments: list[str] = field(default_factory=list)
    avoid_environments: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "environments": self.environments,
            "avoid_environments": self.avoid_environments,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainPrefs":
        return cls(
            environments=data.get("environments", []),
            avoid_environments=data.get("avoid_environments", []),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class QualityCriteria:
    """Quality criteria - must-haves and never-haves."""

    must_have: list[str] = field(default_factory=list)
    never: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "must_have": self.must_have,
            "never": self.never,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QualityCriteria":
        return cls(
            must_have=data.get("must_have", []),
            never=data.get("never", []),
        )


@dataclass
class Exemplar:
    """A reference image (positive or negative)."""

    path: str
    added: datetime
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "added": self.added.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Exemplar":
        return cls(
            path=data["path"],
            added=datetime.fromisoformat(data["added"]),
            notes=data.get("notes", ""),
        )


@dataclass
class ProfileStats:
    """Statistics about profile usage."""

    total_generations: int = 0
    feedback_count: int = 0
    love_count: int = 0
    hate_count: int = 0

    def to_dict(self) -> dict:
        return {
            "total_generations": self.total_generations,
            "feedback_count": self.feedback_count,
            "love_count": self.love_count,
            "hate_count": self.hate_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileStats":
        return cls(
            total_generations=data.get("total_generations", 0),
            feedback_count=data.get("feedback_count", 0),
            love_count=data.get("love_count", 0),
            hate_count=data.get("hate_count", 0),
        )


@dataclass
class TasteProfile:
    """The complete taste profile for a project."""

    version: str = "1.0"
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)

    visual_style: VisualStyle = field(default_factory=VisualStyle)
    composition: CompositionPrefs = field(default_factory=CompositionPrefs)
    domain: DomainPrefs = field(default_factory=DomainPrefs)
    quality: QualityCriteria = field(default_factory=QualityCriteria)

    exemplars: list[Exemplar] = field(default_factory=list)
    anti_exemplars: list[Exemplar] = field(default_factory=list)

    stats: ProfileStats = field(default_factory=ProfileStats)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "visual_style": self.visual_style.to_dict(),
            "composition": self.composition.to_dict(),
            "domain": self.domain.to_dict(),
            "quality": self.quality.to_dict(),
            "exemplars": [e.to_dict() for e in self.exemplars],
            "anti_exemplars": [e.to_dict() for e in self.anti_exemplars],
            "stats": self.stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TasteProfile":
        return cls(
            version=data.get("version", "1.0"),
            created=datetime.fromisoformat(data["created"]) if "created" in data else datetime.now(),
            updated=datetime.fromisoformat(data["updated"]) if "updated" in data else datetime.now(),
            visual_style=VisualStyle.from_dict(data.get("visual_style", {})),
            composition=CompositionPrefs.from_dict(data.get("composition", {})),
            domain=DomainPrefs.from_dict(data.get("domain", {})),
            quality=QualityCriteria.from_dict(data.get("quality", {})),
            exemplars=[Exemplar.from_dict(e) for e in data.get("exemplars", [])],
            anti_exemplars=[Exemplar.from_dict(e) for e in data.get("anti_exemplars", [])],
            stats=ProfileStats.from_dict(data.get("stats", {})),
        )

    def save(self, path: Path) -> None:
        """Save profile to JSON file."""
        self.updated = datetime.now()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TasteProfile":
        """Load profile from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_prompt_context(self) -> str:
        """Generate prompt context from profile for injection into generation prompts."""
        parts = []

        # Visual style
        if self.visual_style.lighting.preference:
            parts.append(f"Lighting: {self.visual_style.lighting.preference}")
        if self.visual_style.color_palette.preference:
            parts.append(f"Color palette: {self.visual_style.color_palette.preference}")
        if self.visual_style.mood.preference:
            parts.append(f"Mood: {self.visual_style.mood.preference}")

        # Composition
        if self.composition.density.preference:
            parts.append(f"Density: {self.composition.density.preference}")
        if self.composition.perspective.preference:
            parts.append(f"Perspective: {self.composition.perspective.preference}")
        if self.composition.foreground.preference:
            parts.append(f"Foreground: {self.composition.foreground.preference}")

        # Quality must-haves
        if self.quality.must_have:
            parts.append(f"Must include: {', '.join(self.quality.must_have)}")

        # Things to avoid
        avoids = []
        if self.visual_style.lighting.avoid:
            avoids.append(self.visual_style.lighting.avoid)
        if self.visual_style.color_palette.avoid:
            avoids.append(self.visual_style.color_palette.avoid)
        if self.quality.never:
            avoids.extend(self.quality.never)
        if avoids:
            parts.append(f"Avoid: {', '.join(avoids)}")

        return ". ".join(parts) if parts else ""


@dataclass
class Generation:
    """A single generation result."""

    id: str
    prompt: str
    enhanced_prompt: str
    timestamp: datetime
    source_image_path: Optional[str] = None
    splat_path: Optional[str] = None
    mesh_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    feedback: Optional["Feedback"] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "timestamp": self.timestamp.isoformat(),
            "source_image_path": self.source_image_path,
            "splat_path": self.splat_path,
            "mesh_path": self.mesh_path,
            "metadata": self.metadata,
            "feedback": self.feedback.to_dict() if self.feedback else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Generation":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            enhanced_prompt=data.get("enhanced_prompt", data["prompt"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_image_path=data.get("source_image_path"),
            splat_path=data.get("splat_path"),
            mesh_path=data.get("mesh_path"),
            metadata=data.get("metadata", {}),
            feedback=Feedback.from_dict(data["feedback"]) if data.get("feedback") else None,
        )


@dataclass
class Feedback:
    """Feedback on a generation."""

    generation_id: str
    timestamp: datetime
    rating: str  # "++", "+", "-", "--", or "text"
    text: str = ""
    extracted_preferences: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generation_id": self.generation_id,
            "timestamp": self.timestamp.isoformat(),
            "rating": self.rating,
            "text": self.text,
            "extracted_preferences": self.extracted_preferences,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Feedback":
        return cls(
            generation_id=data["generation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            rating=data["rating"],
            text=data.get("text", ""),
            extracted_preferences=data.get("extracted_preferences", {}),
        )

    @property
    def is_positive(self) -> bool:
        return self.rating in ("++", "+")

    @property
    def is_negative(self) -> bool:
        return self.rating in ("--", "-")

    @property
    def is_love(self) -> bool:
        return self.rating == "++"

    @property
    def is_hate(self) -> bool:
        return self.rating == "--"
