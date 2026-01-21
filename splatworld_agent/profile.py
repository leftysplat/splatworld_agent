"""
Profile management for SplatWorld Agent.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from splatworld_agent.models import TasteProfile, Generation, Feedback, Exemplar
from splatworld_agent.config import PROJECT_DIR_NAME


class ProfileManager:
    """Manages taste profiles for a project."""

    PROFILE_FILE = "profile.json"
    FEEDBACK_FILE = "feedback.jsonl"
    GENERATIONS_DIR = "generations"
    EXEMPLARS_DIR = "exemplars"
    ANTI_EXEMPLARS_DIR = "anti-exemplars"

    def __init__(self, project_dir: Path):
        """Initialize profile manager for a project directory."""
        self.project_dir = project_dir
        self.splatworld_dir = project_dir / PROJECT_DIR_NAME

    @property
    def profile_path(self) -> Path:
        return self.splatworld_dir / self.PROFILE_FILE

    @property
    def feedback_path(self) -> Path:
        return self.splatworld_dir / self.FEEDBACK_FILE

    @property
    def generations_dir(self) -> Path:
        return self.splatworld_dir / self.GENERATIONS_DIR

    @property
    def exemplars_dir(self) -> Path:
        return self.splatworld_dir / self.EXEMPLARS_DIR

    @property
    def anti_exemplars_dir(self) -> Path:
        return self.splatworld_dir / self.ANTI_EXEMPLARS_DIR

    def is_initialized(self) -> bool:
        """Check if project has been initialized."""
        return self.splatworld_dir.exists() and self.profile_path.exists()

    def initialize(self) -> TasteProfile:
        """Initialize a new project with empty taste profile."""
        # Create directories
        self.splatworld_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(exist_ok=True)
        self.exemplars_dir.mkdir(exist_ok=True)
        self.anti_exemplars_dir.mkdir(exist_ok=True)

        # Create empty profile
        profile = TasteProfile()
        profile.save(self.profile_path)

        # Create empty feedback file
        self.feedback_path.touch()

        return profile

    def load_profile(self) -> TasteProfile:
        """Load the taste profile."""
        if not self.profile_path.exists():
            raise FileNotFoundError(
                f"Profile not found. Run 'splatworld-agent init' first."
            )
        return TasteProfile.load(self.profile_path)

    def save_profile(self, profile: TasteProfile) -> None:
        """Save the taste profile."""
        profile.save(self.profile_path)

    def add_feedback(self, feedback: Feedback) -> None:
        """Add feedback entry to feedback log."""
        with open(self.feedback_path, "a") as f:
            f.write(json.dumps(feedback.to_dict()) + "\n")

        # Update profile stats
        profile = self.load_profile()
        profile.stats.feedback_count += 1
        if feedback.is_love:
            profile.stats.love_count += 1
        elif feedback.is_hate:
            profile.stats.hate_count += 1
        self.save_profile(profile)

    def get_feedback_history(self, limit: Optional[int] = None) -> list[Feedback]:
        """Get feedback history, optionally limited to most recent."""
        if not self.feedback_path.exists():
            return []

        feedbacks = []
        with open(self.feedback_path) as f:
            for line in f:
                if line.strip():
                    feedbacks.append(Feedback.from_dict(json.loads(line)))

        if limit:
            feedbacks = feedbacks[-limit:]

        return feedbacks

    def get_unprocessed_feedback(self, since: Optional[datetime] = None) -> list[Feedback]:
        """Get feedback entries that haven't been processed into preferences yet."""
        all_feedback = self.get_feedback_history()
        if since:
            return [f for f in all_feedback if f.timestamp > since]
        return all_feedback

    def add_exemplar(self, source_path: Path, notes: str = "") -> Exemplar:
        """Add an exemplar image to the profile."""
        # Copy image to exemplars directory
        dest_path = self.exemplars_dir / source_path.name
        import shutil
        shutil.copy2(source_path, dest_path)

        # Add to profile
        exemplar = Exemplar(
            path=str(dest_path.relative_to(self.splatworld_dir)),
            added=datetime.now(),
            notes=notes,
        )

        profile = self.load_profile()
        profile.exemplars.append(exemplar)
        self.save_profile(profile)

        return exemplar

    def add_anti_exemplar(self, source_path: Path, notes: str = "") -> Exemplar:
        """Add an anti-exemplar image to the profile."""
        # Copy image to anti-exemplars directory
        dest_path = self.anti_exemplars_dir / source_path.name
        import shutil
        shutil.copy2(source_path, dest_path)

        # Add to profile
        exemplar = Exemplar(
            path=str(dest_path.relative_to(self.splatworld_dir)),
            added=datetime.now(),
            notes=notes,
        )

        profile = self.load_profile()
        profile.anti_exemplars.append(exemplar)
        self.save_profile(profile)

        return exemplar

    def save_generation(self, generation: Generation) -> Path:
        """Save a generation to the generations directory."""
        # Create date-based subdirectory
        date_dir = self.generations_dir / generation.timestamp.strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)

        # Create generation directory
        gen_dir = date_dir / generation.id
        gen_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata_path = gen_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(generation.to_dict(), f, indent=2)

        # Update profile stats
        profile = self.load_profile()
        profile.stats.total_generations += 1
        self.save_profile(profile)

        return gen_dir

    def get_generation(self, generation_id: str) -> Optional[Generation]:
        """Get a specific generation by ID."""
        # Search through date directories
        for date_dir in self.generations_dir.iterdir():
            if date_dir.is_dir():
                gen_dir = date_dir / generation_id
                if gen_dir.exists():
                    metadata_path = gen_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            return Generation.from_dict(json.load(f))
        return None

    def get_recent_generations(self, limit: int = 10) -> list[Generation]:
        """Get most recent generations."""
        generations = []

        # Get all generation directories sorted by date
        date_dirs = sorted(self.generations_dir.iterdir(), reverse=True)

        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue

            gen_dirs = sorted(date_dir.iterdir(), reverse=True)
            for gen_dir in gen_dirs:
                if not gen_dir.is_dir():
                    continue

                metadata_path = gen_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        generations.append(Generation.from_dict(json.load(f)))

                    if len(generations) >= limit:
                        return generations

        return generations

    def get_last_generation(self) -> Optional[Generation]:
        """Get the most recent generation."""
        recent = self.get_recent_generations(limit=1)
        return recent[0] if recent else None
