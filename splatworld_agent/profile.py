"""
Profile management for SplatWorld Agent.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from splatworld_agent.models import TasteProfile, Generation, Feedback, Exemplar, Session
from splatworld_agent.config import PROJECT_DIR_NAME


class ProfileManager:
    """Manages taste profiles for a project."""

    PROFILE_FILE = "profile.json"
    FEEDBACK_FILE = "feedback.jsonl"
    SESSIONS_FILE = "sessions.jsonl"
    CURRENT_SESSION_FILE = "current_session.json"
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

    @property
    def sessions_path(self) -> Path:
        return self.splatworld_dir / self.SESSIONS_FILE

    @property
    def current_session_path(self) -> Path:
        return self.splatworld_dir / self.CURRENT_SESSION_FILE

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

        # Track all rating types
        if feedback.rating == "++":
            profile.stats.love_count += 1
        elif feedback.rating == "+":
            profile.stats.like_count += 1
        elif feedback.rating == "-":
            profile.stats.dislike_count += 1
        elif feedback.rating == "--":
            profile.stats.hate_count += 1

        self.save_profile(profile)

    def _increment_rating_stat(self, profile: TasteProfile, rating: str) -> None:
        """Increment the counter for a rating type."""
        if rating == "++":
            profile.stats.love_count += 1
        elif rating == "+":
            profile.stats.like_count += 1
        elif rating == "-":
            profile.stats.dislike_count += 1
        elif rating == "--":
            profile.stats.hate_count += 1

    def _decrement_rating_stat(self, profile: TasteProfile, rating: str) -> None:
        """Decrement the counter for a rating type, flooring at 0."""
        if rating == "++":
            profile.stats.love_count = max(0, profile.stats.love_count - 1)
        elif rating == "+":
            profile.stats.like_count = max(0, profile.stats.like_count - 1)
        elif rating == "-":
            profile.stats.dislike_count = max(0, profile.stats.dislike_count - 1)
        elif rating == "--":
            profile.stats.hate_count = max(0, profile.stats.hate_count - 1)

    def add_or_replace_feedback(self, feedback: Feedback) -> tuple[bool, Optional[str]]:
        """Add feedback, replacing any existing for the same generation.

        Returns (was_replacement, old_rating).
        - was_replacement: True if this replaced an existing rating
        - old_rating: The previous rating if replaced, None if new
        """
        existing = self.get_feedback_history()

        # Find and remove existing feedback for this generation
        old_rating = None
        filtered = []
        for fb in existing:
            if fb.generation_id == feedback.generation_id:
                old_rating = fb.rating
            else:
                filtered.append(fb)

        # Add new feedback
        filtered.append(feedback)

        # Atomic rewrite of feedback file
        temp_path = self.feedback_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            for fb in filtered:
                f.write(json.dumps(fb.to_dict()) + "\n")
        temp_path.replace(self.feedback_path)

        # Update profile stats
        profile = self.load_profile()

        if old_rating:
            # This is a re-rate: adjust category counts but not total
            self._decrement_rating_stat(profile, old_rating)
        else:
            # New rating: increment total count
            profile.stats.feedback_count += 1

        # Always increment the new rating category
        self._increment_rating_stat(profile, feedback.rating)

        self.save_profile(profile)

        return (old_rating is not None, old_rating)

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

    # Session management methods

    def start_session(self) -> Session:
        """Start a new session and record current stats."""
        profile = self.load_profile()

        session = Session(
            session_id=f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            started=datetime.now(),
            activity={
                "start_stats": {
                    "generations": profile.stats.total_generations,
                    "feedback": profile.stats.feedback_count,
                    "learns": profile.calibration.learn_count,
                }
            },
        )

        # Write current session file
        with open(self.current_session_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        return session

    def end_session(self, summary: str = "", notes: str = "") -> Optional[Session]:
        """End the current session and save to history."""
        if not self.current_session_path.exists():
            return None

        # Load current session
        with open(self.current_session_path) as f:
            session = Session.from_dict(json.load(f))

        # Calculate activity delta
        activity = self.calculate_session_activity(session.started)
        session.activity.update(activity)
        session.ended = datetime.now()
        session.summary = summary
        session.notes = notes if notes else None

        # Get last generation info
        last_gen = self.get_last_generation()
        if last_gen and last_gen.timestamp >= session.started:
            session.last_generation_id = last_gen.id
            session.last_prompt = last_gen.prompt

        # Append to sessions history
        with open(self.sessions_path, "a") as f:
            f.write(json.dumps(session.to_dict()) + "\n")

        # Remove current session file
        self.current_session_path.unlink()

        return session

    def get_current_session(self) -> Optional[Session]:
        """Get the current active session if one exists."""
        if not self.current_session_path.exists():
            return None

        with open(self.current_session_path) as f:
            return Session.from_dict(json.load(f))

    def get_sessions(self, limit: int = 10) -> list[Session]:
        """Get recent sessions from history."""
        if not self.sessions_path.exists():
            return []

        sessions = []
        with open(self.sessions_path) as f:
            for line in f:
                if line.strip():
                    sessions.append(Session.from_dict(json.loads(line)))

        # Return most recent first
        sessions.reverse()
        return sessions[:limit]

    def calculate_session_activity(self, since: datetime) -> dict:
        """Calculate activity since a given timestamp."""
        profile = self.load_profile()
        start_stats = {}

        # Try to get start stats from current session
        current = self.get_current_session()
        if current and current.activity.get("start_stats"):
            start_stats = current.activity["start_stats"]
        else:
            # Fall back to counting manually
            start_stats = {
                "generations": 0,
                "feedback": 0,
                "learns": 0,
            }

        # Count generations since session start
        generations_since = 0
        for gen in self.get_recent_generations(limit=100):
            if gen.timestamp >= since:
                generations_since += 1
            else:
                break

        # Count feedback since session start
        feedback_since = 0
        for fb in reversed(self.get_feedback_history()):
            if fb.timestamp >= since:
                feedback_since += 1
            else:
                break

        # Calculate conversions (generations with splat_path that were created since session start)
        conversions_since = 0
        for gen in self.get_recent_generations(limit=100):
            if gen.timestamp >= since and gen.splat_path:
                conversions_since += 1

        return {
            "generations": generations_since,
            "feedback": feedback_since,
            "conversions": conversions_since,
            "learns": profile.calibration.learn_count - start_stats.get("learns", 0),
        }

    # Batch state management methods

    def set_current_batch(self, batch_id: str, generation_ids: list[str]) -> None:
        """Set the active batch for numbered image references."""
        session_data = {
            "current_batch_id": batch_id,
            "batch_generation_ids": generation_ids,
            "batch_size": len(generation_ids),
            "batch_started": datetime.now().isoformat()
        }
        # Atomic write
        temp_path = self.current_session_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(session_data, f, indent=2)
        temp_path.replace(self.current_session_path)

    def resolve_image_number(self, image_num: int) -> Optional[str]:
        """Map 1-indexed image number to generation ID."""
        if not self.current_session_path.exists():
            return None
        try:
            with open(self.current_session_path) as f:
                session = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        gen_ids = session.get("batch_generation_ids", [])
        index = image_num - 1  # Convert to 0-indexed
        if 0 <= index < len(gen_ids):
            return gen_ids[index]
        return None

    def get_current_batch_generations(self) -> list[Generation]:
        """Get all generations in the current batch, in order."""
        if not self.current_session_path.exists():
            return []
        try:
            with open(self.current_session_path) as f:
                session = json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
        gen_ids = session.get("batch_generation_ids", [])
        generations = []
        for gen_id in gen_ids:
            gen = self.get_generation(gen_id)
            if gen:
                generations.append(gen)
        return generations

    def get_unrated_in_batch(self, batch_generation_ids: list[str]) -> list[int]:
        """Get 1-indexed image numbers in batch that haven't been rated."""
        feedbacks = {f.generation_id for f in self.get_feedback_history()}

        unrated = []
        for i, gen_id in enumerate(batch_generation_ids, start=1):
            if gen_id not in feedbacks:
                unrated.append(i)

        return unrated
