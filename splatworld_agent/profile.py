"""
Profile management for SplatWorld Agent.
"""

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from splatworld_agent.models import TasteProfile, Generation, Feedback, Exemplar, Session, PromptHistoryEntry
from splatworld_agent.config import PROJECT_DIR_NAME


class ProfileManager:
    """Manages taste profiles for a project."""

    PROFILE_FILE = "profile.json"
    FEEDBACK_FILE = "feedback.jsonl"
    SESSIONS_FILE = "sessions.jsonl"
    PROMPT_HISTORY_FILE = "prompt_history.jsonl"  # HIST-01: Track prompt variants with ratings
    CURRENT_SESSION_FILE = "current_session.json"
    GENERATIONS_DIR = "generations"  # Inside .splatworld (metadata only)
    EXEMPLARS_DIR = "exemplars"
    ANTI_EXEMPLARS_DIR = "anti-exemplars"

    # Visible directories at project root (for user-accessible files)
    IMAGES_DIR = "generated_images"
    SPLATS_DIR = "downloaded_splats"

    # Sequential numbering support (FILE-02)
    IMAGE_REGISTRY_FILE = "image_registry.json"
    METADATA_DIR = "metadata"

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
        """Metadata directory inside .splatworld."""
        return self.splatworld_dir / self.GENERATIONS_DIR

    @property
    def images_dir(self) -> Path:
        """Visible directory for generated images."""
        return self.project_dir / self.IMAGES_DIR

    @property
    def splats_dir(self) -> Path:
        """Visible directory for downloaded splats."""
        return self.project_dir / self.SPLATS_DIR

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

    @property
    def prompt_history_path(self) -> Path:
        """Path to prompt_history.jsonl file for tracking prompt variants."""
        return self.splatworld_dir / self.PROMPT_HISTORY_FILE

    @property
    def metadata_dir(self) -> Path:
        """Directory for metadata files (registry, etc.)."""
        return self.splatworld_dir / self.METADATA_DIR

    @property
    def image_registry_path(self) -> Path:
        """Path to image_registry.json file."""
        return self.metadata_dir / self.IMAGE_REGISTRY_FILE

    def is_initialized(self) -> bool:
        """Check if project has been initialized."""
        return self.splatworld_dir.exists() and self.profile_path.exists()

    def initialize(self) -> TasteProfile:
        """Initialize a new project with empty taste profile."""
        # Create hidden directories (metadata/config)
        self.splatworld_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(exist_ok=True)
        self.exemplars_dir.mkdir(exist_ok=True)
        self.anti_exemplars_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)  # For registry and other metadata

        # Create visible directories (user-accessible files)
        self.images_dir.mkdir(exist_ok=True)
        self.splats_dir.mkdir(exist_ok=True)

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
                f"Profile not found. Run 'splatworld init' first."
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

    def save_generation(self, generation: Generation) -> tuple[Path, Path]:
        """Save a generation to the generations directory.

        Returns:
            Tuple of (image_dir, metadata_dir) where:
            - image_dir: visible directory for images (generated_images/date/id/)
            - metadata_dir: hidden directory for metadata (.splatworld/generations/date/id/)
        """
        date_str = generation.timestamp.strftime("%Y-%m-%d")

        # Create visible directory for images
        image_date_dir = self.images_dir / date_str
        image_date_dir.mkdir(parents=True, exist_ok=True)
        image_dir = image_date_dir / generation.id
        image_dir.mkdir(exist_ok=True)

        # Create hidden directory for metadata
        metadata_date_dir = self.generations_dir / date_str
        metadata_date_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir = metadata_date_dir / generation.id
        metadata_dir.mkdir(exist_ok=True)

        # Save metadata to hidden directory
        metadata_path = metadata_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(generation.to_dict(), f, indent=2)

        # Update profile stats
        profile = self.load_profile()
        profile.stats.total_generations += 1
        self.save_profile(profile)

        return image_dir, metadata_dir

    def get_metadata_dir(self, generation_id: str) -> Optional[Path]:
        """Get metadata directory for a generation ID."""
        for date_dir in self.generations_dir.iterdir():
            if date_dir.is_dir():
                gen_dir = date_dir / generation_id
                if gen_dir.exists():
                    return gen_dir
        return None

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

    def get_all_generations(self) -> list[Generation]:
        """Get all generations (no limit)."""
        generations = []

        if not self.generations_dir.exists():
            return generations

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

    def get_all_unrated_generations(self) -> list[tuple[Generation, dict]]:
        """Get all unrated generations across all batches with batch context.

        Returns list of (Generation, batch_context) tuples sorted by timestamp descending.
        batch_context contains:
            - batch_id: str or None
            - batch_index: int (0-indexed position from metadata)
            - batch_size: int (count of generations with same batch_id)
        """
        # Get all rated generation IDs
        rated_ids = {f.generation_id for f in self.get_feedback_history()}

        # First pass: collect all unrated generations and track batch sizes
        unrated_gens: list[tuple[Generation, Optional[str], int]] = []  # (gen, batch_id, batch_index)
        batch_counts: dict[str, int] = {}  # batch_id -> count of generations

        # Iterate ALL date directories
        if not self.generations_dir.exists():
            return []

        for date_dir in self.generations_dir.iterdir():
            if not date_dir.is_dir():
                continue

            for gen_dir in date_dir.iterdir():
                if not gen_dir.is_dir():
                    continue

                metadata_path = gen_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path) as f:
                        gen_data = json.load(f)
                except json.JSONDecodeError:
                    # Skip corrupted metadata files
                    continue

                gen = Generation.from_dict(gen_data)

                # Track batch size for all generations (rated or not)
                batch_id = gen.metadata.get("batch_id")
                if batch_id:
                    batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1

                # Only collect unrated generations
                if gen.id not in rated_ids:
                    batch_index = gen.metadata.get("batch_index", 0)
                    unrated_gens.append((gen, batch_id, batch_index))

        # Build result with batch context
        result: list[tuple[Generation, dict]] = []
        for gen, batch_id, batch_index in unrated_gens:
            batch_context = {
                "batch_id": batch_id,
                "batch_index": batch_index,
                "batch_size": batch_counts.get(batch_id, 1) if batch_id else 1,
            }
            result.append((gen, batch_context))

        # Sort by timestamp descending (most recent first)
        result.sort(key=lambda x: x[0].timestamp, reverse=True)

        return result

    # Prompt history methods (HIST-01, HIST-02, HIST-03)

    def save_prompt_variant(self, entry: PromptHistoryEntry) -> None:
        """Save a prompt variant to prompt_history.jsonl (HIST-01).

        Append-only storage for all prompt variants tried during training.
        """
        with open(self.prompt_history_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def update_prompt_variant_rating(self, variant_id: str, rating: str) -> bool:
        """Update the rating for a prompt variant.

        Returns True if variant was found and updated, False otherwise.
        """
        if not self.prompt_history_path.exists():
            return False

        entries = self.get_prompt_history()
        updated = False

        for entry in entries:
            if entry.variant_id == variant_id:
                entry.rating = rating
                updated = True
                break

        if updated:
            # Atomic rewrite
            temp_path = self.prompt_history_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            temp_path.replace(self.prompt_history_path)

        return updated

    def get_prompt_history(self, limit: Optional[int] = None, session_id: Optional[str] = None) -> list[PromptHistoryEntry]:
        """Get prompt history, optionally filtered by session (HIST-03).

        Args:
            limit: Maximum number of entries to return (most recent first)
            session_id: Filter to only entries from this training session

        Returns:
            List of PromptHistoryEntry objects, most recent first
        """
        if not self.prompt_history_path.exists():
            return []

        entries = []
        with open(self.prompt_history_path) as f:
            for line in f:
                if line.strip():
                    entry = PromptHistoryEntry.from_dict(json.loads(line))
                    if session_id is None or entry.session_id == session_id:
                        entries.append(entry)

        # Most recent first
        entries.reverse()

        if limit:
            entries = entries[:limit]

        return entries

    def get_variant_lineage(self, variant_id: str) -> list[PromptHistoryEntry]:
        """Get the lineage chain for a variant (HIST-02).

        Returns list starting from oldest ancestor to the given variant.
        """
        if not self.prompt_history_path.exists():
            return []

        # Build lookup dict
        all_entries = {}
        with open(self.prompt_history_path) as f:
            for line in f:
                if line.strip():
                    entry = PromptHistoryEntry.from_dict(json.loads(line))
                    all_entries[entry.variant_id] = entry

        # Trace lineage backwards
        lineage = []
        current_id = variant_id

        while current_id and current_id in all_entries:
            entry = all_entries[current_id]
            lineage.append(entry)
            current_id = entry.parent_variant_id

        # Reverse to get oldest first
        lineage.reverse()
        return lineage

    def get_prompt_history_stats(self) -> dict:
        """Get statistics about prompt history.

        Returns dict with counts for total variants, rated, positive, negative, etc.
        """
        entries = self.get_prompt_history()

        total = len(entries)
        rated = sum(1 for e in entries if e.is_rated)
        positive = sum(1 for e in entries if e.is_positive)
        negative = sum(1 for e in entries if e.is_negative)
        unrated = total - rated

        # Count unique base prompts
        base_prompts = set(e.base_prompt for e in entries)

        # Count unique sessions
        sessions = set(e.session_id for e in entries if e.session_id)

        return {
            "total_variants": total,
            "rated": rated,
            "unrated": unrated,
            "positive": positive,
            "negative": negative,
            "unique_base_prompts": len(base_prompts),
            "training_sessions": len(sessions),
        }

    # Sequential image numbering methods (FILE-02)

    def _atomic_save_profile(self, profile: TasteProfile) -> None:
        """Save profile atomically using temp file + rename.

        This prevents race conditions and ensures profile is never corrupted
        even if interrupted mid-write.
        """
        temp_path = self.profile_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        temp_path.replace(self.profile_path)

    def get_next_image_number(self) -> int:
        """Get the next image number and atomically increment the counter.

        Returns:
            The next sequential image number (starts at 1)

        Thread-safety: Uses atomic file write to prevent race conditions.
        """
        profile = self.load_profile()
        next_num = profile.next_image_number
        profile.next_image_number += 1
        self._atomic_save_profile(profile)
        return next_num

    def _scan_for_highest_image_number(self) -> int:
        """Scan existing images to find the highest image number.

        Used for recovery if counter gets out of sync.

        Returns:
            The highest image number found, or 0 if no numbered images exist.
        """
        highest = 0
        pattern = re.compile(r"^(\d{4})-")  # Match NNNN- prefix

        # Scan images_dir for numbered images
        if self.images_dir.exists():
            for item in self.images_dir.iterdir():
                match = pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    highest = max(highest, num)

        # Also check splats_dir
        if self.splats_dir.exists():
            for item in self.splats_dir.iterdir():
                match = pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    highest = max(highest, num)

        return highest

    def get_image_registry(self) -> dict:
        """Get the image registry mapping old IDs to new sequential numbers.

        Returns:
            Dict mapping old generation IDs to their new sequential numbers.
        """
        if not self.image_registry_path.exists():
            return {}

        with open(self.image_registry_path) as f:
            return json.load(f)

    def register_image(self, old_id: str, new_number: int) -> None:
        """Register a mapping from old generation ID to new sequential number.

        Args:
            old_id: The original generation UUID/ID
            new_number: The new sequential image number
        """
        registry = self.get_image_registry()
        registry[old_id] = new_number

        # Atomic write
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.image_registry_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(registry, f, indent=2)
        temp_path.replace(self.image_registry_path)

    def resolve_to_number(self, ref: str) -> Optional[int]:
        """Resolve a reference (number or old ID) to a sequential image number.

        Args:
            ref: Either a number string ("1", "42") or an old generation ID

        Returns:
            The sequential image number, or None if not found
        """
        # Check if it's already a number
        if ref.isdigit():
            return int(ref)

        # Check if it's a number with leading zeros
        try:
            return int(ref)
        except ValueError:
            pass

        # Look up in registry
        registry = self.get_image_registry()
        return registry.get(ref)

    # Flat file structure methods (FILE-02 Expand phase)

    @property
    def image_metadata_dir(self) -> Path:
        """Directory for per-image metadata files (N-metadata.json)."""
        return self.splatworld_dir / "image_metadata"

    def get_flat_image_path(self, image_number: int) -> Path:
        """Get the flat file path for an image number.

        Args:
            image_number: The sequential image number

        Returns:
            Path to generated_images/N.png
        """
        return self.images_dir / f"{image_number}.png"

    def get_flat_splat_path(self, image_number: int) -> Path:
        """Get the flat file path for a splat number.

        Args:
            image_number: The sequential image number

        Returns:
            Path to downloaded_splats/N.spz
        """
        return self.splats_dir / f"{image_number}.spz"

    def save_image_metadata(self, image_number: int, metadata: dict) -> Path:
        """Save metadata for a flat-numbered image.

        Args:
            image_number: The sequential image number
            metadata: Dict with generation metadata (prompt, generator, etc.)

        Returns:
            Path to the saved metadata file
        """
        self.image_metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.image_metadata_dir / f"{image_number}-metadata.json"

        # Atomic write
        temp_path = metadata_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        temp_path.replace(metadata_path)

        return metadata_path

    def load_image_metadata(self, image_number: int) -> Optional[dict]:
        """Load metadata for a flat-numbered image.

        Args:
            image_number: The sequential image number

        Returns:
            Dict with generation metadata, or None if not found
        """
        metadata_path = self.image_metadata_dir / f"{image_number}-metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    # Migration methods (FILE-03)

    def _collect_nested_generations(self) -> list[Generation]:
        """Collect all generations from nested structure, sorted by timestamp.

        Returns:
            List of Generation objects sorted oldest-first (for migration order)
        """
        generations = []

        if not self.generations_dir.exists():
            return generations

        # Iterate through date directories
        for date_dir in self.generations_dir.iterdir():
            if not date_dir.is_dir():
                continue

            # Iterate through generation directories within each date
            for gen_dir in date_dir.iterdir():
                if not gen_dir.is_dir():
                    continue

                metadata_path = gen_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path) as f:
                        gen_data = json.load(f)
                    gen = Generation.from_dict(gen_data)
                    generations.append(gen)
                except (json.JSONDecodeError, KeyError, IOError):
                    # Skip corrupted metadata
                    continue

        # Sort by timestamp, oldest first (for chronological numbering)
        generations.sort(key=lambda g: g.timestamp)

        return generations

    def migrate_existing_generations(self, dry_run: bool = False) -> dict:
        """Migrate existing nested generations to flat structure.

        Copies images and metadata from:
          generated_images/DATE/UUID/source.png -> generated_images/N.png
          (metadata) -> .splatworld/image_metadata/N-metadata.json

        Also migrates splat files if present:
          downloaded_splats/gen-UUID.spz -> downloaded_splats/N.spz

        Args:
            dry_run: If True, report what would be done without making changes

        Returns:
            Dict with migration stats:
            - migrated: Number of images migrated
            - skipped: Number of already-migrated images (idempotent)
            - splats_migrated: Number of splat files migrated
            - errors: List of error messages
        """
        stats = {
            "migrated": 0,
            "skipped": 0,
            "splats_migrated": 0,
            "errors": [],
        }

        # Get all nested generations sorted by timestamp (oldest first)
        generations = self._collect_nested_generations()

        if not generations:
            return stats

        # Get current registry to check for already-migrated
        registry = self.get_image_registry()

        for gen in generations:
            # Check if already migrated (idempotent)
            if gen.id in registry:
                stats["skipped"] += 1
                continue

            # Find source image in nested structure
            source_path = None
            if gen.source_image_path:
                source_path = Path(gen.source_image_path)
                if not source_path.exists():
                    # Try relative path from generations_dir
                    date_str = gen.timestamp.strftime("%Y-%m-%d")
                    alt_path = self.images_dir / date_str / gen.id / "source.png"
                    if alt_path.exists():
                        source_path = alt_path
                    else:
                        stats["errors"].append(f"Source not found for {gen.id}")
                        continue
            else:
                # No source_image_path, try to locate
                date_str = gen.timestamp.strftime("%Y-%m-%d")
                source_path = self.images_dir / date_str / gen.id / "source.png"
                if not source_path.exists():
                    stats["errors"].append(f"No source image for {gen.id}")
                    continue

            if dry_run:
                # Just count what would be migrated
                stats["migrated"] += 1
                # Check for splat file
                splat_source = self.splats_dir / f"gen-{gen.id}.spz"
                if splat_source.exists():
                    stats["splats_migrated"] += 1
                continue

            # Get next number and migrate
            new_number = self.get_next_image_number()

            # Create destination paths
            dest_image = self.get_flat_image_path(new_number)

            # Ensure directories exist
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.image_metadata_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Copy image file
                shutil.copy2(source_path, dest_image)

                # Save metadata to new location
                metadata = gen.to_dict()
                metadata["original_id"] = gen.id
                metadata["migrated_at"] = datetime.now().isoformat()
                self.save_image_metadata(new_number, metadata)

                # Register the mapping
                self.register_image(gen.id, new_number)

                stats["migrated"] += 1

                # Handle splat file if it exists (gen-UUID.spz format)
                splat_source = self.splats_dir / f"gen-{gen.id}.spz"
                if splat_source.exists():
                    splat_dest = self.get_flat_splat_path(new_number)
                    self.splats_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(splat_source, splat_dest)
                    stats["splats_migrated"] += 1

            except IOError as e:
                stats["errors"].append(f"Failed to migrate {gen.id}: {e}")
                # Note: Counter was incremented but image not created
                # This is acceptable - gaps in numbering are OK

        return stats

    def verify_migration(self) -> dict:
        """Verify migration status and return statistics.

        Returns:
            Dict with verification stats:
            - nested_images: Count of images in nested structure
            - flat_images: Count of images in flat structure (N.png)
            - registered: Count of IDs in registry
            - nested_splats: Count of splats in nested format (gen-UUID.spz)
            - flat_splats: Count of splats in flat format (N.spz)
            - unregistered: List of nested IDs not in registry
            - next_number: Current next_image_number counter value
        """
        nested_count = 0
        flat_count = 0
        nested_splats = 0
        flat_splats = 0
        nested_ids = set()

        # Count nested images and collect IDs
        if self.generations_dir.exists():
            for date_dir in self.generations_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                for gen_dir in date_dir.iterdir():
                    if gen_dir.is_dir():
                        # Check if source.png exists in corresponding images dir
                        date_str = date_dir.name
                        image_path = self.images_dir / date_str / gen_dir.name / "source.png"
                        if image_path.exists():
                            nested_count += 1
                            nested_ids.add(gen_dir.name)

        # Count flat images (N.png pattern)
        if self.images_dir.exists():
            for item in self.images_dir.iterdir():
                if item.is_file() and item.suffix == ".png":
                    # Check if filename is a number
                    try:
                        int(item.stem)
                        flat_count += 1
                    except ValueError:
                        pass

        # Count splats
        if self.splats_dir.exists():
            for item in self.splats_dir.iterdir():
                if item.is_file() and item.suffix == ".spz":
                    if item.stem.startswith("gen-"):
                        nested_splats += 1
                    else:
                        try:
                            int(item.stem)
                            flat_splats += 1
                        except ValueError:
                            pass

        # Get registry
        registry = self.get_image_registry()
        registered_ids = set(registry.keys())

        # Find unregistered nested IDs
        unregistered = nested_ids - registered_ids

        # Get current counter value
        profile = self.load_profile()
        next_number = profile.next_image_number

        return {
            "nested_images": nested_count,
            "flat_images": flat_count,
            "registered": len(registry),
            "nested_splats": nested_splats,
            "flat_splats": flat_splats,
            "unregistered": list(unregistered),
            "next_number": next_number,
        }
