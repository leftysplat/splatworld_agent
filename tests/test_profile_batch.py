"""Tests for ProfileManager batch state methods."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from splatworld_agent.profile import ProfileManager
from splatworld_agent.models import Generation


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory with .splatworld folder."""
    splatworld_dir = tmp_path / ".splatworld"
    splatworld_dir.mkdir()
    (splatworld_dir / "generations").mkdir()

    # Create empty profile
    profile_path = splatworld_dir / "profile.json"
    profile_path.write_text(json.dumps({
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "stats": {"total_generations": 0, "feedback_count": 0},
        "calibration": {"is_calibrated": False}
    }))

    return tmp_path


class TestSetCurrentBatch:
    """Tests for set_current_batch method."""

    def test_creates_session_file(self, temp_project):
        """Should create current_session.json with batch data."""
        manager = ProfileManager(temp_project)

        manager.set_current_batch("batch-123", ["gen-1", "gen-2", "gen-3"])

        session_path = manager.current_session_path
        assert session_path.exists()

        with open(session_path) as f:
            data = json.load(f)

        assert data["current_batch_id"] == "batch-123"
        assert data["batch_generation_ids"] == ["gen-1", "gen-2", "gen-3"]
        assert data["batch_size"] == 3
        assert "batch_started" in data

    def test_overwrites_existing_batch(self, temp_project):
        """Should replace previous batch context."""
        manager = ProfileManager(temp_project)

        manager.set_current_batch("batch-old", ["gen-old"])
        manager.set_current_batch("batch-new", ["gen-a", "gen-b"])

        with open(manager.current_session_path) as f:
            data = json.load(f)

        assert data["current_batch_id"] == "batch-new"
        assert data["batch_generation_ids"] == ["gen-a", "gen-b"]


class TestResolveImageNumber:
    """Tests for resolve_image_number method."""

    def test_resolves_valid_numbers(self, temp_project):
        """Should map 1-indexed numbers to generation IDs."""
        manager = ProfileManager(temp_project)
        manager.set_current_batch("batch-1", ["gen-a", "gen-b", "gen-c"])

        assert manager.resolve_image_number(1) == "gen-a"
        assert manager.resolve_image_number(2) == "gen-b"
        assert manager.resolve_image_number(3) == "gen-c"

    def test_returns_none_for_out_of_range(self, temp_project):
        """Should return None for numbers outside batch size."""
        manager = ProfileManager(temp_project)
        manager.set_current_batch("batch-1", ["gen-a", "gen-b"])

        assert manager.resolve_image_number(0) is None  # 0 is invalid (1-indexed)
        assert manager.resolve_image_number(3) is None  # Only 2 images
        assert manager.resolve_image_number(-1) is None

    def test_returns_none_without_batch(self, temp_project):
        """Should return None when no batch context exists."""
        manager = ProfileManager(temp_project)

        assert manager.resolve_image_number(1) is None

    def test_handles_corrupted_session_file(self, temp_project):
        """Should return None if session file is invalid JSON."""
        manager = ProfileManager(temp_project)
        manager.current_session_path.write_text("not valid json")

        assert manager.resolve_image_number(1) is None


class TestGetCurrentBatchGenerations:
    """Tests for get_current_batch_generations method."""

    def test_returns_empty_without_batch(self, temp_project):
        """Should return empty list when no batch context."""
        manager = ProfileManager(temp_project)

        assert manager.get_current_batch_generations() == []

    def test_returns_generations_in_order(self, temp_project):
        """Should return Generation objects in batch order."""
        manager = ProfileManager(temp_project)

        # Create some test generations
        gen_ids = []
        for i in range(3):
            gen_id = f"gen-test-{i}"
            gen_ids.append(gen_id)

            # Create generation directory and metadata
            date_dir = manager.generations_dir / "2026-01-21"
            date_dir.mkdir(exist_ok=True)
            gen_dir = date_dir / gen_id
            gen_dir.mkdir()

            metadata = {
                "id": gen_id,
                "prompt": f"test prompt {i}",
                "enhanced_prompt": f"test prompt {i}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"batch_index": i}
            }
            (gen_dir / "metadata.json").write_text(json.dumps(metadata))

        # Set batch context
        manager.set_current_batch("batch-test", gen_ids)

        # Get generations
        generations = manager.get_current_batch_generations()

        assert len(generations) == 3
        assert generations[0].id == "gen-test-0"
        assert generations[1].id == "gen-test-1"
        assert generations[2].id == "gen-test-2"
