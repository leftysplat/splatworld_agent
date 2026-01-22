"""Tests for data models."""

import pytest
from datetime import datetime

from splatworld_agent.models import (
    TasteProfile,
    StylePreference,
    VisualStyle,
    Feedback,
    Generation,
    ExplorationMode,
)


class TestStylePreference:
    def test_to_dict(self):
        pref = StylePreference(
            preference="moody lighting",
            avoid="flat lighting",
            confidence=0.8,
        )
        d = pref.to_dict()
        assert d["preference"] == "moody lighting"
        assert d["avoid"] == "flat lighting"
        assert d["confidence"] == 0.8

    def test_from_dict(self):
        d = {"preference": "warm", "avoid": "cold", "confidence": 0.5}
        pref = StylePreference.from_dict(d)
        assert pref.preference == "warm"
        assert pref.avoid == "cold"
        assert pref.confidence == 0.5


class TestTasteProfile:
    def test_empty_profile(self):
        profile = TasteProfile()
        assert profile.version == "1.0"
        assert profile.stats.total_generations == 0

    def test_to_prompt_context_empty(self):
        profile = TasteProfile()
        assert profile.to_prompt_context() == ""

    def test_to_prompt_context_with_preferences(self):
        profile = TasteProfile()
        profile.visual_style.lighting.preference = "moody, dramatic"
        profile.visual_style.color_palette.preference = "warm earth tones"
        profile.quality.must_have = ["realistic lighting"]

        context = profile.to_prompt_context()
        assert "moody, dramatic" in context
        assert "warm earth tones" in context
        assert "realistic lighting" in context

    def test_serialization_roundtrip(self):
        profile = TasteProfile()
        profile.visual_style.lighting.preference = "test"
        profile.stats.total_generations = 5

        d = profile.to_dict()
        loaded = TasteProfile.from_dict(d)

        assert loaded.visual_style.lighting.preference == "test"
        assert loaded.stats.total_generations == 5


class TestFeedback:
    def test_quick_ratings(self):
        love = Feedback(
            generation_id="test",
            timestamp=datetime.now(),
            rating="++",
        )
        assert love.is_positive
        assert love.is_love
        assert not love.is_negative

        hate = Feedback(
            generation_id="test",
            timestamp=datetime.now(),
            rating="--",
        )
        assert hate.is_negative
        assert hate.is_hate
        assert not hate.is_positive


class TestGeneration:
    def test_basic_creation(self):
        gen = Generation(
            id="test-001",
            prompt="modern kitchen",
            enhanced_prompt="modern kitchen with warm lighting",
            timestamp=datetime.now(),
        )
        assert gen.id == "test-001"
        assert gen.prompt == "modern kitchen"


class TestExplorationMode:
    def test_enum_values(self):
        assert ExplorationMode.EXPLORE_WIDE.value == "explore"
        assert ExplorationMode.REFINE_NARROW.value == "refine"

    def test_from_string_explore_variants(self):
        # Test all explore variants
        assert ExplorationMode.from_string("explore") == ExplorationMode.EXPLORE_WIDE
        assert ExplorationMode.from_string("wide") == ExplorationMode.EXPLORE_WIDE
        assert ExplorationMode.from_string("explore_wide") == ExplorationMode.EXPLORE_WIDE
        assert ExplorationMode.from_string("explore-wide") == ExplorationMode.EXPLORE_WIDE
        assert ExplorationMode.from_string("EXPLORE") == ExplorationMode.EXPLORE_WIDE

    def test_from_string_refine_variants(self):
        # Test all refine variants
        assert ExplorationMode.from_string("refine") == ExplorationMode.REFINE_NARROW
        assert ExplorationMode.from_string("narrow") == ExplorationMode.REFINE_NARROW
        assert ExplorationMode.from_string("refine_narrow") == ExplorationMode.REFINE_NARROW
        assert ExplorationMode.from_string("refine-narrow") == ExplorationMode.REFINE_NARROW
        assert ExplorationMode.from_string("REFINE") == ExplorationMode.REFINE_NARROW

    def test_from_string_invalid(self):
        with pytest.raises(ValueError) as exc_info:
            ExplorationMode.from_string("invalid")
        assert "Unknown exploration mode" in str(exc_info.value)
