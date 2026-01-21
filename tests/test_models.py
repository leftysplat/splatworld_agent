"""Tests for data models."""

import pytest
from datetime import datetime

from splatworld_agent.models import (
    TasteProfile,
    StylePreference,
    VisualStyle,
    Feedback,
    Generation,
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
