"""
Learning synthesis for SplatWorld Agent.

Uses Claude to analyze feedback and extract preferences for the taste profile.
"""

import json
from datetime import datetime
from typing import Optional

from anthropic import Anthropic

from .models import TasteProfile, Feedback, Generation, StylePreference


SYNTHESIS_SYSTEM_PROMPT = """You are a taste profile analyzer for a 3D scene generation system.
Your job is to analyze user feedback on generated images and extract preferences.

The user will provide:
1. A generation with its prompt and result
2. User feedback (rating and optional comment)
3. The current taste profile

Based on this, you should identify patterns and suggest updates to the taste profile.

Respond with a JSON object containing:
{
    "analysis": "Brief analysis of what the user liked/disliked",
    "updates": {
        "visual_style": {
            "lighting": {"preference": "...", "avoid": "...", "confidence_delta": 0.1},
            "color_palette": {"preference": "...", "avoid": "...", "confidence_delta": 0.1},
            "mood": {"preference": "...", "avoid": "...", "confidence_delta": 0.1}
        },
        "composition": {
            "density": {"preference": "...", "avoid": "...", "confidence_delta": 0.1},
            "perspective": {"preference": "...", "avoid": "...", "confidence_delta": 0.1},
            "foreground": {"preference": "...", "avoid": "...", "confidence_delta": 0.1}
        },
        "domain": {
            "environments": {"add": [], "remove": []},
            "avoid_environments": {"add": [], "remove": []}
        },
        "quality": {
            "must_have": {"add": [], "remove": []},
            "never": {"add": [], "remove": []}
        }
    }
}

Only include fields that should be updated. Use confidence_delta to indicate how strongly
this feedback should influence the preference (0.05 for weak signal, 0.1 for moderate, 0.2 for strong).

For positive feedback (+ or ++), the prompt elements are preferences.
For negative feedback (- or --), the prompt elements should be avoided.
"""


class LearningEngine:
    """Learns user preferences from feedback using Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the learning engine.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        """
        self.client = Anthropic(api_key=api_key)

    def analyze_feedback(
        self,
        generation: Generation,
        feedback: Feedback,
        profile: TasteProfile,
    ) -> dict:
        """
        Analyze feedback and suggest profile updates.

        Args:
            generation: The generation that received feedback
            feedback: The user's feedback
            profile: Current taste profile

        Returns:
            Analysis results with suggested updates
        """
        user_message = f"""
## Generation
- Prompt: {generation.prompt}
- Enhanced prompt: {generation.enhanced_prompt}
- Had 3D conversion: {generation.splat_path is not None}

## Feedback
- Rating: {feedback.rating} ({self._rating_description(feedback.rating)})
- Comment: {feedback.text or "No comment provided"}

## Current Profile Summary
{profile.to_prompt_context() or "Empty profile - no preferences learned yet"}

Please analyze this feedback and suggest updates to the taste profile.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Parse JSON from response
        text = response.content[0].text

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    def synthesize_from_history(
        self,
        generations: list[Generation],
        feedbacks: list[Feedback],
        profile: TasteProfile,
    ) -> dict:
        """
        Synthesize preferences from multiple feedback entries.

        Args:
            generations: List of generations
            feedbacks: List of feedbacks (matched by generation_id)
            profile: Current taste profile

        Returns:
            Analysis with comprehensive profile updates
        """
        # Build feedback history
        feedback_map = {f.generation_id: f for f in feedbacks}

        history_items = []
        for gen in generations:
            fb = feedback_map.get(gen.id)
            if fb:
                history_items.append({
                    "prompt": gen.prompt,
                    "enhanced_prompt": gen.enhanced_prompt,
                    "rating": fb.rating,
                    "comment": fb.text,
                })

        if not history_items:
            return {"analysis": "No feedback history to analyze", "updates": {}}

        user_message = f"""
## Feedback History ({len(history_items)} items)
{json.dumps(history_items, indent=2)}

## Current Profile Summary
{profile.to_prompt_context() or "Empty profile - no preferences learned yet"}

Please analyze all this feedback holistically and suggest comprehensive updates
to the taste profile. Look for patterns across multiple feedbacks.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    def apply_updates(self, profile: TasteProfile, updates: dict) -> TasteProfile:
        """
        Apply suggested updates to a taste profile.

        Args:
            profile: Profile to update
            updates: Updates from analyze_feedback or synthesize_from_history

        Returns:
            Updated profile
        """
        if "visual_style" in updates:
            vs = updates["visual_style"]
            self._update_style_pref(profile.visual_style.lighting, vs.get("lighting", {}))
            self._update_style_pref(profile.visual_style.color_palette, vs.get("color_palette", {}))
            self._update_style_pref(profile.visual_style.mood, vs.get("mood", {}))

        if "composition" in updates:
            comp = updates["composition"]
            self._update_style_pref(profile.composition.density, comp.get("density", {}))
            self._update_style_pref(profile.composition.perspective, comp.get("perspective", {}))
            self._update_style_pref(profile.composition.foreground, comp.get("foreground", {}))

        if "domain" in updates:
            dom = updates["domain"]
            if "environments" in dom:
                self._update_list(profile.domain.environments, dom["environments"])
            if "avoid_environments" in dom:
                self._update_list(profile.domain.avoid_environments, dom["avoid_environments"])

        if "quality" in updates:
            qual = updates["quality"]
            if "must_have" in qual:
                self._update_list(profile.quality.must_have, qual["must_have"])
            if "never" in qual:
                self._update_list(profile.quality.never, qual["never"])

        profile.updated = datetime.now()
        return profile

    def _rating_description(self, rating: str) -> str:
        """Get human-readable rating description."""
        return {
            "++": "love it",
            "+": "like it",
            "-": "dislike it",
            "--": "hate it",
        }.get(rating, "neutral")

    def _update_style_pref(self, pref: StylePreference, update: dict):
        """Update a style preference with new values."""
        if not update:
            return

        if update.get("preference"):
            if pref.preference:
                # Append if not already present
                if update["preference"].lower() not in pref.preference.lower():
                    pref.preference = f"{pref.preference}, {update['preference']}"
            else:
                pref.preference = update["preference"]

        if update.get("avoid"):
            if pref.avoid:
                if update["avoid"].lower() not in pref.avoid.lower():
                    pref.avoid = f"{pref.avoid}, {update['avoid']}"
            else:
                pref.avoid = update["avoid"]

        if "confidence_delta" in update:
            pref.confidence = min(1.0, pref.confidence + update["confidence_delta"])

    def _update_list(self, current: list, updates: dict):
        """Update a list with add/remove operations."""
        if "add" in updates:
            for item in updates["add"]:
                if item and item not in current:
                    current.append(item)
        if "remove" in updates:
            for item in updates["remove"]:
                if item in current:
                    current.remove(item)


def enhance_prompt(prompt: str, profile: TasteProfile) -> str:
    """
    Enhance a generation prompt with taste profile preferences.

    Args:
        prompt: Original user prompt
        profile: User's taste profile

    Returns:
        Enhanced prompt with injected preferences
    """
    context = profile.to_prompt_context()
    if not context:
        return prompt

    # Build enhanced prompt with preferences
    parts = [prompt]

    # Add style preferences
    if profile.visual_style.lighting.preference:
        parts.append(f"with {profile.visual_style.lighting.preference} lighting")
    if profile.visual_style.color_palette.preference:
        parts.append(f"using {profile.visual_style.color_palette.preference} colors")
    if profile.visual_style.mood.preference:
        parts.append(f"with {profile.visual_style.mood.preference} mood")

    # Add composition preferences
    if profile.composition.perspective.preference:
        parts.append(f"from {profile.composition.perspective.preference} perspective")
    if profile.composition.density.preference:
        parts.append(f"with {profile.composition.density.preference} density")

    # Add quality requirements
    if profile.quality.must_have:
        parts.append(f"must include: {', '.join(profile.quality.must_have)}")

    # Build avoid list
    avoids = []
    if profile.visual_style.lighting.avoid:
        avoids.append(profile.visual_style.lighting.avoid)
    if profile.visual_style.color_palette.avoid:
        avoids.append(profile.visual_style.color_palette.avoid)
    if profile.quality.never:
        avoids.extend(profile.quality.never)

    enhanced = ", ".join(parts)
    if avoids:
        enhanced += f". Avoid: {', '.join(avoids)}"

    return enhanced
