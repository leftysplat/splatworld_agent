"""
Learning synthesis for SplatWorld Agent.

Uses Claude to analyze feedback and extract preferences for the taste profile.
"""

import json
from datetime import datetime
from typing import Optional

from anthropic import Anthropic

from .models import TasteProfile, Feedback, Generation, StylePreference, PromptVariant


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


VARIANT_SYSTEM_PROMPT = """You are a prompt variant generator for a 3D scene generation system.
Your job is to create variations of user prompts that explore their preferences based on feedback patterns.

CRITICAL RULES:
1. The base prompt is SACRED - never remove or fundamentally change the user's core intent
2. Variants ENHANCE the base by adding details, not replacing concepts
3. Always explain your reasoning so the user understands why you made changes
4. Track exactly what you modified from the base prompt

You will receive:
1. A base prompt from the user
2. Recent feedback history (what they liked/disliked)
3. Their current taste profile

Generate a variant that:
- Keeps ALL elements from the base prompt
- Adds enhancements based on positive feedback patterns
- Avoids elements from negative feedback patterns
- Stays anchored to the original concept (no drift into unrelated ideas)

Respond with a JSON object:
{
    "variant_prompt": "The full enhanced prompt text",
    "modifications": ["added warm lighting", "enhanced with foggy atmosphere"],
    "reasoning": "Based on your preference for warm, moody scenes, I added...",
    "anchored_elements": ["forest", "sunset", "cabin"]
}

The "reasoning" field should be conversational and explain your thinking in 1-2 sentences.
Start with phrases like "You seem to like...", "Based on your preference for...", "Since you rated X highly..."
"""


class PromptAdapter:
    """Generates prompt variants anchored to user's base prompts.

    Implements:
    - ADAPT-02: Claude generates prompt variants anchored to user's base prompt
    - ADAPT-04: System shows reasoning ("You seem to like xyz components, so I'm generating...")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the prompt adapter.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        """
        self.client = Anthropic(api_key=api_key)

    def generate_variant(
        self,
        base_prompt: str,
        profile: TasteProfile,
        recent_feedback: Optional[list[tuple[Generation, Feedback]]] = None,
    ) -> PromptVariant:
        """
        Generate a variant of the base prompt anchored to user preferences.

        Args:
            base_prompt: The user's original prompt (sacred - not to be changed)
            profile: Current taste profile with learned preferences
            recent_feedback: Optional list of (generation, feedback) tuples for context

        Returns:
            PromptVariant with the enhanced prompt, modifications list, and reasoning
        """
        # Build feedback context
        feedback_context = self._build_feedback_context(recent_feedback)

        # Build profile context
        profile_context = profile.to_prompt_context() or "No preferences learned yet"

        user_message = f"""
## Base Prompt (DO NOT modify the core concept)
{base_prompt}

## Recent Feedback History
{feedback_context}

## Current Taste Profile
{profile_context}

Generate a variant that enhances this prompt based on the user's preferences.
Remember: Keep the base prompt's core concept intact, only ADD enhancements.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=VARIANT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Parse JSON response
        text = response.content[0].text

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())

        return PromptVariant(
            base_prompt=base_prompt,
            variant_prompt=result["variant_prompt"],
            modifications=result.get("modifications", []),
            reasoning=result.get("reasoning", ""),
            anchored_elements=result.get("anchored_elements", []),
        )

    def generate_variants(
        self,
        base_prompt: str,
        profile: TasteProfile,
        recent_feedback: Optional[list[tuple[Generation, Feedback]]] = None,
        count: int = 3,
    ) -> list[PromptVariant]:
        """
        Generate multiple variants exploring different aspects of preferences.

        Args:
            base_prompt: The user's original prompt
            profile: Current taste profile
            recent_feedback: Optional feedback context
            count: Number of variants to generate (default 3)

        Returns:
            List of PromptVariant objects
        """
        feedback_context = self._build_feedback_context(recent_feedback)
        profile_context = profile.to_prompt_context() or "No preferences learned yet"

        user_message = f"""
## Base Prompt (DO NOT modify the core concept)
{base_prompt}

## Recent Feedback History
{feedback_context}

## Current Taste Profile
{profile_context}

Generate {count} DIFFERENT variants that each explore different aspects of the user's preferences.
Each variant should enhance the base prompt in a unique way.

Respond with a JSON array:
[
    {{
        "variant_prompt": "...",
        "modifications": ["..."],
        "reasoning": "...",
        "anchored_elements": ["..."]
    }},
    ...
]
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=VARIANT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        results = json.loads(text.strip())

        # Handle both array and single object responses
        if isinstance(results, dict):
            results = [results]

        variants = []
        for result in results[:count]:
            variants.append(PromptVariant(
                base_prompt=base_prompt,
                variant_prompt=result["variant_prompt"],
                modifications=result.get("modifications", []),
                reasoning=result.get("reasoning", ""),
                anchored_elements=result.get("anchored_elements", []),
            ))

        return variants

    def _build_feedback_context(
        self,
        recent_feedback: Optional[list[tuple[Generation, Feedback]]],
    ) -> str:
        """Build context string from recent feedback."""
        if not recent_feedback:
            return "No recent feedback available"

        lines = []
        for gen, fb in recent_feedback[-10:]:  # Last 10 items
            rating_desc = {
                "++": "loved",
                "+": "liked",
                "-": "disliked",
                "--": "hated",
            }.get(fb.rating, "rated")

            lines.append(f"- {rating_desc}: \"{gen.prompt}\"")
            if fb.text:
                lines.append(f"  Comment: {fb.text}")

        return "\n".join(lines) if lines else "No recent feedback available"

    def explain_variant(self, variant: PromptVariant) -> str:
        """
        Generate a user-friendly explanation of the variant.

        Returns a formatted string showing the diff and reasoning.
        """
        parts = []

        # Header with diff summary
        parts.append(f"**{variant.get_diff_summary()}**")
        parts.append("")

        # Reasoning (conversational)
        if variant.reasoning:
            parts.append(variant.reasoning)
            parts.append("")

        # What was preserved (anchored elements)
        if variant.anchored_elements:
            preserved = ", ".join(variant.anchored_elements)
            parts.append(f"Preserved from base: {preserved}")

        # What was added
        if variant.modifications:
            parts.append("Modifications:")
            for mod in variant.modifications:
                parts.append(f"  + {mod}")

        return "\n".join(parts)
