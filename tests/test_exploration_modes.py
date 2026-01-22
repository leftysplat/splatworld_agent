"""Test that exploration modes affect variant generation prompts."""

from splatworld_agent.learning import (
    PromptAdapter,
    VARIANT_EXPLORE_WIDE_PROMPT,
    VARIANT_REFINE_NARROW_PROMPT,
)
from splatworld_agent.models import ExplorationMode

def test_system_prompts_differ():
    """Verify the two mode prompts have different instructions."""
    # Both should contain the base prompt
    assert "base prompt is SACRED" in VARIANT_EXPLORE_WIDE_PROMPT
    assert "base prompt is SACRED" in VARIANT_REFINE_NARROW_PROMPT
    
    # Explore mode should have exploration-specific instructions
    assert "EXPLORE WIDELY" in VARIANT_EXPLORE_WIDE_PROMPT
    assert "DIVERSE" in VARIANT_EXPLORE_WIDE_PROMPT
    assert "ORTHOGONAL" in VARIANT_EXPLORE_WIDE_PROMPT
    
    # Refine mode should have refinement-specific instructions
    assert "REFINE NARROWLY" in VARIANT_REFINE_NARROW_PROMPT
    assert "SMALL" in VARIANT_REFINE_NARROW_PROMPT
    assert "TARGETED" in VARIANT_REFINE_NARROW_PROMPT
    
    # They should be different prompts
    assert VARIANT_EXPLORE_WIDE_PROMPT != VARIANT_REFINE_NARROW_PROMPT
    
    print("PASS: System prompts are correctly different")


def test_adapter_selects_correct_prompt():
    """Verify PromptAdapter._get_system_prompt returns correct prompt for mode."""
    adapter = PromptAdapter()
    
    explore_prompt = adapter._get_system_prompt(ExplorationMode.EXPLORE_WIDE)
    refine_prompt = adapter._get_system_prompt(ExplorationMode.REFINE_NARROW)
    
    assert "EXPLORE WIDELY" in explore_prompt
    assert "REFINE NARROWLY" in refine_prompt
    assert explore_prompt != refine_prompt
    
    print("PASS: Adapter selects correct prompt based on mode")


if __name__ == "__main__":
    test_system_prompts_differ()
    test_adapter_selects_correct_prompt()
    print("\nAll exploration mode tests passed!")
