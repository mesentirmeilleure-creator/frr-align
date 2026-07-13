"""Combined reward for FRR-aware GRPO local smoke tests."""

from __future__ import annotations

from frr_align.data.schema import PromptSample
from frr_align.rewards.frr_reward import frr_reward
from frr_align.rewards.grammar_reward import grammar_reward
from frr_align.rewards.safety_reward import safety_reward
from frr_align.weighting.frr_weight_engine import compute_weight


def combined_reward(prompt: str, response: str, sample: PromptSample, scale_by_weight: bool = True) -> float:
    """Combine safety, false-refusal, and French-quality rewards."""

    base = (
        1.0 * safety_reward(prompt, response, sample.expected)
        + 1.0 * frr_reward(response, sample.expected)
        + 0.3 * grammar_reward(response)
    )
    if not scale_by_weight:
        return round(base, 4)

    # Keep reward scaling bounded for GRPO stability.
    weight = min(3.0, max(1.0, compute_weight(prompt)))
    return round(base * weight, 4)
